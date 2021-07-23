/*
 * Avisynth wrapper for BM3DCUDA
 * Copyright (c) 2021 WolframRhodium
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "avisynth/avisynth.h"

using namespace std::string_literals;

extern cudaGraphExec_t get_graphexec(
    float * d_res, float * d_src, float * h_res, 
    int width, int height, int stride, 
    float sigma, int block_step, int bm_range, 
    int radius, int ps_num, int ps_range, 
    bool chroma, float sigma_u, float sigma_v, 
    bool final_, float extractor
) noexcept;

#define checkError(expr) do {                                                      \
    if (cudaError_t result = expr; result != cudaSuccess) [[unlikely]] {           \
        const char * error_str = cudaGetErrorString(result);                       \
        env->ThrowError(("BM3D: '"s + # expr + "' failed: " + error_str).c_str()); \
    }                                                                              \
} while(0)

constexpr int kFast = 4;

struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

template <typename T, auto deleter>
    requires 
        std::default_initializable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T>
struct Resource {
    T data;

    [[nodiscard]] constexpr Resource() noexcept = default;

    [[nodiscard]] constexpr Resource(T x) noexcept : data(x) {}

    [[nodiscard]] constexpr Resource(Resource&& other) noexcept 
        : data(std::exchange(other.data, T{})) 
    { }

    constexpr Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(data);
        data = std::exchange(other.data, T{});
        return *this;
    }

    Resource operator=(Resource other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T x) noexcept {
        if (x) {
            deleter(x);
        }
    }

    constexpr Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};

struct CUDA_Resource {
    Resource<float *, cudaFree> d_src;
    Resource<float *, cudaFree> d_res;
    Resource<float *, cudaFreeHost> h_res;
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs;
};

static inline void SpatialAggregation(
    float * __restrict dstp, 
    const float * __restrict h_res, 
    int width, int height, int s_stride, int d_stride
) noexcept {

    const float * wdst = h_res;
    const float * weight = &h_res[height * d_stride];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = wdst[x] / weight[x];
        }

        dstp += s_stride;
        wdst += d_stride;
        weight += d_stride;
    }
}

class BM3DFilter : public GenericVideoFilter {
    std::vector<int> planes_id;
    PClip ref_node;

    // stored in graphexec: 
    // float sigma[3];
    // int block_step[3];
    // int bm_range[3];
    // int ps_num[3];
    // int ps_range[3];
    // float extractor;

    int radius;
    int num_copy_engines; // fast
    bool chroma;
    bool process[3]; // sigma != 0
    bool final_;

    int d_pitch;
    int device_id;

    std::vector<CUDA_Resource> resources;

    ticket_semaphore semaphore;
    std::unique_ptr<std::atomic_flag[]> locks;    
    int lock() {
        semaphore.acquire();
        if (num_copy_engines > 1) {
            for (int i = 0; i < num_copy_engines; ++i) {
                if (!locks[i].test_and_set(std::memory_order::acquire)) {
                    return i;
                }
            }
            abort(); // impossible
        }
        return 0;
    }
    void unlock(int lock_idx) {
        semaphore.release();
        if (num_copy_engines > 1) {
            locks[lock_idx].clear(std::memory_order::release);
        }
    }

public:
    BM3DFilter(AVSValue args, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
    static AVSValue __cdecl Create(
        AVSValue args, void* user_data, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};

PVideoFrame __stdcall BM3DFilter::GetFrame(int n, IScriptEnvironment* env) {
    int temporal_width = 2 * radius + 1;
    int num_input_frames = temporal_width * (final_ ? 2 : 1); // including ref

    const std::vector srcs = [&](){
        std::vector<PVideoFrame> temp;

        temp.reserve(num_input_frames);

        if (final_) {
            for (int i = -radius; i <= radius; ++i) {
                int clamped_n = std::clamp(n + i, 0, vi.num_frames - 1);
                temp.emplace_back(ref_node->GetFrame(clamped_n, env));
            }
        }

        for (int i = -radius; i <= radius; ++i) {
            int clamped_n = std::clamp(n + i, 0, vi.num_frames - 1);
            temp.emplace_back(child->GetFrame(clamped_n, env));
        }

        return temp;
    }();

    const auto & src = srcs[radius + (final_ ? temporal_width : 0)];

    PVideoFrame dst = env->NewVideoFrameP(vi, const_cast<PVideoFrame *>(&src));

    int lock_idx = lock();

    checkError(cudaSetDevice(device_id));

    float * h_res = resources[lock_idx].h_res;

    const auto & stream = resources[lock_idx].stream;
    int d_stride = d_pitch / sizeof(float);

    int src_width = vi.width;
    int src_height = vi.height / (2 * (2 * radius + 1));

    if (chroma) {
        int width = src_width;
        int height = src_height;
        int s_pitch = src->GetRowSize();
        int s_stride = s_pitch / sizeof(float);
        int width_bytes = width * sizeof(float);

        const auto & graphexec = resources[lock_idx].graphexecs[0];

        float * h_src = h_res;
        for (int outer = 0; outer < (final_ ? 2 : 1); ++outer) {
            for (int plane = 0; plane < std::ssize(process); ++plane) {
                for (int j = 0; j < temporal_width; ++j) {
                    if (plane == 0 || process[plane]) {
                        const auto & current_src = srcs[j + outer * temporal_width];

                        env->BitBlt(
                            reinterpret_cast<BYTE *>(h_src), d_pitch, 
                            current_src->GetReadPtr(planes_id[plane]), s_pitch, 
                            width_bytes, height);
                    }
                    h_src += d_stride * height;
                }
            }
        }

        checkError(cudaGraphLaunch(graphexec, stream));

        checkError(cudaStreamSynchronize(stream));

        for (int plane = 0; plane < std::ssize(process); ++plane) {
            if (process[plane]) {
                BYTE * dstp = dst->GetWritePtr(planes_id[plane]);

                if (radius) {
                    env->BitBlt(
                        dstp, s_pitch, reinterpret_cast<const BYTE *>(h_res), d_pitch, 
                        width_bytes, height * 2 * temporal_width
                    );
                } else {
                    SpatialAggregation(
                        reinterpret_cast<float *>(dstp), h_res, 
                        width, height, s_stride, d_stride
                    );
                }
            }

            h_res += d_stride * height * 2 * temporal_width;
        }
    } else {
        for (int plane = 0; plane < planes_id.size(); plane++) {
            if (process[plane]) {
                int width = src->GetRowSize(planes_id[plane]) / sizeof(float);
                int height = src->GetHeight(planes_id[plane]);
                int s_pitch = src->GetPitch(planes_id[plane]);
                int s_stride = s_pitch / sizeof(float);
                int width_bytes = width * sizeof(float);

                const auto & graphexec = resources[lock_idx].graphexecs[plane];

                float * h_src = h_res;
                for (int i = 0; i < num_input_frames; ++i) {
                    env->BitBlt(
                        reinterpret_cast<BYTE *>(h_src), d_pitch, 
                        srcs[i]->GetReadPtr(planes_id[plane]), s_pitch, 
                        width_bytes, height
                    );
                    h_src += d_stride * height;
                }

                checkError(cudaGraphLaunch(graphexec, stream));

                checkError(cudaStreamSynchronize(stream));

                BYTE * dstp = dst->GetWritePtr(planes_id[plane]);

                if (radius) {
                    env->BitBlt(
                        dstp, s_pitch, reinterpret_cast<const BYTE *>(h_res), d_pitch, 
                        width_bytes, height * 2 * temporal_width
                    );
                } else {
                    SpatialAggregation(
                        reinterpret_cast<float *>(dstp), h_res, 
                        width, height, s_stride, d_stride
                    );
                }
            }
        }
    }

    unlock(lock_idx);
    
    return dst;
}

BM3DFilter::BM3DFilter(AVSValue args, IScriptEnvironment* env)
  : GenericVideoFilter(args[0].AsClip())
{
    env->CheckVersion(8);

    if (
        vi.BitsPerComponent() != 32 || 
        !vi.IsPlanar() || 
        !(vi.IsY() || vi.IsYUV() || vi.IsRGB())
    ) {
        env->ThrowError("BM3D: only 32bit float planar Y/YUV/RGB input supported");
    }

    const int src_width = vi.width;
    const int src_height = vi.height;

    if (vi.IsY()) {
        planes_id = { PLANAR_Y };
    } else if (vi.IsYUV()) {
        planes_id = { PLANAR_Y, PLANAR_U, PLANAR_V };
    } else if (vi.IsRGB()) {
        planes_id = { PLANAR_R, PLANAR_G, PLANAR_B };
    } else {
        env->ThrowError("BM3D: Unknown sample type");
    }

    if (args[1].Defined()) {
        ref_node = args[1].AsClip();
        const auto & ref_vi = ref_node->GetVideoInfo();
        if (
            ref_vi.width != src_width || ref_vi.height != src_height || 
            ref_vi.fps_numerator != vi.fps_numerator || 
            ref_vi.fps_denominator != vi.fps_denominator || 
            ref_vi.num_frames != vi.num_frames || 
            ref_vi.pixel_type != vi.pixel_type
        ) {
            env->ThrowError("BM3D: \"ref\" must be of the same format as \"clip\"");
        }
        final_ = true;
    } else {
        final_ = false;
    }

    auto array_loader = []<typename T>(const AVSValue & arg, T default_value) {
        std::array<T, 3> ret;
        if (!arg.Defined()) {
            ret.fill(default_value);
        } else if (arg.IsArray()) {
            int length = std::min(arg.ArraySize(), 3);
            for (int i = 0; i < length; ++i) {
                if constexpr (std::is_same_v<T, float>) {
                    ret[i] = arg[i].AsFloat();
                } else if (std::is_same_v<T, int>) {
                    ret[i] = arg[i].AsInt();
                }
            }
            for (int i = length; i < 3; ++i) {
                ret[i] = ret[i - 1];
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                ret.fill(arg.AsFloat(default_value));
            } else if (std::is_same_v<T, int>) {
                ret.fill(arg.AsInt(default_value));
            }
        }
        return ret;
    };

    std::array sigma = array_loader(args[2], 3.0f);
    for (int i = 0; i < std::ssize(sigma); ++i) {
        if (sigma[i] < 0.0f) {
            env->ThrowError("BM3D: \"sigma\" must be non-negative");
        }
        if (sigma[i] < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        } else {
            process[i] = true;

            // assumes grayscale input, hard_thr = 2.7
            sigma[i] *= (3.0f / 4.0f) / 255.0f * 64.0f * (final_ ? 1.0f : 2.7f);
        }
    }

    std::array block_step = array_loader(args[3], 8);
    for (const auto & x : block_step) {
        if (x <= 0 || x > 8) {
            env->ThrowError("BM3D: \"block_step\" must be in range [1, 8]");
        }
    }

    std::array bm_range = array_loader(args[4], 8);
    for (const auto & x : bm_range) {
        if (x <= 0 || x > 8) {
            env->ThrowError("BM3D: \"bm_range\" must be in range [1, 8]");
        }
    }

    radius = args[5].AsInt(0);
    if (radius < 0) {
        env->ThrowError("BM3D: \"radius\" must be non-negative");
    } else if (radius > 0) {
        vi.height *= 2 * (2 * radius + 1);
    }

    std::array ps_num = array_loader(args[6], 2);
    for (const auto & x : ps_num) {
        if (x <= 0 || x > 8) {
            env->ThrowError("BM3D: \"ps_num\" must be in range [1, 8]");
        }
    }

    std::array ps_range = array_loader(args[7], 4);
    for (const auto & x : ps_range) {
        if (x <= 0) {
            env->ThrowError("BM3D: \"ps_range\" must be positive");
        }
    }

    chroma = args[8].AsBool(false);
    if (chroma && vi.pixel_type != VideoInfo::CS_YUV444PS) {
        env->ThrowError("BM3D: clip format must be YUV444 when \"chroma\" is true");
    }

    device_id = args[9].AsInt(0);
    {
        int device_count;
        checkError(cudaGetDeviceCount(&device_count));
        if (0 <= device_id && device_id < device_count) {
            checkError(cudaSetDevice(device_id));
        } else {
            env->ThrowError((
                "BM3D: invalid device ID (" + std::to_string(device_id) + ")").c_str());
        }
    }

    bool fast = args[10].AsBool(true);
    num_copy_engines = fast ? kFast : 1;

    int extractor_exp = args[11].AsInt(0);
    float extractor = extractor_exp ? std::ldexp(1.0f, extractor_exp) : 0.0f;

    // GPU resource allocation
    {
        semaphore.current.store(num_copy_engines - 1, std::memory_order::relaxed);

        locks = std::make_unique<std::atomic_flag[]>(num_copy_engines);

        resources.reserve(num_copy_engines);

        int max_width { 
            !process[0] && vi.IsYUV() ? 
            src_width >> vi.GetPlaneWidthSubsampling(PLANAR_U) : 
            src_width
        };
        int max_height { 
            !process[0] && vi.IsYUV() ? 
            src_height >> vi.GetPlaneHeightSubsampling(PLANAR_U) : 
            src_height
        };

        int num_input_planes = chroma ? 3 : 1;
        int temporal_width = 2 * radius + 1;
        for (int i = 0; i < num_copy_engines; ++i) {
            Resource<float *, cudaFree> d_src {};
            if (i == 0) {
                size_t _d_pitch;
                checkError(cudaMallocPitch(
                    &d_src.data, &_d_pitch, max_width * sizeof(float), 
                    (final_ ? 2 : 1) * num_input_planes * temporal_width * max_height));
                d_pitch = static_cast<int>(_d_pitch);
            } else {
                checkError(cudaMalloc(&d_src.data, 
                    (final_ ? 2 : 1) * num_input_planes * temporal_width * max_height * d_pitch));
            }

            Resource<float *, cudaFree> d_res {};
            checkError(cudaMalloc(&d_res.data, 
                num_input_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<float *, cudaFreeHost> h_res {};
            checkError(cudaMallocHost(&h_res.data, 
                num_input_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<cudaStream_t, cudaStreamDestroy> stream {};
            checkError(cudaStreamCreateWithFlags(&stream.data, 
                cudaStreamNonBlocking));

            std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs {};
            if (chroma) {
                graphexecs[0] = get_graphexec(
                    d_res, d_src, h_res, 
                    src_width, src_height, d_pitch / sizeof(float), 
                    sigma[0], block_step[0], bm_range[0], 
                    radius, ps_num[0], ps_range[0], 
                    true, sigma[1], sigma[2], 
                    final_, extractor
                );
            } else {
                for (int plane = 0; plane < planes_id.size(); ++plane) {
                    if (process[plane]) {
                        int plane_width { 
                            vi.IsYUV() && plane != 0 ? 
                            src_width >> vi.GetPlaneWidthSubsampling(PLANAR_U) : 
                            src_width
                        };
                        int plane_height { 
                            vi.IsYUV() && plane != 0 ? 
                            src_height >> vi.GetPlaneHeightSubsampling(PLANAR_U) : 
                            src_height
                        };

                        graphexecs[plane] = get_graphexec(
                            d_res, d_src, h_res, 
                            plane_width, plane_height, d_pitch / sizeof(float), 
                            sigma[plane], block_step[plane], bm_range[plane], 
                            radius, ps_num[plane], ps_range[plane], 
                            false, 0.0f, 0.0f, 
                            final_, extractor
                        );
                    }
                }
            }

            resources.push_back(CUDA_Resource{
                .d_src = std::move(d_src), 
                .d_res = std::move(d_res), 
                .h_res = std::move(h_res), 
                .stream = std::move(stream), 
                .graphexecs = std::move(graphexecs)
            });
        }
    }
}

AVSValue __cdecl BM3DFilter::Create(
    AVSValue args, void* user_data, IScriptEnvironment* env
) {

    return new BM3DFilter(args, env);
}

class VAggregateFilter : public GenericVideoFilter {
    std::vector<int> planes_id;
    int radius;
    int num_planes;

public:
    VAggregateFilter(AVSValue args, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
    static AVSValue __cdecl Create(
        AVSValue args, void* user_data, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};

PVideoFrame __stdcall VAggregateFilter::GetFrame(int n, IScriptEnvironment* env) {
    PVideoFrame src = child->GetFrame(n, env);

    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    std::vector<float> buffer;
    buffer.reserve(2 * vi.height * vi.width);

    for (int plane = 0; plane < planes_id.size(); ++plane) {
        auto srcp = reinterpret_cast<const float *>(src->GetReadPtr(planes_id[plane]));
        int width = (
            vi.IsYUV() && plane != 0 ? 
            vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U) : 
            vi.width
        );
        int height = (
            vi.IsYUV() && plane != 0 ? 
            vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U) : 
            vi.height
        );
        int pitch = dst->GetPitch(planes_id[plane]);
        int stride = pitch / sizeof(float);

        std::fill(buffer.begin(), buffer.end(), 0);

        for (int i = 0; i < 2 * radius + 1; ++i) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    auto num = srcp[(i * 2) * height * stride + row * stride + col];
                    buffer[row * vi.width + col] += num;
                    auto den = srcp[(i * 2 + 1) * height * stride + row * stride + col];
                    buffer[vi.height * vi.width + row * vi.width + col] += den;
                }
            }
        }

        auto dstp = reinterpret_cast<float *>(dst->GetWritePtr(planes_id[plane]));
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                auto num = buffer[row * vi.width + col];
                auto den = buffer[vi.height * vi.width + row * vi.width + col];
                dstp[row * stride + col] = num / (den + std::numeric_limits<float>::epsilon());
            }
        }
    }

    return dst;
}

VAggregateFilter::VAggregateFilter(AVSValue args, IScriptEnvironment* env)
  : GenericVideoFilter(args[0].AsClip())
{
    env->CheckVersion(8);

    if (vi.BitsPerComponent() != 32 || !vi.IsPlanar() || (!vi.IsY() && !vi.IsYUV() && !vi.IsRGB())) {
        env->ThrowError("VAggregate: only 32bit float planar Y/YUV/RGB input supported");
    }

    if (vi.IsY()) {
        planes_id = { PLANAR_Y };
    } else if (vi.IsYUV()) {
        planes_id = { PLANAR_Y, PLANAR_U, PLANAR_V };
    } else if (vi.IsRGB()) {
        planes_id = { PLANAR_R, PLANAR_G, PLANAR_B };
    } else {
        env->ThrowError("VAggregate: Unknown sample type");
    }

    radius = args[1].AsInt(0);
    if (radius <= 0) {
        env->ThrowError("VAggregate: \"radius\" must be positive");
    }

    vi.height /= 2 * (2 * radius + 1);
}

AVSValue __cdecl VAggregateFilter::Create(
    AVSValue args, void* user_data, IScriptEnvironment* env
) {

    return new VAggregateFilter(args, env);
}

const AVS_Linkage *AVS_linkage {};

extern "C" __declspec(dllexport) 
const char* __stdcall AvisynthPluginInit3(
    IScriptEnvironment* env, const AVS_Linkage* const vectors
) {

    AVS_linkage = vectors;

    env->AddFunction("BM3DCUDA", 
        "c[ref]c"
        "[sigma]f[block_step]i[bm_range]"
        "i[radius]i[ps_num]i[ps_range]"
        "i[chroma]b[device_id]i[fast]b[extractor_exp]i"
        , BM3DFilter::Create, nullptr);

    env->AddFunction("VAggregate", "c[radius]i", VAggregateFilter::Create, nullptr);

   return "BM3D algorithm";
}
