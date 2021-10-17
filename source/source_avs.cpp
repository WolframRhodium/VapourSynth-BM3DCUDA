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

#include <avisynth.h>

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

constexpr int kNumStreams = 4;

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

static inline void Aggregation(
    float * __restrict dstp, int d_stride,
    const float * __restrict srcp, int s_stride,
    int width, int height
) noexcept {

    const float * wdst = srcp;
    const float * weight = &srcp[height * s_stride];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = wdst[x] / weight[x];
        }

        dstp += d_stride;
        wdst += s_stride;
        weight += s_stride;
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
    int num_streams; // fast
    bool chroma;
    bool process[3]; // sigma != 0

    int d_pitch;
    int device_id;

    ticket_semaphore semaphore;

    std::vector<CUDA_Resource> resources;
    std::mutex resources_lock;

    // final estimation
    bool final_() const {
        return static_cast<bool>(ref_node);
    }

    CUDA_Resource acquire() {
        semaphore.acquire();
        resources_lock.lock();
        auto resource = std::move(resources.back());
        resources.pop_back();
        resources_lock.unlock();
        return resource;
    }

    void release(CUDA_Resource && resource) {
        resources_lock.lock();
        resources.push_back(std::move(resource));
        resources_lock.unlock();
        semaphore.release();
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
    int num_input_frames = temporal_width * (final_() ? 2 : 1); // including ref

    const std::vector srcs = [&](){
        std::vector<PVideoFrame> temp;

        temp.reserve(num_input_frames);

        if (final_()) {
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

    const auto & src = srcs[radius + (final_() ? temporal_width : 0)];

    PVideoFrame dst = env->NewVideoFrameP(vi, const_cast<PVideoFrame *>(&src));

    auto resource = acquire();

    checkError(cudaSetDevice(device_id));

    float * h_res = resource.h_res;

    const auto & stream = resource.stream;
    int d_stride = d_pitch / sizeof(float);

    if (chroma) {
        // planar YUV444PS input
        int width_bytes = src->GetRowSize(PLANAR_Y);
        int width = width_bytes / sizeof(float);
        int height = src->GetHeight(PLANAR_Y);

        float * h_src = h_res;
        for (int outer = 0; outer < (final_() ? 2 : 1); ++outer) {
            for (int plane = 0; plane < std::ssize(process); ++plane) {
                for (int j = 0; j < temporal_width; ++j) {
                    if (plane == 0 || process[plane]) {
                        const auto & current_src = srcs[j + outer * temporal_width];
                        auto srcp = current_src->GetReadPtr(planes_id[plane]);
                        int src_pitch = current_src->GetPitch(planes_id[plane]);

                        env->BitBlt(
                            reinterpret_cast<BYTE *>(h_src), d_pitch,
                            srcp, src_pitch,
                            width_bytes, height);
                    }
                    h_src += d_stride * height;
                }
            }
        }

        const auto & graphexec = resource.graphexecs[0];
        checkError(cudaGraphLaunch(graphexec, stream));
        checkError(cudaStreamSynchronize(stream));

        for (int plane = 0; plane < std::ssize(process); ++plane) {
            if (!process[plane]) {
                continue;
            }

            BYTE * dstp = dst->GetWritePtr(planes_id[plane]);
            int dst_pitch = dst->GetPitch(planes_id[plane]);
            int dst_stride = dst_pitch / sizeof(float);

            if (radius) {
                env->BitBlt(
                    dstp, dst_pitch,
                    reinterpret_cast<const BYTE *>(h_res), d_pitch,
                    width_bytes, height * 2 * temporal_width
                );
            } else {
                Aggregation(
                    reinterpret_cast<float *>(dstp), dst_stride,
                    h_res, d_stride,
                    width, height
                );
            }

            h_res += d_stride * height * 2 * temporal_width;
        }
    } else {
        for (unsigned plane = 0; plane < planes_id.size(); plane++) {
            if (!process[plane]) {
                continue;
            }

            int width_bytes = src->GetRowSize(planes_id[plane]);
            int width = width_bytes / sizeof(float);
            int height = src->GetHeight(planes_id[plane]);

            float * h_src = h_res;
            for (int i = 0; i < num_input_frames; ++i) {
                const auto & current_src = srcs[i];
                auto srcp = current_src->GetReadPtr(planes_id[plane]);
                int src_pitch = current_src->GetPitch(planes_id[plane]);

                env->BitBlt(
                    reinterpret_cast<BYTE *>(h_src), d_pitch,
                    srcp, src_pitch,
                    width_bytes, height
                );

                h_src += d_stride * height;
            }

            const auto & graphexec = resource.graphexecs[plane];
            checkError(cudaGraphLaunch(graphexec, stream));
            checkError(cudaStreamSynchronize(stream));

            BYTE * dstp = dst->GetWritePtr(planes_id[plane]);
            int dst_pitch = dst->GetPitch(planes_id[plane]);
            int dst_stride = dst_pitch / sizeof(float);
            if (radius) {
                env->BitBlt(
                    dstp, dst_pitch,
                    reinterpret_cast<const BYTE *>(h_res), d_pitch,
                    width_bytes, height * 2 * temporal_width
                );
            } else {
                Aggregation(
                    reinterpret_cast<float *>(dstp), dst_stride,
                    h_res, d_stride,
                    width, height
                );
            }
        }
    }

    release(std::move(resource));

    return dst;
}

BM3DFilter::BM3DFilter(AVSValue args, IScriptEnvironment* env)
  : GenericVideoFilter(args[0].AsClip()), ref_node()
{
    env->CheckVersion(8);

    if (
        vi.BitsPerComponent() != 32 ||
        !vi.IsPlanar() ||
        !(vi.IsY() || vi.IsYUV() || vi.IsRGB())
    ) {
        env->ThrowError("BM3D_CUDA: only 32bit float planar Y/YUV/RGB input supported");
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
        env->ThrowError("BM3D_CUDA: Unknown sample type");
    }

    if (args[1].Defined()) {
        ref_node = args[1].AsClip();
        if (
            const auto & ref_vi = ref_node->GetVideoInfo();
            ref_vi.width != src_width || ref_vi.height != src_height ||
            ref_vi.fps_numerator != vi.fps_numerator ||
            ref_vi.fps_denominator != vi.fps_denominator ||
            ref_vi.num_frames != vi.num_frames ||
            ref_vi.pixel_type != vi.pixel_type
        ) {
            env->ThrowError("BM3D_CUDA: \"ref\" must be of the same format as \"clip\"");
        }
    }

    auto array_loader = [](const AVSValue & arg, const auto default_value) {
        using T = std::remove_const_t<decltype(default_value)>;
        std::array<T, 3> ret;
        if (!arg.Defined() || arg.ArraySize() == 0) {
            ret.fill(default_value);
        } else {
            int length = std::min(arg.ArraySize(), 3);
            for (int i = 0; i < length; ++i) {
                if constexpr (std::is_same_v<T, float>) {
                    ret[i] = static_cast<float>(arg[i].AsFloat());
                } else if (std::is_same_v<T, int>) {
                    ret[i] = arg[i].AsInt();
                }
            }
            for (int i = length; i < 3; ++i) {
                ret[i] = ret[i - 1];
            }
        }
        return ret;
    };

    std::array sigma = array_loader(args[2], 3.0f);
    for (int i = 0; i < std::ssize(sigma); ++i) {
        if (sigma[i] < 0.0f) {
            env->ThrowError("BM3D_CUDA: \"sigma\" must be non-negative");
        }
        if (sigma[i] < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        } else {
            process[i] = true;

            // assumes grayscale input, hard_thr = 2.7
            sigma[i] *= (3.0f / 4.0f) / 255.0f * 64.0f * (final_() ? 1.0f : 2.7f);
        }
    }

    std::array block_step = array_loader(args[3], 8);
    for (const auto & x : block_step) {
        if (x <= 0 || x > 8) {
            env->ThrowError("BM3D_CUDA: \"block_step\" must be in range [1, 8]");
        }
    }

    std::array bm_range = array_loader(args[4], 9);
    for (const auto & x : bm_range) {
        if (x <= 0) {
            env->ThrowError("BM3D_CUDA: \"bm_range\" must be positive");
        }
    }

    radius = args[5].AsInt(0);
    if (radius < 0) {
        env->ThrowError("BM3D_CUDA: \"radius\" must be non-negative");
    } else if (radius > 0) {
        vi.height *= 2 * (2 * radius + 1);
    }

    std::array ps_num = array_loader(args[6], 2);
    for (const auto & x : ps_num) {
        if (x <= 0) {
            env->ThrowError("BM3D_CUDA: \"ps_num\" must be positive");
        }
    }

    std::array ps_range = array_loader(args[7], 4);
    for (const auto & x : ps_range) {
        if (x <= 0) {
            env->ThrowError("BM3D_CUDA: \"ps_range\" must be positive");
        }
    }

    chroma = args[8].AsBool(false);
    if (chroma && vi.pixel_type != VideoInfo::CS_YUV444PS) {
        env->ThrowError("BM3D_CUDA: clip format must be YUV444PS when \"chroma\" is true");
    }

    device_id = args[9].AsInt(0);
    {
        int device_count;
        checkError(cudaGetDeviceCount(&device_count));
        if (0 <= device_id && device_id < device_count) {
            checkError(cudaSetDevice(device_id));
        } else {
            env->ThrowError(
                ("BM3D_CUDA: invalid device ID (" + std::to_string(device_id) + ")").c_str()
            );
        }
    }

    bool fast = args[10].AsBool(true);
    num_streams = fast ? kNumStreams : 1;

    int extractor_exp = args[11].AsInt(0);
    float extractor = extractor_exp ? std::ldexp(1.0f, extractor_exp) : 0.0f;

    // GPU resource allocation
    {
        semaphore.current.store(num_streams - 1, std::memory_order::relaxed);

        resources.reserve(num_streams);

        int max_width {
            vi.IsYUV() && !process[0] ?
            src_width >> vi.GetPlaneWidthSubsampling(PLANAR_U) :
            src_width
        };
        int max_height {
            vi.IsYUV() && !process[0] ?
            src_height >> vi.GetPlaneHeightSubsampling(PLANAR_U) :
            src_height
        };

        int num_input_planes = chroma ? 3 : 1;
        int temporal_width = 2 * radius + 1;
        for (int i = 0; i < num_streams; ++i) {
            Resource<float *, cudaFree> d_src {};
            if (i == 0) {
                size_t _d_pitch;
                checkError(cudaMallocPitch(
                    &d_src.data, &_d_pitch, max_width * sizeof(float),
                    (final_() ? 2 : 1) * num_input_planes * temporal_width * max_height));
                d_pitch = static_cast<int>(_d_pitch);
            } else {
                checkError(cudaMalloc(&d_src.data,
                    (final_() ? 2 : 1) * num_input_planes * temporal_width * max_height * d_pitch));
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
                    final_(), extractor
                );
            } else {
                for (unsigned plane = 0; plane < planes_id.size(); ++plane) {
                    if (!process[plane]) {
                        continue;
                    }

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
                        final_(), extractor
                    );
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

const AVS_Linkage *AVS_linkage {};

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(
    IScriptEnvironment* env, const AVS_Linkage* const vectors
) {

    AVS_linkage = vectors;

    env->AddFunction("BM3D_CUDA",
        "c[ref]c"
        "[sigma]f+[block_step]i+[bm_range]i+"
        "[radius]i[ps_num]i+[ps_range]i+"
        "[chroma]b[device_id]i[fast]b[extractor_exp]i"
        , BM3DFilter::Create, nullptr);

   return "BM3D algorithm";
}
