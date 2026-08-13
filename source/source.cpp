/*
 * VapourSynth wrapper for BM3DCUDA
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
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <cuda_runtime.h>

#include <VapourSynth.h>
#include <VSHelper.h>

using namespace std::string_literals;

extern cudaGraphExec_t get_graphexec(
    float * d_res, float * d_src, float * h_res,
    int width, int height, int stride,
    float sigma, int block_step, int bm_range,
    int radius, int ps_num, int ps_range,
    bool chroma, float sigma_u, float sigma_v,
    bool final_, float extractor
) noexcept;

extern std::variant<cudaGraphExec_t, std::string> get_fused_graphexec(
    float * d_res, float * d_src, float * h_res,
    int * d_params, int * h_params,
    int width, int height, int stride,
    float sigma, int block_step, int bm_range,
    int radius, int ps_num, int ps_range,
    bool chroma, float sigma_u, float sigma_v,
    bool final_, float extractor
) noexcept;

extern std::variant<cudaGraphExec_t, std::string> get_rolling_graphexec(
    float * d_accum, float * d_scratch, float * d_src,
    float * h_src, float * h_output,
    int * d_params, int * h_params,
    int width, int height, int stride,
    const float sigma[3], const int block_step[3],
    const int bm_range[3], const int ps_num[3], const int ps_range[3],
    int radius, int chunk_size, int process_mask, int video_planes,
    int subsampling_w, int subsampling_h,
    bool chroma, bool final_, float extractor
) noexcept;

#define checkError(expr) do {                                            \
    if (cudaError_t result = expr; result != cudaSuccess) [[unlikely]] { \
        const char * error_str = cudaGetErrorString(result);             \
        return set_error("'"s + # expr + "' failed: " + error_str);      \
    }                                                                    \
} while(0)

#define PLUGIN_ID "com.wolframrhodium.bm3dcuda"
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

struct BM3DData {
    VSNodeRef * node;
    VSNodeRef * ref_node;
    const VSVideoInfo * vi;

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
    bool fused;
    bool zero_init;

    int d_pitch;
    size_t params_offset;
    int device_id;

    ticket_semaphore semaphore;
    std::vector<CUDA_Resource> resources;
    std::mutex resources_lock;
};

static inline void Aggregation(
    float * VS_RESTRICT dstp, int dst_stride,
    const float * VS_RESTRICT srcp, int src_stride,
    int width, int height
) noexcept {
    const float * wdst = srcp;
    const float * weight = &srcp[height * src_stride];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = wdst[x] / weight[x];
        }
        dstp += dst_stride;
        wdst += src_stride;
        weight += src_stride;
    }
}

static void VS_CC BM3DInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    BM3DData * d = static_cast<BM3DData *>(*instanceData);

    if (d->radius && !d->fused) {
        VSVideoInfo vi = *d->vi;
        vi.height *= 2 * (2 * d->radius + 1);
        vsapi->setVideoInfo(&vi, 1, node);
    } else {
        vsapi->setVideoInfo(d->vi, 1, node);
    }
}

static const VSFrameRef *VS_CC BM3DGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<BM3DData *>(*instanceData);

    if (activationReason == arInitial) {
        int64_t request_radius = d->fused ? 2LL * d->radius : d->radius;
        int start_frame = static_cast<int>(
            std::max<int64_t>(static_cast<int64_t>(n) - request_radius, 0));
        int end_frame = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(n) + request_radius, d->vi->numFrames - 1));

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        if (d->final_) {
            for (int i = start_frame; i <= end_frame; ++i) {
                vsapi->requestFrameFilter(i, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        if (auto error = cudaSetDevice(d->device_id); error != cudaSuccess) {
            vsapi->setFilterError(
                ("BM3D: "s + cudaGetErrorString(error)).c_str(),
                frameCtx
            );
            return nullptr;
        }

        int radius = d->radius;
        int temporal_width = 2 * radius + 1;
        bool final_ = d->final_;
        bool fused = d->fused;
        int64_t request_radius = fused ? 2LL * radius : radius;
        int start_frame = static_cast<int>(
            std::max<int64_t>(static_cast<int64_t>(n) - request_radius, 0));
        int end_frame = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(n) + request_radius, d->vi->numFrames - 1));
        int input_width = fused ? end_frame - start_frame + 1 : temporal_width;
        int num_input_frames = input_width * (final_ ? 2 : 1); // including ref

        using freeFrame_t = decltype(vsapi->freeFrame);
        const std::vector srcs = [&](){
            std::vector<std::unique_ptr<const VSFrameRef, const freeFrame_t &>> temp;

            temp.reserve(num_input_frames);

            if (final_) {
                for (int i = 0; i < input_width; ++i) {
                    int clamped_n = fused ? start_frame + i :
                        std::clamp(n - radius + i, 0, d->vi->numFrames - 1);
                    temp.emplace_back(
                        vsapi->getFrameFilter(clamped_n, d->ref_node, frameCtx),
                        vsapi->freeFrame
                    );
                }
            }

            for (int i = 0; i < input_width; ++i) {
                int clamped_n = fused ? start_frame + i :
                    std::clamp(n - radius + i, 0, d->vi->numFrames - 1);
                temp.emplace_back(
                    vsapi->getFrameFilter(clamped_n, d->node, frameCtx),
                    vsapi->freeFrame
                );
            }

            return temp;
        }();

        int center_index = fused ? n - start_frame : radius;
        const VSFrameRef * src = srcs[center_index + (final_ ? input_width : 0)].get();

        std::unique_ptr<VSFrameRef, const freeFrame_t &> dst { nullptr, vsapi->freeFrame };
        if (radius && !fused) {
            dst.reset(
                vsapi->newVideoFrame(
                    d->vi->format, d->vi->width,
                    d->vi->height * 2 * temporal_width,
                    src, core)
            );
            for (int i = 0; i < d->vi->format->numPlanes; ++i) {
                if (d->zero_init && !d->process[i]) {
                    auto ptr = vsapi->getWritePtr(dst.get(), i);
                    auto height = vsapi->getFrameHeight(dst.get(), i);
                    auto pitch = vsapi->getStride(dst.get(), i);
                    memset(ptr, 0, height * pitch);
                }
            }
        } else {
            const VSFrameRef * fr[] = {
                d->process[0] ? nullptr : src,
                d->process[1] ? nullptr : src,
                d->process[2] ? nullptr : src
            };
            const int pl[] = { 0, 1, 2 };

            dst.reset(
                vsapi->newVideoFrame2(
                    d->vi->format, d->vi->width,
                    d->vi->height, fr, pl, src, core)
            );
        }

        d->semaphore.acquire();
        d->resources_lock.lock();
        auto resource = std::move(d->resources.back());
        d->resources.pop_back();
        d->resources_lock.unlock();

        const auto set_error = [&](const std::string & error_message) {
            d->resources_lock.lock();
            d->resources.push_back(std::move(resource));
            d->resources_lock.unlock();
            d->semaphore.release();

            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);

            return nullptr;
        };

        float * const h_res = resource.h_res;
        cudaStream_t stream = resource.stream;
        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);

        if (d->chroma) {
            int width = vsapi->getFrameWidth(src, 0);
            int height = vsapi->getFrameHeight(src, 0);
            int s_pitch = vsapi->getStride(src, 0);
            int s_stride = s_pitch / sizeof(float);
            int width_bytes = width * sizeof(float);

            cudaGraphExec_t graphexec = resource.graphexecs[0];

            float * h_src = h_res;
            for (int outer = 0; outer < (final_ ? 2 : 1); ++outer) {
                for (int i = 0; i < std::ssize(d->process); ++i) {
                    for (int j = 0; j < input_width; ++j) {
                        if (i == 0 || d->process[i]) {
                            auto current_src = srcs[j + outer * input_width].get();

                            vs_bitblt(
                                h_src, d_pitch,
                                vsapi->getReadPtr(current_src, i), s_pitch,
                                width_bytes, height
                            );
                        }
                        h_src += d_stride * height;
                    }
                    if (fused) {
                        h_src += d_stride * height * (4 * radius + 1 - input_width);
                    }
                }
            }

            if (fused) {
                int * params = reinterpret_cast<int *>(
                    reinterpret_cast<char *>(h_res) + d->params_offset);
                params[0] = input_width;
                for (int center = 0; center < temporal_width; ++center) {
                    int center_frame = static_cast<int>(std::clamp<int64_t>(
                        static_cast<int64_t>(n) - radius + center,
                        0, d->vi->numFrames - 1));
                    params[1 + 2 * center] = center_frame - start_frame;
                    params[2 + 2 * center] = n - center_frame + radius;
                }
            }

            checkError(cudaGraphLaunch(graphexec, stream));

            checkError(cudaStreamSynchronize(stream));

            float * h_dst = h_res;
            for (int plane = 0; plane < std::ssize(d->process); ++plane) {
                if (!d->process[plane]) {
                    h_dst += d_stride * height * 2 * (fused ? 1 : temporal_width);
                    continue;
                }

                float * dstp = reinterpret_cast<float *>(
                    vsapi->getWritePtr(dst.get(), plane));

                if (radius && !fused) {
                    vs_bitblt(
                        dstp, s_pitch, h_dst, d_pitch,
                        width_bytes, height * 2 * temporal_width
                    );
                } else if (fused) {
                    vs_bitblt(dstp, s_pitch, h_dst, d_pitch, width_bytes, height);
                } else {
                    Aggregation(dstp, s_stride, h_dst, d_stride, width, height);
                }

                h_dst += d_stride * height * 2 * (fused ? 1 : temporal_width);
            }
        } else { // !d->chroma
            for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                if (!d->process[plane]) {
                    continue;
                }

                int width = vsapi->getFrameWidth(src, plane);
                int height = vsapi->getFrameHeight(src, plane);
                int s_pitch = vsapi->getStride(src, plane);
                int s_stride = s_pitch / sizeof(float);
                int width_bytes = width * sizeof(float);

                cudaGraphExec_t graphexec = resource.graphexecs[plane];

                float * h_src = h_res;
                for (int i = 0; i < num_input_frames; ++i) {
                    vs_bitblt(
                        h_src, d_pitch,
                        vsapi->getReadPtr(srcs[i].get(), plane), s_pitch,
                        width_bytes, height
                    );
                    h_src += d_stride * height;
                    if (fused && (i + 1) % input_width == 0) {
                        h_src += d_stride * height * (4 * radius + 1 - input_width);
                    }
                }

                if (fused) {
                    int * params = reinterpret_cast<int *>(
                        reinterpret_cast<char *>(h_res) + d->params_offset);
                    params[0] = input_width;
                    for (int center = 0; center < temporal_width; ++center) {
                        int center_frame = static_cast<int>(std::clamp<int64_t>(
                            static_cast<int64_t>(n) - radius + center,
                            0, d->vi->numFrames - 1));
                        params[1 + 2 * center] = center_frame - start_frame;
                        params[2 + 2 * center] = n - center_frame + radius;
                    }
                }

                checkError(cudaGraphLaunch(graphexec, stream));

                checkError(cudaStreamSynchronize(stream));

                float * dstp = reinterpret_cast<float *>(
                    vsapi->getWritePtr(dst.get(), plane));

                if (radius && !fused) {
                    vs_bitblt(
                        dstp, s_pitch, h_res, d_pitch,
                        width_bytes, height * 2 * temporal_width
                    );
                } else if (fused) {
                    vs_bitblt(dstp, s_pitch, h_res, d_pitch, width_bytes, height);
                } else {
                    Aggregation(dstp, s_stride, h_res, d_stride, width, height);
                }
            }
        }

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        if (radius && !fused) {
            VSMap * dst_prop { vsapi->getFramePropsRW(dst.get()) };

            vsapi->propSetInt(dst_prop, "BM3D_V_radius", d->radius, paReplace);

            int64_t process[3] { d->process[0], d->process[1], d->process[2] };
            vsapi->propSetIntArray(dst_prop, "BM3D_V_process", process, 3);
        }

        return dst.release();
    }

    return nullptr;
}

static void VS_CC BM3DFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<BM3DData *>(instanceData);

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->ref_node);

    cudaSetDevice(d->device_id);

    delete d;
}

static void BM3DCreateImpl(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi, bool fused
) noexcept {

    auto d { std::make_unique<BM3DData>() };

    const auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("BM3D: " + error_message).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->ref_node);
    };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);
    const int width = d->vi->width;
    const int height = d->vi->height;
    const int bits_per_sample = d->vi->format->bitsPerSample;

    if (
        !isConstantFormat(d->vi) || d->vi->format->sampleType == stInteger ||
        (d->vi->format->sampleType == stFloat && bits_per_sample != 32)) {
        return set_error("only constant format 32bit float input supported");
    }

    int error;

    d->ref_node = vsapi->propGetNode(in, "ref", 0, &error);
    bool final_;
    if (error) {
        d->ref_node = nullptr;
        final_ = false;
    } else {
        auto ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (ref_vi->format->id != d->vi->format->id) {
            return set_error("\"ref\" must be of the same format as \"clip\"");
        } else if (ref_vi->width != width || ref_vi->height != height ) {
            return set_error("\"ref\" must be of the same dimensions as \"clip\"");
        } else if (ref_vi->numFrames != d->vi->numFrames) {
            return set_error("\"ref\" must be of the same number of frames as \"clip\"");
        }

        final_ = true;
    }
    d->final_ = final_;
    d->fused = fused;

    float sigma[3];
    for (int i = 0; i < std::ssize(sigma); ++i) {
        sigma[i] = static_cast<float>(
            vsapi->propGetFloat(in, "sigma", i, &error));

        if (error) {
            sigma[i] = (i == 0) ? 3.0f : sigma[i - 1];
        } else if (sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be non-negative");
        }

        if (sigma[i] < std::numeric_limits<float>::epsilon()) {
            d->process[i] = false;
        } else {
            d->process[i] = true;
        }
    }
    for (int i = 0; i < std::ssize(sigma); ++i) {
        // assumes grayscale input, hard_thr = 2.7
        sigma[i] *= (3.0f / 4.0f) / 255.0f * 64.0f * (final_ ? 1.0f : 2.7f);
    }

    int block_step[3];
    for (int i = 0; i < std::ssize(block_step); ++i) {
        block_step[i] = int64ToIntS(
            vsapi->propGetInt(in, "block_step", i, &error));

        if (error) {
            block_step[i] = (i == 0) ? 8 : block_step[i - 1];
        } else if (block_step[i] <= 0 || block_step[i] > 8) {
            return set_error("\"block_step\" must be in range [1, 8]");
        }
    }

    int bm_range[3];
    for (int i = 0; i < std::ssize(bm_range); ++i) {
        bm_range[i] = int64ToIntS(
            vsapi->propGetInt(in, "bm_range", i, &error));

        if (error) {
            bm_range[i] = (i == 0) ? 9 : bm_range[i - 1];
        } else if (bm_range[i] <= 0) {
            return set_error("\"bm_range\" must be positive");
        }
    }

    const int radius = [&](){
        int temp = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &error));
        if (error) {
            return 0;
        }
        return temp;
    }();
    if (radius < 0) {
        return set_error("\"radius\" must be non-negative");
    }
    if (fused && radius > (std::numeric_limits<int>::max() - 1) / 4) {
        return set_error("\"radius\" is too large for fused temporal processing");
    }
    d->radius = radius;

    int ps_num[3];
    for (int i = 0; i < std::ssize(ps_num); ++i) {
        ps_num[i] = int64ToIntS(
            vsapi->propGetInt(in, "ps_num", i, &error));

        if (error) {
            ps_num[i] = (i == 0) ? 2 : ps_num[i - 1];
        } else if (ps_num[i] <= 0 || ps_num[i] > 8) {
            return set_error("\"ps_num\" must be in range [1, 8]");
        }
    }

    int ps_range[3];
    for (int i = 0; i < std::ssize(ps_range); ++i) {
        ps_range[i] = int64ToIntS(
            vsapi->propGetInt(in, "ps_range", i, &error));

        if (error) {
            ps_range[i] = (i == 0) ? 4 : ps_range[i - 1];
        } else if (ps_range[i] <= 0) {
            return set_error("\"ps_range\" must be positive");
        }
    }

    const bool chroma = [&](){
        bool temp = !!vsapi->propGetInt(in, "chroma", 0, &error);
        if (error) {
            return false;
        }
        return temp;
    }();
    if (chroma && d->vi->format->id != pfYUV444PS) {
        return set_error("clip format must be YUV444 when \"chroma\" is true");
    }
    d->chroma = chroma;

    const int device_id = [&](){
        int temp = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
        if (error) {
            return 0;
        }
        return temp;
    }();
    int device_count;
    checkError(cudaGetDeviceCount(&device_count));
    if (0 <= device_id && device_id < device_count) {
        checkError(cudaSetDevice(device_id));
    } else {
        return set_error("invalid device ID (" + std::to_string(device_id) + ")");
    }
    d->device_id = device_id;

    const bool fast = [&](){
        bool temp = !!vsapi->propGetInt(in, "fast", 0, &error);
        if (error) {
            return true;
        }
        return temp;
    }();
    const int num_copy_engines { fast ? kFast : 1 };
    d->num_copy_engines = num_copy_engines;

    const float extractor = [&](){
        int temp = int64ToIntS(vsapi->propGetInt(in, "extractor_exp", 0, &error));
        if (error) {
            return 0.0f;
        }
        return (temp ? std::ldexp(1.0f, temp) : 0.0f);
    }();

    d->zero_init = !!vsapi->propGetInt(in, "zero_init", 0, &error);
    if (error) {
        d->zero_init = true;
    }

    // GPU resource allocation
    {
        d->semaphore.current.store(num_copy_engines - 1, std::memory_order::relaxed);

        d->resources.reserve(num_copy_engines);

        int max_width { d->process[0] ? width : width >> d->vi->format->subSamplingW };
        int max_height { d->process[0] ? height : height >> d->vi->format->subSamplingH };

        const int num_planes { chroma ? 3 : 1 };
        const int temporal_width = 2 * radius + 1;
        const int source_temporal_width = fused ? 4 * radius + 1 : temporal_width;
        if (fused) {
            const size_t min_temporal_stride =
                static_cast<size_t>(max_width) * max_height;
            const size_t max_kernel_offset = std::numeric_limits<int>::max();
            if (
                min_temporal_stride > max_kernel_offset ||
                static_cast<size_t>(source_temporal_width) >
                    max_kernel_offset / min_temporal_stride ||
                static_cast<size_t>(num_planes) > max_kernel_offset /
                    (min_temporal_stride * source_temporal_width)) {
                return set_error("\"radius\" is too large for the clip dimensions");
            }
        }
        const size_t source_rows = static_cast<size_t>(final_ ? 2 : 1) *
            static_cast<size_t>(num_planes) * source_temporal_width * max_height;
        const size_t result_rows = static_cast<size_t>(num_planes) *
            temporal_width * 2 * max_height;
        const size_t buffer_rows = std::max(source_rows, result_rows);
        const size_t params_size = fused ?
            (1 + 2 * static_cast<size_t>(temporal_width)) * sizeof(int) : 0;
        const size_t min_pitch = static_cast<size_t>(max_width) * sizeof(float);
        const size_t params_rows = fused ?
            (params_size + min_pitch - 1) / min_pitch : 0;
        size_t d_pitch;
        int d_stride;
        for (int i = 0; i < num_copy_engines; ++i) {
            Resource<float *, cudaFree> d_src {};
            if (i == 0) {
                checkError(cudaMallocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float),
                    fused ? buffer_rows + params_rows : source_rows));
                if (d_pitch >
                    static_cast<size_t>(std::numeric_limits<int>::max())) {
                    return set_error("device pitch exceeds the supported range");
                }
                d_stride = static_cast<int>(d_pitch / sizeof(float));
                if (fused) {
                    const size_t temporal_stride =
                        static_cast<size_t>(max_height) * d_stride;
                    const size_t max_kernel_offset = std::numeric_limits<int>::max();
                    if (
                        temporal_stride > max_kernel_offset ||
                        static_cast<size_t>(source_temporal_width) >
                            max_kernel_offset / temporal_stride ||
                        static_cast<size_t>(num_planes) > max_kernel_offset /
                            (temporal_stride * source_temporal_width)) {
                        return set_error("\"radius\" is too large for the device pitch");
                    }
                }
                d->d_pitch = static_cast<int>(d_pitch);
                d->params_offset = buffer_rows * d_pitch;
            } else {
                checkError(cudaMalloc(&d_src.data,
                    fused ? buffer_rows * d_pitch + params_size : source_rows * d_pitch));
            }

            Resource<float *, cudaFree> d_res {};
            checkError(cudaMalloc(&d_res.data, result_rows * d_pitch));

            Resource<float *, cudaFreeHost> h_res {};
            checkError(cudaMallocHost(&h_res.data,
                buffer_rows * d_pitch + params_size));

            Resource<cudaStream_t, cudaStreamDestroy> stream {};
            checkError(cudaStreamCreateWithFlags(&stream.data,
                cudaStreamNonBlocking));

            std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs {};
            if (d->chroma) {
                if (fused) {
                    const auto result = get_fused_graphexec(
                        d_res, d_src, h_res,
                        reinterpret_cast<int *>(reinterpret_cast<char *>(d_src.data) + d->params_offset),
                        reinterpret_cast<int *>(reinterpret_cast<char *>(h_res.data) + d->params_offset),
                        width, height, d_stride,
                        sigma[0], block_step[0], bm_range[0],
                        radius, ps_num[0], ps_range[0],
                        true, sigma[1], sigma[2], final_, extractor);
                    if (std::holds_alternative<cudaGraphExec_t>(result)) {
                        graphexecs[0] = std::get<cudaGraphExec_t>(result);
                    } else {
                        return set_error(std::get<std::string>(result));
                    }
                } else {
                    graphexecs[0] = get_graphexec(
                        d_res, d_src, h_res,
                        width, height, d_stride,
                        sigma[0], block_step[0], bm_range[0],
                        radius, ps_num[0], ps_range[0],
                        true, sigma[1], sigma[2], final_, extractor);
                }
            } else {
                auto subsamplingW = d->vi->format->subSamplingW;
                auto subsamplingH = d->vi->format->subSamplingH;

                for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                    if (d->process[plane]) {
                        int plane_width { plane == 0 ? width : width >> subsamplingW };
                        int plane_height { plane == 0 ? height : height >> subsamplingH };

                        if (fused) {
                            const auto result = get_fused_graphexec(
                                d_res, d_src, h_res,
                                reinterpret_cast<int *>(reinterpret_cast<char *>(d_src.data) + d->params_offset),
                                reinterpret_cast<int *>(reinterpret_cast<char *>(h_res.data) + d->params_offset),
                                plane_width, plane_height, d_stride,
                                sigma[plane], block_step[plane], bm_range[plane],
                                radius, ps_num[plane], ps_range[plane],
                                false, 0.0f, 0.0f, final_, extractor);
                            if (std::holds_alternative<cudaGraphExec_t>(result)) {
                                graphexecs[plane] = std::get<cudaGraphExec_t>(result);
                            } else {
                                return set_error(std::get<std::string>(result));
                            }
                        } else {
                            graphexecs[plane] = get_graphexec(
                                d_res, d_src, h_res,
                                plane_width, plane_height, d_stride,
                                sigma[plane], block_step[plane], bm_range[plane],
                                radius, ps_num[plane], ps_range[plane],
                                false, 0.0f, 0.0f, final_, extractor);
                        }
                    }
                }
            }

            d->resources.push_back(CUDA_Resource{
                .d_src = std::move(d_src),
                .d_res = std::move(d_res),
                .h_res = std::move(h_res),
                .stream = std::move(stream),
                .graphexecs = std::move(graphexecs)
            });
        }
    }

    vsapi->createFilter(
        in, out, fused ? "BM3Dv2" : "BM3D",
        BM3DInit, BM3DGetFrame, BM3DFree,
        fmParallel, 0, d.release(), core
    );
}

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {
    BM3DCreateImpl(in, out, userData, core, vsapi, false);
}

struct RollingResource {
    Resource<float *, cudaFree> d_src;
    Resource<float *, cudaFree> d_scratch;
    Resource<float *, cudaFree> d_accum;
    Resource<float *, cudaFreeHost> h_src;
    Resource<float *, cudaFreeHost> h_output;
    Resource<int *, cudaFree> d_params;
    Resource<int *, cudaFreeHost> h_params;
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    Resource<cudaGraphExec_t, cudaGraphExecDestroy> graphexec;
};

struct RollingData {
    VSNodeRef * node {};
    VSNodeRef * ref_node {};
    const VSVideoInfo * vi {};

    int radius {};
    int chunk_size {};
    int device_id {};
    int d_pitch {};
    bool chroma {};
    bool final_ {};
    bool process[3] {};

    size_t source_rows {};
    size_t output_rows {};
    size_t output_plane_rows[3] {};
    size_t output_step_rows[3] {};
    RollingResource resource;

    mutable std::shared_mutex cache_lock;
    int cached_start { -1 };
    std::vector<const VSFrameRef *> cached_frames;
};

struct RollingRequest {
    int chunk_start;
};

static bool checked_mul(size_t lhs, size_t rhs, size_t & result) noexcept {
    if (lhs && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool checked_add(size_t lhs, size_t rhs, size_t & result) noexcept {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static void VS_CC RollingInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {
    auto d = static_cast<RollingData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef * rolling_cache_get(
    const RollingData * d, int n, const VSAPI * vsapi
) noexcept {
    std::shared_lock lock { d->cache_lock };
    const int offset = n - d->cached_start;
    if (offset < 0 || offset >= std::ssize(d->cached_frames)) {
        return nullptr;
    }
    return vsapi->cloneFrameRef(d->cached_frames[offset]);
}

static const VSFrameRef *VS_CC RollingGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {
    auto d = static_cast<RollingData *>(*instanceData);

    if (activationReason == arInitial) {
        const int chunk_start = n / d->chunk_size * d->chunk_size;
        *frameData = new RollingRequest { chunk_start };

        if (const VSFrameRef * cached = rolling_cache_get(d, n, vsapi)) {
            delete static_cast<RollingRequest *>(*frameData);
            *frameData = nullptr;
            return cached;
        }

        const int64_t first = std::max<int64_t>(
            static_cast<int64_t>(chunk_start) - 2LL * d->radius, 0);
        const int64_t last = std::min<int64_t>(
            static_cast<int64_t>(chunk_start) + d->chunk_size - 1 +
                2LL * d->radius,
            d->vi->numFrames - 1);
        for (int64_t frame = first; frame <= last; ++frame) {
            vsapi->requestFrameFilter(static_cast<int>(frame), d->node, frameCtx);
        }
        if (d->final_) {
            for (int64_t frame = first; frame <= last; ++frame) {
                vsapi->requestFrameFilter(
                    static_cast<int>(frame), d->ref_node, frameCtx);
            }
        }
        return nullptr;
    }

    if (activationReason == arError) {
        delete static_cast<RollingRequest *>(*frameData);
        *frameData = nullptr;
        return nullptr;
    }

    if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    std::unique_ptr<RollingRequest> request {
        static_cast<RollingRequest *>(*frameData)
    };
    *frameData = nullptr;

    if (const VSFrameRef * cached = rolling_cache_get(d, n, vsapi)) {
        return cached;
    }

    const auto set_error = [&](const std::string & error_message) {
        vsapi->setFilterError(
            ("BM3Dv2 rolling: " + error_message).c_str(), frameCtx);
        return static_cast<const VSFrameRef *>(nullptr);
    };

    if (cudaError_t result = cudaSetDevice(d->device_id); result != cudaSuccess) {
        return set_error(cudaGetErrorString(result));
    }

    const int chunk_start = request->chunk_start;
    const int valid_outputs = std::min(
        d->chunk_size, d->vi->numFrames - chunk_start);
    const int logical_first = chunk_start - 2 * d->radius;
    const int logical_source_width = d->chunk_size + 4 * d->radius;
    const int centers = d->chunk_size + 2 * d->radius;
    const int first_frame = std::max(logical_first, 0);
    const int last_frame = std::min(
        chunk_start + d->chunk_size - 1 + 2 * d->radius,
        d->vi->numFrames - 1);
    const int unique_frames = last_frame - first_frame + 1;

    using freeFrame_t = decltype(vsapi->freeFrame);
    using FramePtr = std::unique_ptr<const VSFrameRef, const freeFrame_t &>;
    std::vector<FramePtr> source_frames;
    std::vector<FramePtr> reference_frames;
    source_frames.reserve(unique_frames);
    reference_frames.reserve(d->final_ ? unique_frames : 0);
    for (int frame = first_frame; frame <= last_frame; ++frame) {
        source_frames.emplace_back(
            vsapi->getFrameFilter(frame, d->node, frameCtx), vsapi->freeFrame);
    }
    if (d->final_) {
        for (int frame = first_frame; frame <= last_frame; ++frame) {
            reference_frames.emplace_back(
                vsapi->getFrameFilter(frame, d->ref_node, frameCtx),
                vsapi->freeFrame);
        }
    }

    RollingResource & resource = d->resource;
    const int d_stride = d->d_pitch / sizeof(float);
    auto stage_plane = [&](
        float * & h_dst, const std::vector<FramePtr> & frames, int plane,
        int plane_height
    ) {
        const int width = vsapi->getFrameWidth(frames.front().get(), plane);
        const int source_pitch = vsapi->getStride(frames.front().get(), plane);
        for (int logical = 0; logical < logical_source_width; ++logical) {
            const int frame = std::clamp(
                logical_first + logical, 0, d->vi->numFrames - 1);
            const VSFrameRef * source = frames[frame - first_frame].get();
            vs_bitblt(
                h_dst, d->d_pitch, vsapi->getReadPtr(source, plane),
                source_pitch, width * sizeof(float), plane_height);
            h_dst += static_cast<size_t>(d_stride) * plane_height;
        }
    };

    if (d->source_rows) {
        float * h_dst = resource.h_src;
        if (d->chroma) {
            if (d->final_) {
                for (int plane = 0; plane < 3; ++plane) {
                    stage_plane(h_dst, reference_frames, plane, d->vi->height);
                }
            }
            for (int plane = 0; plane < 3; ++plane) {
                stage_plane(h_dst, source_frames, plane, d->vi->height);
            }
        } else {
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }
                const int plane_height = vsapi->getFrameHeight(
                    source_frames.front().get(), plane);
                if (d->final_) {
                    stage_plane(h_dst, reference_frames, plane, plane_height);
                }
                stage_plane(h_dst, source_frames, plane, plane_height);
            }
        }

        resource.h_params.data[0] = logical_source_width;
        for (int center = 0; center < centers; ++center) {
            const int frame = std::clamp(
                chunk_start - d->radius + center,
                0, d->vi->numFrames - 1);
            resource.h_params.data[1 + center] = frame - logical_first;
        }

        if (cudaError_t result = cudaGraphLaunch(
            resource.graphexec, resource.stream); result != cudaSuccess) {
            return set_error(cudaGetErrorString(result));
        }
        if (cudaError_t result = cudaStreamSynchronize(resource.stream);
            result != cudaSuccess) {
            return set_error(cudaGetErrorString(result));
        }
    }

    std::vector<const VSFrameRef *> new_cache;
    new_cache.reserve(valid_outputs);
    for (int output = 0; output < valid_outputs; ++output) {
        const VSFrameRef * source =
            source_frames[chunk_start + output - first_frame].get();
        const VSFrameRef * plane_sources[] {
            d->process[0] ? nullptr : source,
            d->process[1] ? nullptr : source,
            d->process[2] ? nullptr : source
        };
        constexpr int planes[] { 0, 1, 2 };
        VSFrameRef * destination = vsapi->newVideoFrame2(
            d->vi->format, d->vi->width, d->vi->height,
            plane_sources, planes, source, core);

        for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            const int plane_width = vsapi->getFrameWidth(source, plane);
            const int plane_height = vsapi->getFrameHeight(source, plane);
            const size_t row = d->output_plane_rows[plane] +
                static_cast<size_t>(output) * d->output_step_rows[plane];
            const float * h_source = resource.h_output.data + row * d_stride;
            vs_bitblt(
                vsapi->getWritePtr(destination, plane),
                vsapi->getStride(destination, plane),
                h_source, d->d_pitch,
                plane_width * sizeof(float), plane_height);
        }
        new_cache.push_back(destination);
    }

    std::vector<const VSFrameRef *> old_cache;
    {
        std::unique_lock lock { d->cache_lock };
        old_cache = std::move(d->cached_frames);
        d->cached_frames = std::move(new_cache);
        d->cached_start = chunk_start;
    }
    for (const VSFrameRef * frame : old_cache) {
        vsapi->freeFrame(frame);
    }
    return rolling_cache_get(d, n, vsapi);
}

static void VS_CC RollingFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {
    auto d = static_cast<RollingData *>(instanceData);
    std::vector<const VSFrameRef *> cached_frames;
    {
        std::unique_lock lock { d->cache_lock };
        cached_frames = std::move(d->cached_frames);
        d->cached_start = -1;
    }
    for (const VSFrameRef * frame : cached_frames) {
        vsapi->freeFrame(frame);
    }
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->ref_node);
    cudaSetDevice(d->device_id);
    delete d;
}

static void RollingCreate(
    const VSMap *in, VSMap *out, int chunk_size,
    VSCore *core, const VSAPI *vsapi
) noexcept {
    try {
    auto d = std::make_unique<RollingData>();

    const auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("BM3Dv2 rolling: " + error_message).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->ref_node);
    };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);
    const int width = d->vi->width;
    const int height = d->vi->height;
    if (
        !isConstantFormat(d->vi) || d->vi->format->sampleType == stInteger ||
        d->vi->format->bitsPerSample != 32) {
        return set_error("only constant format 32bit float input supported");
    }

    int error;
    d->ref_node = vsapi->propGetNode(in, "ref", 0, &error);
    if (error) {
        d->ref_node = nullptr;
        d->final_ = false;
    } else {
        const VSVideoInfo * ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (ref_vi->format->id != d->vi->format->id) {
            return set_error("\"ref\" must be of the same format as \"clip\"");
        }
        if (ref_vi->width != width || ref_vi->height != height) {
            return set_error("\"ref\" must be of the same dimensions as \"clip\"");
        }
        if (ref_vi->numFrames != d->vi->numFrames) {
            return set_error("\"ref\" must be of the same number of frames as \"clip\"");
        }
        d->final_ = true;
    }

    float sigma[3];
    for (int plane = 0; plane < 3; ++plane) {
        sigma[plane] = static_cast<float>(
            vsapi->propGetFloat(in, "sigma", plane, &error));
        if (error) {
            sigma[plane] = plane ? sigma[plane - 1] : 3.0f;
        } else if (sigma[plane] < 0.0f) {
            return set_error("\"sigma\" must be non-negative");
        }
        d->process[plane] = sigma[plane] >=
            std::numeric_limits<float>::epsilon();
    }
    for (int plane = 0; plane < 3; ++plane) {
        sigma[plane] *= (3.0f / 4.0f) / 255.0f * 64.0f *
            (d->final_ ? 1.0f : 2.7f);
    }

    int block_step[3];
    int bm_range[3];
    int ps_num[3];
    int ps_range[3];
    for (int plane = 0; plane < 3; ++plane) {
        block_step[plane] = int64ToIntS(
            vsapi->propGetInt(in, "block_step", plane, &error));
        if (error) {
            block_step[plane] = plane ? block_step[plane - 1] : 8;
        } else if (block_step[plane] <= 0 || block_step[plane] > 8) {
            return set_error("\"block_step\" must be in range [1, 8]");
        }

        bm_range[plane] = int64ToIntS(
            vsapi->propGetInt(in, "bm_range", plane, &error));
        if (error) {
            bm_range[plane] = plane ? bm_range[plane - 1] : 9;
        } else if (bm_range[plane] <= 0) {
            return set_error("\"bm_range\" must be positive");
        }

        ps_num[plane] = int64ToIntS(
            vsapi->propGetInt(in, "ps_num", plane, &error));
        if (error) {
            ps_num[plane] = plane ? ps_num[plane - 1] : 2;
        } else if (ps_num[plane] <= 0 || ps_num[plane] > 8) {
            return set_error("\"ps_num\" must be in range [1, 8]");
        }

        ps_range[plane] = int64ToIntS(
            vsapi->propGetInt(in, "ps_range", plane, &error));
        if (error) {
            ps_range[plane] = plane ? ps_range[plane - 1] : 4;
        } else if (ps_range[plane] <= 0) {
            return set_error("\"ps_range\" must be positive");
        }
    }

    d->radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &error));
    if (error) {
        d->radius = 0;
    }
    if (d->radius <= 0) {
        return set_error("\"radius\" must be positive");
    }
    if (d->radius >
        (std::numeric_limits<int>::max() - chunk_size) / 4) {
        return set_error("\"radius\" is too large for rolling temporal processing");
    }
    d->chunk_size = chunk_size;

    d->chroma = !!vsapi->propGetInt(in, "chroma", 0, &error);
    if (error) {
        d->chroma = false;
    }
    if (d->chroma && d->vi->format->id != pfYUV444PS) {
        return set_error("clip format must be YUV444 when \"chroma\" is true");
    }

    d->device_id = int64ToIntS(
        vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        d->device_id = 0;
    }
    int device_count;
    checkError(cudaGetDeviceCount(&device_count));
    if (d->device_id < 0 || d->device_id >= device_count) {
        return set_error(
            "invalid device ID (" + std::to_string(d->device_id) + ")");
    }
    checkError(cudaSetDevice(d->device_id));

    const float extractor = [&]() {
        const int exponent = int64ToIntS(
            vsapi->propGetInt(in, "extractor_exp", 0, &error));
        if (error) {
            return 0.0f;
        }
        return exponent ? std::ldexp(1.0f, exponent) : 0.0f;
    }();

    int process_mask = 0;
    for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
        process_mask |= static_cast<int>(d->process[plane]) << plane;
    }

    if (process_mask) {
        const int source_width = chunk_size + 4 * d->radius;
        const int temporal_width = 2 * d->radius + 1;
        const int clips = d->final_ ? 2 : 1;
        const int max_width = d->process[0] ? width :
            width >> d->vi->format->subSamplingW;
        const int max_height = d->process[0] ? height :
            height >> d->vi->format->subSamplingH;
        const int graph_planes = d->chroma ? 3 : 1;
        if (max_width > std::numeric_limits<int>::max() - 255) {
            return set_error("clip width exceeds CUDA grid limits");
        }
        if (max_height > 65535) {
            return set_error("clip height exceeds CUDA grid limits");
        }

        size_t min_temporal_stride;
        size_t max_source_offset;
        size_t max_scratch_offset;
        if (
            !checked_mul(static_cast<size_t>(max_width), max_height,
                min_temporal_stride) ||
            !checked_mul(min_temporal_stride, source_width,
                max_source_offset) ||
            !checked_mul(min_temporal_stride, temporal_width * 2LL,
                max_scratch_offset) ||
            !checked_mul(max_source_offset, graph_planes,
                max_source_offset) ||
            !checked_mul(max_scratch_offset, graph_planes,
                max_scratch_offset) ||
            max_source_offset > std::numeric_limits<int>::max() ||
            max_scratch_offset > std::numeric_limits<int>::max()) {
            return set_error("clip dimensions exceed CUDA indexing limits");
        }

        if (d->chroma) {
            size_t plane_source_rows;
            if (
                !checked_mul(static_cast<size_t>(source_width), height,
                    plane_source_rows) ||
                !checked_mul(plane_source_rows, clips * 3LL,
                    d->source_rows) ||
                !checked_mul(static_cast<size_t>(chunk_size), 6LL * height,
                d->output_rows)) {
                return set_error("rolling buffer size overflow");
            }
            for (int plane = 0; plane < 3; ++plane) {
                d->output_plane_rows[plane] =
                    static_cast<size_t>(plane) * 2 * height;
                d->output_step_rows[plane] = 6LL * height;
            }
        } else {
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }
                const int plane_height = plane ?
                    height >> d->vi->format->subSamplingH : height;
                size_t source_rows;
                size_t output_rows;
                size_t updated;
                if (
                    !checked_mul(static_cast<size_t>(clips) * source_width,
                        plane_height, source_rows) ||
                    !checked_mul(static_cast<size_t>(chunk_size) * 2,
                        plane_height, output_rows) ||
                    !checked_add(d->source_rows, source_rows, updated)) {
                    return set_error("rolling buffer size overflow");
                }
                d->source_rows = updated;
                d->output_plane_rows[plane] = d->output_rows;
                d->output_step_rows[plane] = 2LL * plane_height;
                if (!checked_add(d->output_rows, output_rows, updated)) {
                    return set_error("rolling buffer size overflow");
                }
                d->output_rows = updated;
            }
        }

        size_t scratch_rows;
        if (!checked_mul(
            static_cast<size_t>(graph_planes) * temporal_width * 2,
            max_height, scratch_rows)) {
            return set_error("rolling scratch size overflow");
        }

        size_t pitch;
        checkError(cudaMallocPitch(
            &d->resource.d_src.data, &pitch,
            static_cast<size_t>(max_width) * sizeof(float), d->source_rows));
        if (
            pitch > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            pitch % sizeof(float)) {
            return set_error("device pitch exceeds the supported range");
        }
        d->d_pitch = static_cast<int>(pitch);
        const size_t d_stride = pitch / sizeof(float);
        size_t temporal_stride;
        size_t source_offset;
        size_t scratch_offset;
        if (
            !checked_mul(d_stride, max_height, temporal_stride) ||
            !checked_mul(temporal_stride, source_width, source_offset) ||
            !checked_mul(temporal_stride, temporal_width * 2LL,
                scratch_offset) ||
            !checked_mul(source_offset, graph_planes, source_offset) ||
            !checked_mul(scratch_offset, graph_planes, scratch_offset) ||
            source_offset > std::numeric_limits<int>::max() ||
            scratch_offset > std::numeric_limits<int>::max()) {
            return set_error("device pitch exceeds CUDA indexing limits");
        }

        size_t source_bytes;
        size_t scratch_bytes;
        size_t output_bytes;
        if (
            !checked_mul(d->source_rows, pitch, source_bytes) ||
            !checked_mul(scratch_rows, pitch, scratch_bytes) ||
            !checked_mul(d->output_rows, pitch, output_bytes)) {
            return set_error("rolling allocation size overflow");
        }

        checkError(cudaMalloc(&d->resource.d_scratch.data, scratch_bytes));
        checkError(cudaMalloc(&d->resource.d_accum.data, output_bytes));
        checkError(cudaMallocHost(&d->resource.h_src.data, source_bytes));
        checkError(cudaMallocHost(&d->resource.h_output.data, output_bytes));

        const size_t params_count = 1 + static_cast<size_t>(chunk_size) +
            2 * static_cast<size_t>(d->radius);
        size_t params_bytes;
        if (!checked_mul(params_count, sizeof(int), params_bytes)) {
            return set_error("rolling parameter size overflow");
        }
        checkError(cudaMalloc(&d->resource.d_params.data, params_bytes));
        checkError(cudaMallocHost(&d->resource.h_params.data, params_bytes));
        checkError(cudaStreamCreateWithFlags(
            &d->resource.stream.data, cudaStreamNonBlocking));

        const auto graph = get_rolling_graphexec(
            d->resource.d_accum, d->resource.d_scratch,
            d->resource.d_src, d->resource.h_src, d->resource.h_output,
            d->resource.d_params, d->resource.h_params,
            width, height, static_cast<int>(d_stride),
            sigma, block_step, bm_range, ps_num, ps_range,
            d->radius, chunk_size, process_mask,
            d->vi->format->numPlanes,
            d->vi->format->subSamplingW, d->vi->format->subSamplingH,
            d->chroma, d->final_, extractor);
        if (std::holds_alternative<std::string>(graph)) {
            return set_error(std::get<std::string>(graph));
        }
        d->resource.graphexec = std::get<cudaGraphExec_t>(graph);
    }

    vsapi->createFilter(
        in, out, "BM3Dv2 rolling",
        RollingInit, RollingGetFrame, RollingFree,
        fmParallelRequests, 0, d.release(), core);
    } catch (const std::bad_alloc &) {
        vsapi->setError(out, "BM3Dv2 rolling: memory allocation failed");
    }
}

struct VAggregateData {
    VSNodeRef * node;

    VSNodeRef * src_node;
    const VSVideoInfo * src_vi;

    std::array<bool, 3> process; // sigma != 0

    int radius;

    std::unordered_map<std::thread::id, float *> buffer;
    std::shared_mutex buffer_lock;
};

static void VS_CC VAggregateInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) {

    VAggregateData * d = static_cast<VAggregateData *>(*instanceData);

    vsapi->setVideoInfo(d->src_vi, 1, node);
}

static const VSFrameRef *VS_CC VAggregateGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {

    auto * d = static_cast<VAggregateData *>(*instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->src_vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->src_node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src_frame = vsapi->getFrameFilter(n, d->src_node, frameCtx);

        std::vector<const VSFrameRef *> vbm3d_frames;
        vbm3d_frames.reserve(2 * d->radius + 1);
        for (int i = n - d->radius; i <= n + d->radius; ++i) {
            auto frame_id = std::clamp(i, 0, d->src_vi->numFrames - 1);
            vbm3d_frames.emplace_back(vsapi->getFrameFilter(frame_id, d->node, frameCtx));
        }

        float * buffer {};
        {
            const auto thread_id = std::this_thread::get_id();
            bool init = true;

            {
                std::shared_lock _ { d->buffer_lock };

                try {
                    const auto & const_buffer = d->buffer;
                    buffer = const_buffer.at(thread_id);
                } catch (const std::out_of_range &) {
                    init = false;
                }
            }

            if (!init) {
                assert(d->process[0] || d->src_vi->format->numPlanes > 1);

                const int max_width {
                    d->process[0] ?
                    vsapi->getFrameWidth(src_frame, 0) :
                    vsapi->getFrameWidth(src_frame, 1)
                };

                buffer = reinterpret_cast<float *>(std::malloc(2 * max_width * sizeof(float)));

                std::lock_guard _ { d->buffer_lock };
                d->buffer.emplace(thread_id, buffer);
            }
        }

        const VSFrameRef * fr[] {
            d->process[0] ? nullptr : src_frame,
            d->process[1] ? nullptr : src_frame,
            d->process[2] ? nullptr : src_frame
        };
        constexpr int pl[] { 0, 1, 2 };
        auto dst_frame = vsapi->newVideoFrame2(
            d->src_vi->format,
            d->src_vi->width, d->src_vi->height,
            fr, pl, src_frame, core);

        for (int plane = 0; plane < d->src_vi->format->numPlanes; ++plane) {
            if (d->process[plane]) {
                int plane_width = vsapi->getFrameWidth(src_frame, plane);
                int plane_height = vsapi->getFrameHeight(src_frame, plane);
                int plane_stride = vsapi->getStride(src_frame, plane) / sizeof(float);

                std::vector<const float *> srcps;
                srcps.reserve(2 * d->radius + 1);
                for (int i = 0; i < 2 * d->radius + 1; ++i) {
                    srcps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(vbm3d_frames[i], plane)));
                }

                auto dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst_frame, plane));

                for (int y = 0; y < plane_height; ++y) {
                    memset(buffer, 0, 2 * plane_width * sizeof(float));
                    for (int i = 0; i < 2 * d->radius + 1; ++i) {
                        const float * agg_src = srcps[i];
                        // bm3d.VAggregate implements zero padding in temporal dimension
                        // here we implements replication padding
                        agg_src += (
                            std::clamp(2 * d->radius - i, n - d->src_vi->numFrames + 1 + d->radius, n + d->radius)
                            * 2 * plane_height + y) * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[x] += agg_src[x];
                        }
                        agg_src += plane_height * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[plane_width + x] += agg_src[x];
                        }
                    }
                    for (int x = 0; x < plane_width; ++x) {
                        dstp[x] = buffer[x] / buffer[plane_width + x];
                    }
                    dstp += plane_stride;
                }
            }
        }

        for (const auto & frame : vbm3d_frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->freeFrame(src_frame);

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC VAggregateFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    VAggregateData * d = static_cast<VAggregateData *>(instanceData);

    for (const auto & [_, ptr] : d->buffer) {
        std::free(ptr);
    }

    vsapi->freeNode(d->src_node);
    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC VAggregateCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {

    auto d { std::make_unique<VAggregateData>() };

    const auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("VAggregate: " + error_message).c_str());
        if (d->src_node) {
            vsapi->freeNode(d->src_node);
        }
        if (d->node) {
            vsapi->freeNode(d->node);
        }
    };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(d->node);
    d->src_node = vsapi->propGetNode(in, "src", 0, nullptr);
    d->src_vi = vsapi->getVideoInfo(d->src_node);

    const int num_planes = d->src_vi->format->numPlanes;
    if (num_planes > static_cast<int>(d->process.size())) {
        return set_error("source clip has too many planes");
    }

    d->radius = (vi->height / d->src_vi->height - 2) / 4;

    d->process.fill(false);
    int num_planes_args = vsapi->propNumElements(in, "planes");
    for (int i = 0; i < num_planes_args; ++i) {
        int error;
        int plane = int64ToIntS(vsapi->propGetInt(in, "planes", i, &error));
        if (error) {
            return set_error("\"planes\" must contain only integers");
        }
        if (plane < 0 || plane >= num_planes) {
            return set_error("\"planes\" contains an out-of-range plane index");
        }
        if (d->process[plane]) {
            return set_error("\"planes\" contains a duplicate plane index");
        }
        d->process[plane] = true;
    }

    VSCoreInfo core_info;
    vsapi->getCoreInfo2(core, &core_info);
    d->buffer.reserve(core_info.numThreads);

    vsapi->createFilter(
        in, out, "VAggregate",
        VAggregateInit, VAggregateGetFrame, VAggregateFree,
        fmParallel, 0, d.release(), core);
}

static VSMap * copy_bm3d_args(
    const VSMap * in, const VSAPI * vsapi
) noexcept {
    VSMap * result = vsapi->createMap();
    for (int key_index = 0; key_index < vsapi->propNumKeys(in); ++key_index) {
        const char * key = vsapi->propGetKey(in, key_index);
        if (
            std::string_view { key } == "temporal_mode" ||
            std::string_view { key } == "rolling_chunk") {
            continue;
        }
        const int elements = vsapi->propNumElements(in, key);
        for (int index = 0; index < elements; ++index) {
            switch (vsapi->propGetType(in, key)) {
            case ptInt:
                vsapi->propSetInt(
                    result, key,
                    vsapi->propGetInt(in, key, index, nullptr), paAppend);
                break;
            case ptFloat:
                vsapi->propSetFloat(
                    result, key,
                    vsapi->propGetFloat(in, key, index, nullptr), paAppend);
                break;
            case ptNode: {
                VSNodeRef * node = vsapi->propGetNode(in, key, index, nullptr);
                vsapi->propSetNode(result, key, node, paAppend);
                vsapi->freeNode(node);
                break;
            }
            default:
                break;
            }
        }
    }
    return result;
}

static void VS_CC BM3Dv2Create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {
    int error;
    std::string temporal_mode { "fused" };
    if (vsapi->propNumElements(in, "temporal_mode") >= 0) {
        const char * value = vsapi->propGetData(
            in, "temporal_mode", 0, &error);
        const int size = vsapi->propGetDataSize(
            in, "temporal_mode", 0, &error);
        if (error) {
            vsapi->setError(out, "BM3Dv2: \"temporal_mode\" must be a string");
            return;
        }
        temporal_mode.assign(value, size);
    }
    if (
        temporal_mode != "fused" && temporal_mode != "legacy" &&
        temporal_mode != "rolling") {
        vsapi->setError(
            out,
            "BM3Dv2: \"temporal_mode\" must be one of fused, legacy, or rolling");
        return;
    }

    const bool chunk_supplied =
        vsapi->propNumElements(in, "rolling_chunk") >= 0;
    if (chunk_supplied && temporal_mode != "rolling") {
        vsapi->setError(
            out,
            "BM3Dv2: \"rolling_chunk\" is valid only when temporal_mode is rolling");
        return;
    }
    int rolling_chunk = 16;
    if (chunk_supplied) {
        rolling_chunk = int64ToIntS(
            vsapi->propGetInt(in, "rolling_chunk", 0, &error));
        if (error || rolling_chunk < 1 || rolling_chunk > 64) {
            vsapi->setError(
                out, "BM3Dv2: \"rolling_chunk\" must be in range [1, 64]");
            return;
        }
    }

    int radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &error));
    if (error) {
        radius = 0;
    }
    if (temporal_mode == "rolling" && radius <= 0) {
        vsapi->setError(
            out, "BM3Dv2: rolling temporal mode requires radius greater than zero");
        return;
    }

    std::array<bool, 3> process;
    process.fill(true);

    int num_sigma_args = vsapi->propNumElements(in, "sigma");
    for (int i = 0; i < std::min(3, num_sigma_args); ++i) {
        auto sigma = vsapi->propGetFloat(in, "sigma", i, nullptr);
        if (sigma < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        }
    }
    if (num_sigma_args > 0) { // num_sigma_args may be -1
        for (int i = num_sigma_args; i < 3; ++i) {
            process[i] = process[i - 1];
        }
    }

    bool skip = true;
    auto src = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto src_vi = vsapi->getVideoInfo(src);
    const int source_planes = src_vi->format->numPlanes;
    for (int i = 0; i < source_planes; ++i) {
        skip &= !process[i];
    }
    if (skip && temporal_mode != "rolling") {
        vsapi->propSetNode(out, "clip", src, paReplace);
        vsapi->freeNode(src);
        return ;
    }

    if (radius > 0) {
        vsapi->freeNode(src);
        if (temporal_mode == "fused") {
            BM3DCreateImpl(in, out, userData, core, vsapi, true);
        } else if (temporal_mode == "rolling") {
            RollingCreate(in, out, rolling_chunk, core, vsapi);
        } else {
            auto plugin = vsapi->getPluginById(PLUGIN_ID, core);
            VSMap * bm3d_in = copy_bm3d_args(in, vsapi);

            auto map = vsapi->invoke(plugin, "BM3D", bm3d_in);
            vsapi->freeMap(bm3d_in);
            if (auto invoke_error = vsapi->getError(map); invoke_error) {
                vsapi->setError(out, invoke_error);
                vsapi->freeMap(map);
                return;
            }

            VSNodeRef * original = vsapi->propGetNode(in, "clip", 0, nullptr);
            vsapi->propSetNode(map, "src", original, paReplace);
            vsapi->freeNode(original);
            for (int plane = 0; plane < source_planes; ++plane) {
                if (process[plane]) {
                    vsapi->propSetInt(map, "planes", plane, paAppend);
                }
            }

            auto aggregate = vsapi->invoke(plugin, "VAggregate", map);
            vsapi->freeMap(map);
            if (auto invoke_error = vsapi->getError(aggregate); invoke_error) {
                vsapi->setError(out, invoke_error);
                vsapi->freeMap(aggregate);
                return;
            }
            auto node = vsapi->propGetNode(aggregate, "clip", 0, nullptr);
            vsapi->freeMap(aggregate);
            vsapi->propSetNode(out, "clip", node, paReplace);
            vsapi->freeNode(node);
        }
        return;
    }

    auto plugin = vsapi->getPluginById(PLUGIN_ID, core);
    VSMap * bm3d_in = copy_bm3d_args(in, vsapi);
    auto map = vsapi->invoke(plugin, "BM3D", bm3d_in);
    vsapi->freeMap(bm3d_in);
    if (auto error = vsapi->getError(map); error) {
        vsapi->setError(out, error);
        vsapi->freeMap(map);
        vsapi->freeNode(src);
        return ;
    }

    if (radius == 0) {
        // spatial BM3D should handle everything itself
        auto node = vsapi->propGetNode(map, "clip", 0, nullptr);
        vsapi->freeMap(map);
        vsapi->propSetNode(out, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->freeNode(src);
        return ;
    }

    vsapi->propSetNode(map, "src", src, paReplace);
    vsapi->freeNode(src);

    for (int i = 0; i < 3; ++i) {
        if (process[i]) {
            vsapi->propSetInt(map, "planes", i, paAppend);
        }
    }

    auto map2 = vsapi->invoke(plugin, "VAggregate", map);
    vsapi->freeMap(map);
    if (auto error = vsapi->getError(map2); error) {
        vsapi->setError(out, error);
        vsapi->freeMap(map2);
        return ;
    }

    auto node = vsapi->propGetNode(map2, "clip", 0, nullptr);
    vsapi->freeMap(map2);
    vsapi->propSetNode(out, "clip", node, paReplace);
    vsapi->freeNode(node);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) {

    configFunc(
        PLUGIN_ID, "bm3dcuda",
        "BM3D algorithm implemented in CUDA",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    constexpr auto bm3d_args {
        "clip:clip;"
        "ref:clip:opt;"
        "sigma:float[]:opt;"
        "block_step:int[]:opt;"
        "bm_range:int[]:opt;"
        "radius:int:opt;"
        "ps_num:int[]:opt;"
        "ps_range:int[]:opt;"
        "chroma:int:opt;"
        "device_id:int:opt;"
        "fast:int:opt;"
        "extractor_exp:int:opt;"
        "zero_init:int:opt;"
    };

    registerFunc("BM3D", bm3d_args, BM3DCreate, nullptr, plugin);

    registerFunc(
        "VAggregate",
        "clip:clip;"
        "src:clip;"
        "planes:int[];",
        VAggregateCreate, nullptr, plugin);

    constexpr auto bm3dv2_args {
        "clip:clip;"
        "ref:clip:opt;"
        "sigma:float[]:opt;"
        "block_step:int[]:opt;"
        "bm_range:int[]:opt;"
        "radius:int:opt;"
        "ps_num:int[]:opt;"
        "ps_range:int[]:opt;"
        "chroma:int:opt;"
        "device_id:int:opt;"
        "fast:int:opt;"
        "extractor_exp:int:opt;"
        "zero_init:int:opt;"
        "temporal_mode:data:opt;"
        "rolling_chunk:int:opt;"
    };
    registerFunc("BM3Dv2", bm3dv2_args, BM3Dv2Create, nullptr, plugin);
}
