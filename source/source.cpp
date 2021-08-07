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
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>

using namespace std::string_literals;

extern cudaGraphExec_t get_graphexec(
    float * d_res, float * d_src, float * h_res,
    int width, int height, int stride,
    float sigma, int block_step, int bm_range,
    int radius, int ps_num, int ps_range,
    bool chroma, float sigma_u, float sigma_v,
    bool final_, float extractor
) noexcept;

#define PLUGIN_ID "com.wolframrhodium.bm3dcuda"

#define checkError(expr) do {                                            \
    if (cudaError_t result = expr; result != cudaSuccess) [[unlikely]] { \
        const char * error_str = cudaGetErrorString(result);             \
        return set_error("'"s + # expr + "' failed: " + error_str);      \
    }                                                                    \
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

    int d_pitch;
    int device_id;

    ticket_semaphore semaphore;
    std::unique_ptr<std::atomic_flag[]> resource_locks;
    std::vector<CUDA_Resource> resources;

    bool unsafe;
    // only used by unsafe=true:
    std::unique_ptr<std::atomic_flag[]> frame_locks; // write lock
    VSNodeRef * buffer_node;
};

static inline void Accumulation(
    float * VS_RESTRICT dstp, int dst_stride,
    const float * VS_RESTRICT srcp, int src_stride,
    int width, int height
) noexcept {

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] += srcp[x];
        }

        dstp += dst_stride;
        srcp += src_stride;
    }
}

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

    if (d->radius) {
        VSVideoInfo vi = *d->vi;
        if (d->unsafe) {
            // returns a signal
            vi.format = vsapi->getFormatPreset(pfGray8, core);
            vi.width = 1;
            vi.height = 1;
        } else {
            vi.height *= 2 * (2 * d->radius + 1);
        }
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
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        if (d->final_) {
            for (int i = start_frame; i <= end_frame; ++i) {
                vsapi->requestFrameFilter(i, d->ref_node, frameCtx);
            }
        }
        if (d->radius && d->unsafe) {
            for (int i = start_frame; i <= end_frame; ++i) {
                vsapi->requestFrameFilter(i, d->buffer_node, frameCtx);
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
        int num_input_frames = temporal_width * (final_ ? 2 : 1); // including ref

        using freeFrame_t = decltype(vsapi->freeFrame);
        const std::vector srcs = [&](){
            std::vector<std::unique_ptr<const VSFrameRef, const freeFrame_t &>> temp;

            temp.reserve(num_input_frames);

            if (final_) {
                for (int i = -radius; i <= radius; ++i) {
                    int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                    temp.emplace_back(
                        vsapi->getFrameFilter(clamped_n, d->ref_node, frameCtx),
                        vsapi->freeFrame
                    );
                }
            }

            for (int i = -radius; i <= radius; ++i) {
                int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                temp.emplace_back(
                    vsapi->getFrameFilter(clamped_n, d->node, frameCtx),
                    vsapi->freeFrame
                );
            }

            return temp;
        }();

        const VSFrameRef * src = srcs[radius + (final_ ? temporal_width : 0)].get();

        std::vector<std::unique_ptr<VSFrameRef, const freeFrame_t &>> dsts;
        if (radius) {
            if (d->unsafe) {
                dsts.reserve(temporal_width);

                for (int i = -radius; i <= radius; ++i) {
                    int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                    dsts.emplace_back(
                        const_cast<VSFrameRef *>(
                            vsapi->getFrameFilter(clamped_n, d->buffer_node, frameCtx)
                        ),
                        vsapi->freeFrame
                    ); // accumulated into input buffer
                }
            } else {
                dsts.emplace_back(
                    vsapi->newVideoFrame(
                        d->vi->format, d->vi->width,
                        d->vi->height * 2 * temporal_width,
                        src, core),
                    vsapi->freeFrame
                );
            }
        } else {
            const VSFrameRef * fr[] = {
                d->process[0] ? nullptr : src,
                d->process[1] ? nullptr : src,
                d->process[2] ? nullptr : src
            };
            const int pl[] = { 0, 1, 2 };

            dsts.emplace_back(
                vsapi->newVideoFrame2(
                    d->vi->format, d->vi->width,
                    d->vi->height, fr, pl, src, core),
                vsapi->freeFrame
            );
        }

        int lock_idx = 0;
        d->semaphore.acquire();
        if (d->num_copy_engines > 1) {
            for (int i = 0; i < d->num_copy_engines; ++i) {
                if (!d->resource_locks[i].test_and_set(std::memory_order::acquire)) {
                    lock_idx = i;
                    break;
                }
            }
        }

        const auto set_error = [&](const std::string & error_message) {
            if (d->num_copy_engines > 1) {
                d->resource_locks[lock_idx].clear(std::memory_order::release);
                d->semaphore.release();
            }

            vsapi->setFilterError(("BM3D: " + error_message).c_str(), frameCtx);

            return nullptr;
        };

        float * const h_res = d->resources[lock_idx].h_res;
        cudaStream_t stream = d->resources[lock_idx].stream;
        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);

        if (d->chroma) {
            int width = vsapi->getFrameWidth(src, 0);
            int height = vsapi->getFrameHeight(src, 0);
            int s_pitch = vsapi->getStride(src, 0);
            int s_stride = s_pitch / sizeof(float);
            int width_bytes = width * sizeof(float);

            cudaGraphExec_t graphexec = d->resources[lock_idx].graphexecs[0];

            float * h_src = h_res;
            for (int outer = 0; outer < (final_ ? 2 : 1); ++outer) {
                for (int i = 0; i < std::ssize(d->process); ++i) {
                    for (int j = 0; j < temporal_width; ++j) {
                        if (i == 0 || d->process[i]) {
                            auto current_src = srcs[j + outer * temporal_width].get();

                            vs_bitblt(
                                h_src, d_pitch,
                                vsapi->getReadPtr(current_src, i), s_pitch,
                                width_bytes, height
                            );
                        }
                        h_src += d_stride * height;
                    }
                }
            }

            checkError(cudaGraphLaunch(graphexec, stream));

            checkError(cudaStreamSynchronize(stream));

            float * h_dst = h_res;
            for (int plane = 0; plane < std::ssize(d->process); ++plane) {
                if (!d->process[plane]) {
                    h_dst += d_stride * height * 2 * temporal_width;
                    continue;
                }

                if (radius && d->unsafe) {
                    for (int i = 0; i < temporal_width; ++i) {
                        auto frame_lock_idx = std::clamp(
                            n - radius + i, 0, d->vi->numFrames - 1);
                        auto & frame_lock = d->frame_locks[frame_lock_idx];
                        while (frame_lock.test_and_set(std::memory_order::acquire)) {
                            frame_lock.wait(true, std::memory_order::relaxed);
                        }

                        auto dstp = reinterpret_cast<float *>(
                            vsapi->getWritePtr(dsts[i].get(), plane));

                        Accumulation(
                            dstp, s_stride,
                            &h_dst[i * 2 * height * d_stride], d_stride,
                            width, 2 * height);

                        frame_lock.clear(std::memory_order::release);
                        frame_lock.notify_one();
                    }
                } else {
                    float * dstp = reinterpret_cast<float *>(
                        vsapi->getWritePtr(dsts[0].get(), plane));

                    if (radius) {
                        vs_bitblt(
                            dstp, s_pitch, h_dst, d_pitch,
                            width_bytes, height * 2 * temporal_width
                        );
                    } else {
                        Aggregation(
                            dstp, s_stride,
                            h_dst, d_stride,
                            width, height
                        );
                    }
                }

                h_dst += d_stride * height * 2 * temporal_width;
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

                cudaGraphExec_t graphexec = d->resources[lock_idx].graphexecs[plane];

                float * h_src = h_res;
                for (int i = 0; i < num_input_frames; ++i) {
                    vs_bitblt(
                        h_src, d_pitch,
                        vsapi->getReadPtr(srcs[i].get(), plane), s_pitch,
                        width_bytes, height
                    );
                    h_src += d_stride * height;
                }

                checkError(cudaGraphLaunch(graphexec, stream));

                checkError(cudaStreamSynchronize(stream));

                if (radius && d->unsafe) {
                    for (int i = 0; i < temporal_width; ++i) {
                        auto frame_lock_idx = std::clamp(
                            n - radius + i, 0, d->vi->numFrames - 1);
                        auto & frame_lock = d->frame_locks[frame_lock_idx];
                        while (frame_lock.test_and_set(std::memory_order::acquire)) {
                            frame_lock.wait(true, std::memory_order::relaxed);
                        }

                        auto dstp = reinterpret_cast<float *>(
                            vsapi->getWritePtr(dsts[i].get(), plane));

                        Accumulation(
                            dstp, s_stride,
                            &h_res[i * 2 * height * d_stride], d_stride,
                            width, 2 * height);

                        frame_lock.clear(std::memory_order::release);
                        frame_lock.notify_one();
                    }
                } else {
                    float * dstp = reinterpret_cast<float *>(
                        vsapi->getWritePtr(dsts[0].get(), plane));

                    if (radius) {
                        vs_bitblt(
                            dstp, s_pitch, h_res, d_pitch,
                            width_bytes, height * 2 * temporal_width
                        );
                    } else {
                        Aggregation(
                            dstp, s_stride,
                            h_res, d_stride,
                            width, height
                        );
                    }
                }
            }
        }

        if (d->num_copy_engines > 1) {
            d->resource_locks[lock_idx].clear(std::memory_order::release);
        }
        d->semaphore.release();

        if (radius && !d->unsafe) {
            VSMap * dst_prop { vsapi->getFramePropsRW(dsts[0].get()) };

            vsapi->propSetInt(dst_prop, "BM3D_V_radius", d->radius, paReplace);

            int64_t process[3] { d->process[0], d->process[1], d->process[2] };
            vsapi->propSetIntArray(dst_prop, "BM3D_V_process", process, 3);
        }

        if (radius && d->unsafe) {
            // returns a signal
            auto ret = vsapi->newVideoFrame(
                vsapi->getFormatPreset(pfGray8, core),
                1, 1, src, core);

            return ret;
        } else {
            return dsts[0].release();
        }
    }

    return nullptr;
}

static void VS_CC BM3DFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<BM3DData *>(instanceData);

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->ref_node);
    vsapi->freeNode(d->buffer_node);

    cudaSetDevice(d->device_id);

    delete d;
}

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<BM3DData>() };

    const auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("BM3D: " + error_message).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->ref_node);
        vsapi->freeNode(d->buffer_node);
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

            // assumes grayscale input, hard_thr = 2.7
            sigma[i] *= (3.0f / 4.0f) / 255.0f * 64.0f * (final_ ? 1.0f : 2.7f);
        }
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

    bool unsafe = !!vsapi->propGetInt(in, "unsafe", 0, &error);
    if (error) {
        unsafe = false;
    }
    if (unsafe) {
        if (radius == 0) {
            return set_error("spatial denoising is always safe");
        }
        if (extractor != 0.0f) {
            return set_error("unsafe mode is not deterministic");
        }
        d->frame_locks = std::make_unique<std::atomic_flag[]>(d->vi->numFrames);
    }
    d->unsafe = unsafe;

    // GPU resource allocation
    {
        d->semaphore.current.store(num_copy_engines - 1, std::memory_order::relaxed);

        d->resource_locks = std::make_unique<std::atomic_flag[]>(num_copy_engines);

        d->resources.reserve(num_copy_engines);

        int max_width { d->process[0] ? width : width >> d->vi->format->subSamplingW };
        int max_height { d->process[0] ? height : height >> d->vi->format->subSamplingH };

        int num_planes { chroma ? 3 : 1 };
        int temporal_width = 2 * radius + 1;
        size_t d_pitch;
        int d_stride;
        for (int i = 0; i < num_copy_engines; ++i) {
            Resource<float *, cudaFree> d_src {};
            if (i == 0) {
                checkError(cudaMallocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float),
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height));
                d_stride = static_cast<int>(d_pitch / sizeof(float));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(cudaMalloc(&d_src.data,
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height * d_pitch));
            }

            Resource<float *, cudaFree> d_res {};
            checkError(cudaMalloc(&d_res.data,
                num_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<float *, cudaFreeHost> h_res {};
            checkError(cudaMallocHost(&h_res.data,
                num_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<cudaStream_t, cudaStreamDestroy> stream {};
            checkError(cudaStreamCreateWithFlags(&stream.data,
                cudaStreamNonBlocking));

            std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs {};
            if (d->chroma) {
                graphexecs[0] = get_graphexec(
                    d_res, d_src, h_res,
                    width, height, d_stride,
                    sigma[0], block_step[0], bm_range[0],
                    radius, ps_num[0], ps_range[0],
                    true, sigma[1], sigma[2],
                    final_, extractor
                );
            } else {
                auto subsamplingW = d->vi->format->subSamplingW;
                auto subsamplingH = d->vi->format->subSamplingH;

                for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                    if (d->process[plane]) {
                        int plane_width { plane == 0 ? width : width >> subsamplingW };
                        int plane_height { plane == 0 ? height : height >> subsamplingH };

                        graphexecs[plane] = get_graphexec(
                            d_res, d_src, h_res,
                            plane_width, plane_height, d_stride,
                            sigma[plane], block_step[plane], bm_range[plane],
                            radius, ps_num[plane], ps_range[plane],
                            false, 0.0f, 0.0f,
                            final_, extractor
                        );
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

    if (radius && unsafe) {
        VSMap * args = vsapi->createMap();
        vsapi->propSetNode(args, "clip", d->node, paReplace);

        VSMap * ret = vsapi->invoke(
            vsapi->getPluginById(PLUGIN_ID, core),
            "MakeBuffer", args);

        auto uncached_buffer_node = vsapi->propGetNode(ret, "clip", 0, nullptr);
        vsapi->freeMap(ret);

        vsapi->propSetNode(args, "clip", uncached_buffer_node, paReplace);
        vsapi->freeNode(uncached_buffer_node);

        ret = vsapi->invoke(
            vsapi->getPluginById("com.vapoursynth.std", core),
            "Cache", args);

        vsapi->freeMap(args);

        if (auto error = vsapi->getError(ret); error) {
            vsapi->freeMap(ret);
            return set_error(error);
        }

        d->buffer_node = vsapi->propGetNode(ret, "clip", 0, nullptr);
        vsapi->freeMap(ret);
    }

    vsapi->createFilter(
        in, out, "BM3D",
        BM3DInit, BM3DGetFrame, BM3DFree,
        fmParallel, 0, d.get(), core
    );

    if (radius && unsafe) {
        auto signal_node = vsapi->propGetNode(out, "clip", 0, nullptr);

        VSMap * args = vsapi->createMap();

        vsapi->propSetNode(args, "clip", d->buffer_node, paReplace);
        vsapi->propSetNode(args, "signal", signal_node, paReplace);
        vsapi->freeNode(signal_node);
        vsapi->propSetInt(args, "radius", d->radius, paReplace);
        int64_t process [] { d->process[0], d->process[1], d->process[2] };
        vsapi->propSetIntArray(args, "process", process, std::ssize(process));

        VSMap * ret = vsapi->invoke(
            vsapi->getPluginById(PLUGIN_ID, core),
            "VAggregate", args);
        vsapi->freeMap(args);

        auto ret_node = vsapi->propGetNode(ret, "clip", 0, nullptr);
        vsapi->propSetNode(out, "clip", ret_node, paReplace);
        vsapi->freeNode(ret_node);
        vsapi->freeMap(ret);
    }

    [[maybe_unused]] auto _ = d.release();
}

typedef struct {
    std::unique_ptr<const VSVideoInfo> out_vi;
} MakeBufferData;

static void VS_CC MakeBufferInit(
    VSMap *in, VSMap *out, void **instanceData,
    VSNode *node, VSCore *core, const VSAPI *vsapi
) {

    auto d = static_cast<MakeBufferData *>(*instanceData);
    vsapi->setVideoInfo(d->out_vi.get(), 1, node);
}

static const VSFrameRef *VS_CC MakeBufferGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {

    auto d = static_cast<MakeBufferData *>(*instanceData);

    if (activationReason == arInitial) {
        auto dst = vsapi->newVideoFrame(
            d->out_vi->format, d->out_vi->width, d->out_vi->height,
            nullptr, core);

        for (int plane = 0; plane < d->out_vi->format->numPlanes; ++plane) {
            auto dstp = vsapi->getWritePtr(dst, plane);
            auto height = vsapi->getFrameHeight(dst, plane);
            auto pitch = vsapi->getStride(dst, plane);
            memset(dstp, 0, height * pitch);
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC MakeBufferFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) {

    auto d = static_cast<MakeBufferData *>(instanceData);

    delete d;
}

static void VS_CC MakeBufferCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {

    auto d = std::make_unique<MakeBufferData>();

    auto node = vsapi->propGetNode(in, "clip", 0, nullptr);

    auto vi = std::make_unique<VSVideoInfo>(*vsapi->getVideoInfo(node));
    vi->height *= 2;
    d->out_vi = std::move(vi);

    vsapi->freeNode(node);

    vsapi->createFilter(
        in, out,
        "MakeBuffer",
        MakeBufferInit, MakeBufferGetFrame, MakeBufferFree,
        fmParallel,
        0, d.release(), core);
}

struct VAggregateData {
    VSNodeRef * node;
    VSNodeRef * signal_node;
    const VSVideoInfo * in_vi;
    int radius;
    bool process[3];
};

static void VS_CC VAggregateInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    VAggregateData * d = static_cast<VAggregateData *>(*instanceData);

    auto vi = *d->in_vi;
    vi.height /= 2;
    vsapi->setVideoInfo(&vi, 1, node);
}

static const VSFrameRef *VS_CC VAggregateGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    using freeFrame_t = decltype(vsapi->freeFrame);

    auto d = static_cast<VAggregateData *>(*instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->in_vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->signal_node, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        std::unique_ptr<const VSFrameRef, const freeFrame_t &> src {
            vsapi->getFrameFilter(n, d->node, frameCtx),
            vsapi->freeFrame
        };

        VSFrameRef * dst = [&](){
            int width = d->in_vi->width;
            int height = d->in_vi->height / 2;
            auto prop_frame = vsapi->getFrameFilter(n, d->signal_node, frameCtx);
            auto temp = vsapi->newVideoFrame(
                d->in_vi->format, width, height, prop_frame, core);
            vsapi->freeFrame(prop_frame);
            return temp;
        }();

        for (int plane = 0; plane < d->in_vi->format->numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }

            int width = vsapi->getFrameWidth(src.get(), plane);
            int height = vsapi->getFrameHeight(src.get(), plane) / 2;
            int stride = vsapi->getStride(src.get(), plane) / sizeof(float);

            const float * srcp = reinterpret_cast<const float *>(
                vsapi->getReadPtr(src.get(), plane));
            float * dstp = reinterpret_cast<float *>(
                vsapi->getWritePtr(dst, plane));
            Aggregation(dstp, stride, srcp, stride, width, height);
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC VAggregateFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<VAggregateData *>(instanceData);

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->signal_node);

    delete d;
}

static void VS_CC VAggregateCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<VAggregateData>() };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->in_vi = vsapi->getVideoInfo(d->node);

    d->signal_node = vsapi->propGetNode(in, "signal", 0, nullptr);

    d->radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, nullptr));

    auto process = vsapi->propGetIntArray(in, "process", nullptr);
    for (int i = 0; i < std::ssize(d->process); ++i) {
        d->process[i] = !!process[i];
    }

    vsapi->createFilter(
        in, out, "VAggregate",
        VAggregateInit, VAggregateGetFrame, VAggregateFree,
        fmParallel, 0, d.release(), core
    );
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) {

    configFunc(
        PLUGIN_ID, "bm3dcuda",
        "BM3D algorithm implemented in CUDA",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("BM3D",
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
        "unsafe:int:opt;",
        BM3DCreate, nullptr, plugin
    );

    registerFunc(
        "MakeBuffer",
        "clip:clip;",
        MakeBufferCreate, nullptr, plugin
    );

    registerFunc("VAggregate",
        "clip:clip;"
        "signal:clip;"
        "radius:int;"
        "process:int[]:opt;",
        VAggregateCreate, nullptr, plugin
    );
}
