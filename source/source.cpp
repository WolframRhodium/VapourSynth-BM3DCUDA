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
#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cuda_runtime.h"

#include "vapoursynth/VapourSynth.h"
#include "vapoursynth/VSHelper.h"

using namespace std::string_literals;

extern cudaGraphExec_t get_graphexec(
    float * d_res, float * d_src, float * h_res, 
    int width, int height, int stride, 
    float sigma, int block_step, int bm_range, 
    int radius, int ps_num, int ps_range, 
    bool chroma, float sigma_u, float sigma_v, 
    bool final_);

#define checkError(expr) do {                                                               \
    cudaError_t __err = expr;                                                               \
    if (__err != cudaSuccess) {                                                             \
        return set_error("'"s + # expr + "' failed: " + cudaGetErrorString(__err));         \
    }                                                                                       \
} while(0)

constexpr int kFast = 4;

struct ticket_semaphore {
    std::atomic<int> ticket {};
    std::atomic<int> current {};

    void acquire() {
        int tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            int curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

template <typename T, auto deleter>
struct Resource {
    T data;

    constexpr Resource() : data() {}

    constexpr Resource(T x) : data(x) {}

    constexpr operator T() {
        return data;
    }

    constexpr auto deleter_(T x) {
        if (x) {
            deleter(x);
        }
    }

    constexpr Resource & operator = (T x) {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() {
        deleter_(data);
    }
};

struct CUDA_Resource {
    Resource<float *, cudaFree> d_src;
    Resource<float *, cudaFree> d_res;
    Resource<float *, cudaFreeHost> h_res;
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    Resource<cudaGraphExec_t, cudaGraphExecDestroy> graphexecs[3];

    CUDA_Resource(float * d_src, float * d_res, float * h_res, cudaStream_t stream) :
        d_src(d_src), d_res(d_res), h_res(h_res), stream(stream), graphexecs() {}
};

struct BM3DData {
    VSNodeRef * node;
    VSNodeRef * ref_node;
    const VSVideoInfo * vi;

    float sigma[3];
    int block_step[3];
    int bm_range[3];
    int radius;
    int ps_num[3];
    int ps_range[3];
    int num_copy_engines; // fast
    bool chroma;
    bool process[3]; // sigma != 0
    bool final_;

    int d_pitch;
    int device_id;

    ticket_semaphore semaphore;
    std::unique_ptr<std::atomic_flag[]> locks;
    std::vector<CUDA_Resource> resources;
};

static inline void Aggregation(
    float * VS_RESTRICT dstp, 
    const float * VS_RESTRICT h_res, 
    int width, int height, int s_stride, int d_stride
) {

    const float * wdst = h_res;
    const float * weight = &h_res[height * d_stride];

    for (auto y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = wdst[x] / weight[x];
        }

        dstp += s_stride;
        wdst += d_stride;
        weight += d_stride;
    }
}

static void VS_CC BM3DInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node, 
    VSCore *core, const VSAPI *vsapi
) {

    BM3DData * d = static_cast<BM3DData *>(*instanceData);

    if (d->radius) {
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
) {

    using freeFrame_t = decltype(vsapi->freeFrame);

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

        std::vector<std::unique_ptr<const VSFrameRef, const freeFrame_t &>> srcs;
        srcs.reserve(num_input_frames);

        if (final_) {
            for (int i = -radius; i <= radius; ++i) {
                int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                srcs.emplace_back(
                    vsapi->getFrameFilter(clamped_n, d->ref_node, frameCtx), 
                    vsapi->freeFrame);
            }
            for (int i = -radius; i <= radius; ++i) {
                int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                srcs.emplace_back(
                    vsapi->getFrameFilter(clamped_n, d->node, frameCtx), 
                    vsapi->freeFrame);
            }
        } else {
            for (int i = -radius; i <= radius; ++i) {
                int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                srcs.emplace_back(
                    vsapi->getFrameFilter(clamped_n, d->node, frameCtx), 
                    vsapi->freeFrame);
            }
        }

        const VSFrameRef * src = srcs[radius + (final_ ? temporal_width : 0)].get();

        std::unique_ptr<VSFrameRef, const freeFrame_t &> dst { nullptr, vsapi->freeFrame };
        if (radius) {
            dst.reset(vsapi->newVideoFrame(
                d->vi->format, d->vi->width, d->vi->height * 2 * temporal_width, 
                src, core));
        } else {
            const VSFrameRef * fr[] = { 
                d->process[0] ? nullptr : src, 
                d->process[1] ? nullptr : src, 
                d->process[2] ? nullptr : src
            };
            const int pl[] = { 0, 1, 2 };

            dst.reset(vsapi->newVideoFrame2(
                d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core));
        }

        int lock_idx = 0;
        if (d->num_copy_engines > 1) {
            d->semaphore.acquire();

            for (int i = 0; i < d->num_copy_engines; ++i) {
                if (!d->locks[i].test_and_set(std::memory_order::acquire)) {
                    lock_idx = i;
                    break;
                }
            }
        }

        float * d_src = d->resources[lock_idx].d_src;
        float * d_res = d->resources[lock_idx].d_res;
        float * h_res = d->resources[lock_idx].h_res;
        cudaStream_t stream = d->resources[lock_idx].stream;
        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);

        if (d->chroma) {
            int width = vsapi->getFrameWidth(src, 0);
            int height = vsapi->getFrameHeight(src, 0);
            int s_pitch = vsapi->getStride(src, 0);
            int s_stride = s_pitch / sizeof(float);
            int width_bytes = width * sizeof(float);

            if (!d->resources[lock_idx].graphexecs[0]) {
                float sigma = d->sigma[0];
                int block_step = d->block_step[0];
                int bm_range = d->bm_range[0];
                int ps_num = d->ps_num[0];
                int ps_range = d->ps_range[0];

                d->resources[lock_idx].graphexecs[0] = get_graphexec(
                    d_res, d_src, h_res, 
                    width, height, d_stride, 
                    sigma, block_step, bm_range, 
                    radius, ps_num, ps_range, 
                    true, d->sigma[1], d->sigma[2], 
                    final_);
            }
            cudaGraphExec_t graphexec = d->resources[lock_idx].graphexecs[0];

            float * h_src = h_res;
            for (int outer = 0; outer < (final_ ? 2 : 1); ++outer) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < temporal_width; ++j) {
                        if (i == 0 || d->process[i]) {
                            vs_bitblt(
                                h_src, d_pitch, 
                                vsapi->getReadPtr(srcs[j + outer * temporal_width].get(), i), s_pitch, 
                                width_bytes, height);
                        }
                        h_src += d_stride * height;
                    }
                }
            }

            cudaGraphLaunch(graphexec, stream);

            if (auto error = cudaStreamSynchronize(stream); error != cudaSuccess) {
                vsapi->setFilterError(
                    ("BM3D: "s + cudaGetErrorString(error)).c_str(), 
                    frameCtx
                );
                dst = nullptr;
                goto FINALIZE;
            }

            for (int plane = 0; plane < 3; ++plane) {
                if (d->process[plane]) {
                    float * dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst.get(), plane));

                    if (radius) {
                        vs_bitblt(dstp, s_pitch, h_res, d_pitch, 
                            width_bytes, height * 2 * temporal_width);
                    } else {
                        Aggregation(
                            dstp, h_res, 
                            width, height, s_stride, d_stride);
                    }
                }

                h_res += d_stride * height * 2 * temporal_width;
            }
        } else {
            for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
                if (d->process[plane]) {
                    int width = vsapi->getFrameWidth(src, plane);
                    int height = vsapi->getFrameHeight(src, plane);
                    int s_pitch = vsapi->getStride(src, plane);
                    int s_stride = s_pitch / sizeof(float);
                    int width_bytes = width * sizeof(float);

                    if (!d->resources[lock_idx].graphexecs[plane]) {
                        float sigma = d->sigma[plane];
                        int block_step = d->block_step[plane];
                        int bm_range = d->bm_range[plane];
                        int ps_num = d->ps_num[plane];
                        int ps_range = d->ps_range[plane];

                        d->resources[lock_idx].graphexecs[plane] = get_graphexec(
                            d_res, d_src, h_res, 
                            width, height, d_stride, 
                            sigma, block_step, bm_range, 
                            radius, ps_num, ps_range, 
                            false, 0.0f, 0.0f, 
                            final_);
                    }
                    cudaGraphExec_t graphexec = d->resources[lock_idx].graphexecs[plane];

                    float * h_src = h_res;
                    for (int i = 0; i < num_input_frames; ++i) {
                        vs_bitblt(
                            h_src, d_pitch, 
                            vsapi->getReadPtr(srcs[i].get(), plane), s_pitch, 
                            width_bytes, height);
                        h_src += d_stride * height;
                    }

                    cudaGraphLaunch(graphexec, stream);

                    if (auto error = cudaStreamSynchronize(stream); error != cudaSuccess) {
                        vsapi->setFilterError(
                            ("BM3D: "s + cudaGetErrorString(error)).c_str(), 
                            frameCtx
                        );
                        dst = nullptr;
                        goto FINALIZE;
                    }

                    float * dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst.get(), plane));

                    if (radius) {
                        vs_bitblt(dstp, s_pitch, h_res, d_pitch, 
                            width_bytes, height * 2 * temporal_width);
                    } else {
                        Aggregation(
                            dstp, h_res, 
                            width, height, s_stride, d_stride);
                    }
                }
            }
        }

FINALIZE:
        if (d->num_copy_engines > 1) {
            d->locks[lock_idx].clear(std::memory_order::release);
            d->semaphore.release();
        }

        if (radius && dst) {
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
) {

    auto d = static_cast<BM3DData *>(instanceData);

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->ref_node);

    delete d;
}

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, void *userData, 
    VSCore *core, const VSAPI *vsapi
) {

    auto d { std::make_unique<BM3DData>() };

    auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("BM3D: " + error_message).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->ref_node);
    };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);
    int width = d->vi->width;
    int height = d->vi->height;
    int bits_per_sample = d->vi->format->bitsPerSample;

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

    for (int i = 0; i < std::ssize(d->sigma); ++i) {
        float sigma = static_cast<float>(
            vsapi->propGetFloat(in, "sigma", i, &error));

        if (error) {
            sigma = (i == 0) ? 3.0f : d->sigma[i - 1];
        } else if (sigma < 0.0f) {
            return set_error("\"sigma\" must be non-negative");
        }

        if (sigma < std::numeric_limits<float>::epsilon()) {
            d->process[i] = false;
        } else {
            d->process[i] = true;

            // assumes grayscale input, hard_thr = 2.7
            sigma *= (3.0f / 4.0f) / 255.0f * 64.0f * (final_ ? 1.0f : 2.7f);
        }

        d->sigma[i] = sigma;
    }

    for (int i = 0; i < std::ssize(d->block_step); ++i) {
        int block_step = int64ToIntS(
            vsapi->propGetInt(in, "block_step", i, &error));

        if (error) {
            block_step = (i == 0) ? 8 : d->block_step[i - 1];
        } else if (block_step <= 0 || block_step > 8) {
            return set_error("\"block_step\" must be in range [1, 8]");
        }

        d->block_step[i] = block_step;
    }

    for (int i = 0; i < std::ssize(d->bm_range); ++i) {
        int bm_range = int64ToIntS(
            vsapi->propGetInt(in, "bm_range", i, &error));

        if (error) {
            bm_range = (i == 0) ? 9 : d->bm_range[i - 1];
        } else if (bm_range <= 0) {
            return set_error("\"bm_range\" must be positive");
        }

        d->bm_range[i] = bm_range;
    }

    int radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &error));
    if (error) {
        radius = 0;
    } else if (radius < 0) {
        return set_error("\"radius\" must be non-negative");
    }
    d->radius = radius;

    for (int i = 0; i < std::ssize(d->ps_num); ++i) {
        int ps_num = int64ToIntS(
            vsapi->propGetInt(in, "ps_num", i, &error));

        if (error) {
            ps_num = (i == 0) ? 2 : d->ps_num[i - 1];
        } else if (ps_num <= 0 || ps_num > 8) {
            return set_error("\"ps_num\" must be in range [1, 8]");
        }

        d->ps_num[i] = ps_num;
    }

    for (int i = 0; i < std::ssize(d->ps_range); ++i) {
        int ps_range = int64ToIntS(
            vsapi->propGetInt(in, "ps_range", i, &error));

        if (error) {
            ps_range = (i == 0) ? 4 : d->ps_range[i - 1];
        } else if (ps_range <= 0) {
            return set_error("\"ps_range\" must be positive");
        }

        d->ps_range[i] = ps_range;
    }

    bool chroma = !!vsapi->propGetInt(in, "chroma", 0, &error);
    if (error) {
        chroma = false;
    }
    if (chroma && d->vi->format->id != pfYUV444PS) {
        return set_error("clip format must be YUV444 when \"chroma\" is true");
    }
    d->chroma = chroma;

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }
    int device_count;
    checkError(cudaGetDeviceCount(&device_count));
    if (0 <= device_id && device_id < device_count) {
        checkError(cudaSetDevice(device_id));
    } else {
        return set_error("invalid device ID (" + std::to_string(device_id) + ")");
    }
    d->device_id = device_id;

    bool fast = !!vsapi->propGetInt(in, "fast", 0, &error);
    if (error) {
        fast = true;
    }
    int num_copy_engines { fast ? kFast : 1 }; 
    d->num_copy_engines = num_copy_engines;

    // GPU resource allocation
    {
        d->semaphore.current.store(num_copy_engines - 1, std::memory_order::relaxed);

        d->locks = std::move(std::make_unique<std::atomic_flag[]>(num_copy_engines));

        d->resources.reserve(num_copy_engines);

        int max_width, max_height;
        if (d->process[0]) {
            max_width = width;
            max_height = height;
        } else {
            max_width = width >> d->vi->format->subSamplingW;
            max_height = height >> d->vi->format->subSamplingH;
        }

        int num_planes { chroma ? 3 : 1 };
        int temporal_width = 2 * radius + 1;
        size_t d_pitch;
        for (int i = 0; i < num_copy_engines; ++i) {
            float * d_src;
            if (i == 0) {
                checkError(cudaMallocPitch(
                    &d_src, &d_pitch, max_width * sizeof(float), 
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(cudaMalloc(&d_src, 
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height * d_pitch));
            }

            float * d_res;
            checkError(cudaMalloc(&d_res, 
                num_planes * temporal_width * 2 * max_height * d_pitch));

            float * h_res;
            checkError(cudaHostAlloc(&h_res, 
                num_planes * temporal_width * 2 * max_height * d_pitch, 
                cudaHostAllocDefault));

            cudaStream_t stream;
            checkError(cudaStreamCreateWithFlags(&stream, 
                cudaStreamNonBlocking));

            d->resources.emplace_back(d_src, d_res, h_res, stream);
        }
    }

    vsapi->createFilter(
        in, out, "BM3D", 
        BM3DInit, BM3DGetFrame, BM3DFree, 
        fast ? fmParallel : fmParallelRequests, 0, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) {

    configFunc(
        "com.WolframRhodium.BM3DCUDA", "bm3dcuda", "BM3D algorithm implemented in CUDA", 
        VAPOURSYNTH_API_VERSION, 1, plugin);

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
        "fast:int:opt;",
        BM3DCreate, nullptr, plugin);
}
