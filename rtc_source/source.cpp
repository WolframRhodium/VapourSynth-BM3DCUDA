/*
 * VapourSynth wrapper for BM3DCUDA_RTC
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
#include <cfloat>
#include <ios>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#ifdef _WIN64
#include <windows.h>
#endif

#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>

#include "kernel.hpp"

#ifdef _MSC_VER
#if defined (_WINDEF_) && defined(min) && defined(max)
#undef min
#undef max
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

using namespace std::string_literals;

#define checkError(expr) do {                                                        \
    CUresult __err = expr;                                                           \
    if (__err != CUDA_SUCCESS) [[unlikely]] {                                        \
        const char * error_str;                                                      \
        cuGetErrorString(__err, &error_str);                                         \
        return set_error("'"s + # expr + "' failed: " + error_str);                  \
    }                                                                                \
} while(0)

#define checkNVRTCError(expr) do {                                                   \
    nvrtcResult __err = expr;                                                        \
    if (__err != NVRTC_SUCCESS) [[unlikely]] {                                       \
        return set_error("'"s + # expr + "' failed: " + nvrtcGetErrorString(__err)); \
    }                                                                                \
} while(0)

#define checkFilterError(expr) do {                                                  \
    CUresult __err = expr;                                                           \
    if (__err != CUDA_SUCCESS) [[unlikely]] {                                        \
        const char * error_str;                                                      \
        cuGetErrorString(__err, &error_str);                                         \
        const std::string error = "BM3D_RTC: '"s + # expr + "' faild: " + error_str; \
        vsapi->setFilterError(error.c_str(), frameCtx);                              \
        dst = nullptr;                                                               \
        goto FINALIZE;                                                               \
    }                                                                                \
} while(0)

constexpr int kFast = 4;

struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
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

    constexpr Resource() noexcept = default;

    constexpr Resource(Resource&& other) noexcept 
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

    constexpr auto deleter_(T x) const noexcept {
        if (x) {
            deleter(x);
        }
    }

    constexpr Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr Resource(T x) noexcept : data(x) {}

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};

struct CUDA_Resource {
    Resource<CUdeviceptr, cuMemFree> d_src;
    Resource<CUdeviceptr, cuMemFree> d_res;
    Resource<float *, cuMemFreeHost> h_res;
    Resource<CUstream, cuStreamDestroy> stream;
    Resource<CUgraphExec, cuGraphExecDestroy> graphexecs[3];
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

    int radius;
    int num_copy_engines; // fast
    bool chroma;
    bool process[3]; // sigma != 0
    bool final_;

    int d_pitch;

    Resource<CUdevice, cuDevicePrimaryCtxRelease> device;
    CUcontext context; // use primary context
    ticket_semaphore semaphore;
    std::unique_ptr<std::atomic_flag[]> locks;
    Resource<CUmodule, cuModuleUnload> modules[3];
    std::vector<CUDA_Resource> resources;
};

std::pair<CUmodule, std::string> compile(
    int width, int height, int stride, 
    float sigma, int block_step, int bm_range, 
    int radius, int ps_num, int ps_range, 
    bool chroma, float sigma_u, float sigma_v, 
    bool final_, CUdevice device
) {

    auto set_error = [](auto error_message) {
        return std::make_pair(CUmodule{}, error_message);
    };

    nvrtcProgram program;
    std::ostringstream kernel_source_io;
    kernel_source_io
        << "__device__ static const int width = " << width << ";\n"
        << "__device__ static const int height = " << height << ";\n"
        << "__device__ static const int stride = " << stride << ";\n"
        << "__device__ static const float sigma_y = " 
            << std::hexfloat << sigma << ";\n"
        << "__device__ static const int block_step = " << block_step << ";\n"
        << "__device__ static const int bm_range = " << bm_range << ";\n"
        << "__device__ static const int _radius = " << radius << ";\n"
        << "__device__ static const int ps_num = " << ps_num << ";\n"
        << "__device__ static const int ps_range = " << ps_range << ";\n"
        << "__device__ static const float sigma_u = " 
            << std::hexfloat << sigma_u << ";\n"
        << "__device__ static const float sigma_v = " 
            << std::hexfloat << sigma_v << ";\n"
        << "__device__ static const bool temporal = " << (radius > 0) << ";\n"
        << "__device__ static const bool chroma = " << chroma << ";\n"
        << "__device__ static const bool final_ = " << final_ << ";\n"
        << "__device__ static const float FLT_MAX = " 
            << std::hexfloat << FLT_MAX << ";\n"
        << "__device__ static const float FLT_EPSILON = " 
            << std::hexfloat << FLT_EPSILON << ";\n"
        << kernel_source_template;
    std::string kernel_source = kernel_source_io.str();
    checkNVRTCError(nvrtcCreateProgram(
        &program, kernel_source.c_str(), nullptr, 0, nullptr, nullptr));

    int major;
    checkError(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    int minor;
    checkError(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    int compute_capability = major * 10 + minor;

    // find maximum supported architecture
    int num_archs;
    checkNVRTCError(nvrtcGetNumSupportedArchs(&num_archs));
    const auto supported_archs = std::make_unique<int []>(num_archs);
    checkNVRTCError(nvrtcGetSupportedArchs(supported_archs.get()));
    bool generate_cubin = compute_capability <= supported_archs[num_archs - 1];

    std::string arch_str;
    if (generate_cubin) {
        arch_str = "-arch=sm_" + std::to_string(compute_capability);
    } else {
        arch_str = "-arch=compute_" + std::to_string(supported_archs[num_archs - 1]);
    }

    const char * opts[] = { arch_str.c_str(), "-use_fast_math", "-std=c++17" };
    checkNVRTCError(nvrtcCompileProgram(program, int{std::ssize(opts)}, opts));

    std::unique_ptr<char[]> image;
    if (generate_cubin) {
        size_t cubin_size;
        checkNVRTCError(nvrtcGetCUBINSize(program, &cubin_size));
        image = std::make_unique<char[]>(cubin_size);
        checkNVRTCError(nvrtcGetCUBIN(program, image.get()));
    } else {
        size_t ptx_size;
        checkNVRTCError(nvrtcGetPTXSize(program, &ptx_size));
        image = std::make_unique<char[]>(ptx_size);
        checkNVRTCError(nvrtcGetPTX(program, image.get()));
    }

    CUmodule module_;
    checkError(cuModuleLoadData(&module_, image.get()));

    checkNVRTCError(nvrtcDestroyProgram(&program));

    return {module_, ""};
}

CUgraphExec get_graphexec(
    CUdeviceptr d_res, CUdeviceptr d_src, float * h_res, 
    int width, int height, int stride, 
    int block_step, int radius, bool chroma,
    bool final_, CUcontext context, CUfunction function
) {
    size_t pitch { stride * sizeof(float) };
    int temporal_width { 2 * radius + 1 };
    int num_planes { chroma ? 3 : 1 };

    CUgraph graph;
    cuGraphCreate(&graph, 0);

    CUgraphNode n_HtoD;
    {
        size_t logical_height { 
            static_cast<size_t>((final_ ? 2 : 1) * num_planes * temporal_width * height) 
        };

        CUDA_MEMCPY3D copy_params {
            .srcMemoryType = CU_MEMORYTYPE_HOST, 
            .srcHost = h_res, 
            .srcPitch = pitch, 
            .dstMemoryType = CU_MEMORYTYPE_DEVICE, 
            .dstDevice = d_src, 
            .dstPitch = pitch, 
            .WidthInBytes = width * sizeof(float), 
            .Height = logical_height, 
            .Depth = 1, 
        };

        cuGraphAddMemcpyNode(&n_HtoD, graph, nullptr, 0, &copy_params, context);
    }

    CUgraphNode n_memset;
    {
        size_t logical_height { 
            static_cast<size_t>(num_planes * temporal_width * 2 * height) 
        };

        CUDA_MEMSET_NODE_PARAMS memset_params {
            .dst = d_res, 
            .pitch = pitch, 
            .value = 0, 
            .elementSize = 4, 
            .width = static_cast<size_t>(width), 
            .height = logical_height
        };

        cuGraphAddMemsetNode(&n_memset, graph, nullptr, 0, &memset_params, context);
    }

    CUgraphNode n_kernel;
    {
        CUgraphNode dependencies[] { n_HtoD, n_memset };

        void * kernel_args[] {
            &d_res, &d_src
        };

        CUDA_KERNEL_NODE_PARAMS node_params {
            .func = function, 
            .gridDimX = static_cast<unsigned int>((width + (4 * block_step - 1)) / (4 * block_step)), 
            .gridDimY = static_cast<unsigned int>((height + (block_step - 1)) / block_step), 
            .gridDimZ = 1, 
            .blockDimX = 32, 
            .blockDimY = 1, 
            .blockDimZ = 1, 
            .sharedMemBytes = 0, 
            .kernelParams = kernel_args, 
        };

        cuGraphAddKernelNode(
            &n_kernel, graph, dependencies, std::size(dependencies), &node_params);
    }

    CUgraphNode n_DtoH;
    {
        CUgraphNode dependencies[] { n_kernel };

        size_t logical_height { 
            static_cast<size_t>(num_planes * temporal_width * 2 * height) 
        };

        CUDA_MEMCPY3D copy_params {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE, 
            .srcDevice = d_res, 
            .srcPitch = pitch, 
            .dstMemoryType = CU_MEMORYTYPE_HOST, 
            .dstHost = h_res, 
            .dstPitch = pitch, 
            .WidthInBytes = width * sizeof(float), 
            .Height = logical_height, 
            .Depth = 1, 
        };

        cuGraphAddMemcpyNode(
            &n_DtoH, graph, dependencies, std::size(dependencies), &copy_params, context);
    }

    CUgraphExec graphexec;
    cuGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0);

    cuGraphDestroy(graph);

    return graphexec;
}

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
        int radius = d->radius;
        int temporal_width = 2 * radius + 1;
        bool final_ = d->final_;
        int num_input_frames = temporal_width * (final_ ? 2 : 1); // including ref

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
                for (int i = -radius; i <= radius; ++i) {
                    int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                    temp.emplace_back(
                        vsapi->getFrameFilter(clamped_n, d->node, frameCtx), 
                        vsapi->freeFrame
                    );
                }
            } else {
                for (int i = -radius; i <= radius; ++i) {
                    int clamped_n = std::clamp(n + i, 0, d->vi->numFrames - 1);
                    temp.emplace_back(
                        vsapi->getFrameFilter(clamped_n, d->node, frameCtx), 
                        vsapi->freeFrame
                    );
                }
            }

            return temp;
        }();

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
                d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core)
            );
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

        float * h_res = d->resources[lock_idx].h_res;
        CUstream stream = d->resources[lock_idx].stream;
        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);

        checkFilterError(cuCtxPushCurrent(d->context));

        if (d->chroma) {
            int width = vsapi->getFrameWidth(src, 0);
            int height = vsapi->getFrameHeight(src, 0);
            int s_pitch = vsapi->getStride(src, 0);
            int s_stride = s_pitch / sizeof(float);
            int width_bytes = width * sizeof(float);

            CUgraphExec graphexec = d->resources[lock_idx].graphexecs[0];

            float * h_src = h_res;
            for (int outer = 0; outer < (final_ ? 2 : 1); ++outer) {
                for (int i = 0; i < 3; ++i) {
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

            checkFilterError(cuGraphLaunch(graphexec, stream));

            checkFilterError(cuStreamSynchronize(stream));

            for (int plane = 0; plane < 3; ++plane) {
                if (d->process[plane]) {
                    float * dstp = reinterpret_cast<float *>(
                        vsapi->getWritePtr(dst.get(), plane));

                    if (radius) {
                        vs_bitblt(
                            dstp, s_pitch, h_res, d_pitch, 
                            width_bytes, height * 2 * temporal_width
                        );
                    } else {
                        Aggregation(
                            dstp, h_res, 
                            width, height, s_stride, d_stride
                        );
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

                    CUgraphExec graphexec = d->resources[lock_idx].graphexecs[plane];

                    float * h_src = h_res;
                    for (int i = 0; i < num_input_frames; ++i) {
                        vs_bitblt(
                            h_src, d_pitch, 
                            vsapi->getReadPtr(srcs[i].get(), plane), s_pitch, 
                            width_bytes, height
                        );
                        h_src += d_stride * height;
                    }

                    checkFilterError(cuGraphLaunch(graphexec, stream));

                    checkFilterError(cuStreamSynchronize(stream));

                    float * dstp = reinterpret_cast<float *>(
                        vsapi->getWritePtr(dst.get(), plane));

                    if (radius) {
                        vs_bitblt(
                            dstp, s_pitch, h_res, d_pitch, 
                            width_bytes, height * 2 * temporal_width
                        );
                    } else {
                        Aggregation(
                            dstp, h_res, 
                            width, height, s_stride, d_stride
                        );
                    }
                }
            }
        }

        checkFilterError(cuCtxPopCurrent(nullptr));

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
        vsapi->setError(out, ("BM3D_RTC: " + error_message).c_str());
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

    const bool fast = [&](){
        bool temp = !!vsapi->propGetInt(in, "fast", 0, &error);
        if (error) {
            return true;
        }
        return temp;
    }();
    const int num_copy_engines { fast ? kFast : 1 }; 
    d->num_copy_engines = num_copy_engines;

    d->semaphore.current.store(num_copy_engines - 1, std::memory_order::relaxed);
    d->locks = std::make_unique<std::atomic_flag[]>(num_copy_engines);

    // GPU related
    {
        checkError(cuInit(0));

        const int device_id = [&](){
            int temp = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
            if (error) {
                return 0;
            }
            return temp;
        }();
        int device_count;
        checkError(cuDeviceGetCount(&device_count));
        CUdevice device_;
        if (0 <= device_id && device_id < device_count) {
            checkError(cuDeviceGet(&device_, device_id));
        } else {
            return set_error("invalid device ID (" + std::to_string(device_id) + ")");
        }

        CUcontext context;
        checkError(cuDevicePrimaryCtxRetain(&context, device_));
        Resource<CUdevice, cuDevicePrimaryCtxRelease> device { device_ };
        checkError(cuCtxPushCurrent(context));
        d->context = context;

        d->resources.reserve(num_copy_engines);

        const int max_width { d->process[0] ? width : width >> d->vi->format->subSamplingW };
        const int max_height { d->process[0] ? height : height >> d->vi->format->subSamplingH };

        const int num_planes { chroma ? 3 : 1 };
        const int temporal_width = 2 * radius + 1;

#ifdef _WIN64
        const std::string plugin_path = 
            vsapi->getPluginPath(vsapi->getPluginById("com.WolframRhodium.BM3DCUDA_RTC", core));
        std::string folder_path = plugin_path.substr(0, plugin_path.find_last_of('/'));
        int nvrtc_major, nvrtc_minor;
        nvrtcVersion(&nvrtc_major, &nvrtc_minor);
        const int nvrtc_version = nvrtc_major * 10 + nvrtc_minor;
        const std::string dll_path = 
            folder_path + "/nvrtc-builtins64_" + std::to_string(nvrtc_version) + ".dll";
        const Resource<HMODULE, FreeLibrary> dll_handle = LoadLibraryA(dll_path.c_str());
#endif

        size_t d_pitch;
        int d_stride;
        CUfunction functions[3];
        for (int i = 0; i < num_copy_engines; ++i) {
            CUdeviceptr d_src_;
            if (i == 0) {
                checkError(cuMemAllocPitch(
                    &d_src_, &d_pitch, max_width * sizeof(float), 
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height, 16
                ));
                d_stride = static_cast<int>(d_pitch / sizeof(float));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(cuMemAlloc(&d_src_, 
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height * d_pitch));
            }
            Resource<CUdeviceptr, cuMemFree> d_src { d_src_ };

            CUdeviceptr d_res_;
            checkError(cuMemAlloc(&d_res_, 
                num_planes * temporal_width * 2 * max_height * d_pitch));
            Resource<CUdeviceptr, cuMemFree> d_res { d_res_ };

            void * h_res_;
            checkError(cuMemAllocHost(&h_res_, 
                num_planes * temporal_width * 2 * max_height * d_pitch));
            Resource<float *, cuMemFreeHost> h_res { static_cast<float *>(h_res_) };

            CUstream stream_;
            checkError(cuStreamCreate(&stream_, 
                CU_STREAM_NON_BLOCKING));
            Resource<CUstream, cuStreamDestroy> stream { stream_ };

            CUgraphExec graphexecs[3] {};
            if (chroma) {
                if (i == 0) {
                    auto [module_, error] = compile(
                        width, height, d_stride, 
                        sigma[0], block_step[0], bm_range[0], 
                        radius, ps_num[0], ps_range[0], 
                        true, sigma[1], sigma[2], 
                        final_, device
                    );

                    if (error.empty()) {
                        d->modules[0] = module_;
                    } else {
                        return set_error(error);
                    }

                    checkError(cuModuleGetFunction(&functions[0], d->modules[0], "bm3d"));
                }

                graphexecs[0] = get_graphexec(
                    d_res, d_src, h_res, 
                    width, height, d_stride, 
                    block_step[0], radius, 
                    true, final_, context, functions[0]
                );
            } else {
                auto subsamplingW = d->vi->format->subSamplingW;
                auto subsamplingH = d->vi->format->subSamplingH;

                for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                    if (d->process[plane]) {
                        int plane_width { plane == 0 ? width : width >> subsamplingW };
                        int plane_height { plane == 0 ? height : height >> subsamplingH };

                        if (i == 0) {
                            auto [module_, error] = compile(
                                plane_width, plane_height, d_stride, 
                                sigma[plane], block_step[plane], bm_range[plane], 
                                radius, ps_num[plane], ps_range[plane], 
                                false, 0.0f, 0.0f, final_, device
                            );

                            if (error.empty()) {
                                d->modules[plane] = module_;
                            } else {
                                return set_error(error);
                            }

                            checkError(cuModuleGetFunction(
                                &functions[plane], d->modules[plane], "bm3d"));
                        }

                        graphexecs[plane] = get_graphexec(
                            d_res, d_src, h_res, 
                            plane_width, plane_height, d_stride, 
                            block_step[plane], radius, 
                            false, final_, context, functions[plane]
                        );
                    }
                }
            }

            d->resources.push_back(CUDA_Resource{
                .d_src = std::move(d_src), 
                .d_res = std::move(d_res), 
                .h_res = std::move(h_res), 
                .stream = std::move(stream), 
                .graphexecs = { graphexecs[0], graphexecs[1], graphexecs[2] }
            });
        }

        d->device = CUdevice{ std::move(device) };

        checkError(cuCtxPopCurrent(nullptr));
    }

    vsapi->createFilter(
        in, out, "BM3D", 
        BM3DInit, BM3DGetFrame, BM3DFree, 
        fast ? fmParallel : fmParallelRequests, 0, d.release(), core
    );
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) {

    configFunc(
        "com.WolframRhodium.BM3DCUDA_RTC", "bm3dcuda_rtc", 
        "BM3D algorithm implemented in CUDA (NVRTC)", 
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
        "fast:int:opt;",
        BM3DCreate, nullptr, plugin
    );
}
