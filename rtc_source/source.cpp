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
#include <array>
#include <cctype>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <ios>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#include <VapourSynth.h>
#include <VSHelper.h>

#include "kernel.hpp"

#ifdef _MSC_VER
#   if defined (_WINDEF_) && defined(min) && defined(max)
#       undef min
#       undef max
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#endif

using namespace std::string_literals;

#define checkError(expr) do {                                                         \
    if (CUresult result = expr; result != CUDA_SUCCESS) [[unlikely]] {                \
        const char * error_str;                                                       \
        cuGetErrorString(result, &error_str);                                         \
        return set_error("'"s + # expr + "' failed: " + error_str);                   \
    }                                                                                 \
} while(0)

#define checkNVRTCError(expr) do {                                                    \
    if (nvrtcResult result = expr; result != NVRTC_SUCCESS) [[unlikely]] {            \
        return set_error("'"s + # expr + "' failed: " + nvrtcGetErrorString(result)); \
    }                                                                                 \
} while(0)

constexpr int kFast = 4;

static VSPlugin * myself = nullptr;

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

    Resource& operator=(Resource&& other) noexcept {
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
            x = T{};
        }
    }

    Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};

struct CUDA_Resource {
    Resource<CUdeviceptr, cuMemFree> d_src;
    Resource<CUdeviceptr, cuMemFree> d_res;
    Resource<float *, cuMemFreeHost> h_res;
    Resource<CUstream, cuStreamDestroy> stream;
    std::array<Resource<CUgraphExec, cuGraphExecDestroy>, 3> graphexecs;
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
    // float extractor

    int radius;
    int num_copy_engines; // fast
    bool chroma;
    bool process[3]; // sigma != 0
    bool final_;
    std::string bm_error_s[3];
    std::string transform_2d_s[3];
    std::string transform_1d_s[3];

    int d_pitch;

    CUdevice device;
    CUcontext context; // use primary context
    ticket_semaphore semaphore;
    Resource<CUmodule, cuModuleUnload> modules[3];
    std::vector<CUDA_Resource> resources;
    std::mutex resources_lock;
};

static std::variant<CUmodule, std::string> compile(
    int width, int height, int stride,
    float sigma, int block_step, int bm_range,
    int radius, int ps_num, int ps_range,
    bool chroma, float sigma_u, float sigma_v,
    bool final_,
    const std::string & bm_error_s,
    const std::string & transform_2d_s,
    const std::string & transform_1d_s,
    float extractor,
    CUdevice device
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    std::ostringstream kernel_source_io;
    kernel_source_io
        << kernel_header_template
        << "#define transform_2d " << transform_2d_s << "\n"
        << "#define transform_1d " << transform_1d_s << "\n"
        << "#define bm_error " << bm_error_s << "\n"
        << std::hexfloat << std::boolalpha
        << "__device__ static const int width = " << width << ";\n"
        << "__device__ static const int height = " << height << ";\n"
        << "__device__ static const int stride = " << stride << ";\n"
        << "__device__ static const float sigma_y = " << sigma << ";\n"
        << "__device__ static const int block_step = " << block_step << ";\n"
        << "__device__ static const int bm_range = " << bm_range << ";\n"
        << "__device__ static const int _radius = " << radius << ";\n"
        << "__device__ static const int ps_num = " << ps_num << ";\n"
        << "__device__ static const int ps_range = " << ps_range << ";\n"
        << "__device__ static const float sigma_u = " << sigma_u << ";\n"
        << "__device__ static const float sigma_v = " << sigma_v << ";\n"
        << "__device__ static const bool temporal = " << (radius > 0) << ";\n"
        << "__device__ static const bool chroma = " << chroma << ";\n"
        << "__device__ static const bool final_ = " << final_ << ";\n"
        << "__device__ static const float extractor = " << extractor << ";\n"
        << "__device__ static const float FLT_MAX = "
            << std::numeric_limits<float>::max() << ";\n"
        << "__device__ static const float FLT_EPSILON = "
            << std::numeric_limits<float>::epsilon() << ";\n"
        << kernel_source_template;
    const std::string kernel_source = kernel_source_io.str();

    nvrtcProgram program;
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

    const std::string arch_str = {
        generate_cubin ?
        "-arch=sm_" + std::to_string(compute_capability) :
        "-arch=compute_" + std::to_string(supported_archs[num_archs - 1])
    };

    const char * opts[] = {
        arch_str.c_str(),
        "-use_fast_math",
        "-std=c++17",
        "-modify-stack-limit=false"
    };

    if (nvrtcCompileProgram(program, int{std::ssize(opts)}, opts) != NVRTC_SUCCESS) {
        size_t log_size;
        checkNVRTCError(nvrtcGetProgramLogSize(program, &log_size));
        std::string error_message;
        error_message.resize(log_size);
        checkNVRTCError(nvrtcGetProgramLog(program, error_message.data()));
        return set_error(error_message);
    }

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

    checkNVRTCError(nvrtcDestroyProgram(&program));

    CUmodule module_;
    checkError(cuModuleLoadData(&module_, image.get()));

    return module_;
}

static std::variant<CUgraphExec, std::string> get_graphexec(
    CUdeviceptr d_res, CUdeviceptr d_src, float * h_res,
    int width, int height, int stride,
    int block_step, int radius, bool chroma,
    bool final_, CUcontext context, CUfunction function
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    size_t pitch { stride * sizeof(float) };
    int temporal_width { 2 * radius + 1 };
    int num_planes { chroma ? 3 : 1 };

    CUgraph graph_;
    checkError(cuGraphCreate(&graph_, 0));
    Resource<CUgraph, cuGraphDestroy> graph { graph_ };

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

        checkError(cuGraphAddMemcpyNode(&n_HtoD, graph, nullptr, 0, &copy_params, context));
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

        checkError(cuGraphAddMemsetNode(&n_memset, graph, nullptr, 0, &memset_params, context));
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

        checkError(cuGraphAddKernelNode(
            &n_kernel, graph, dependencies, std::size(dependencies), &node_params));
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

        checkError(cuGraphAddMemcpyNode(
            &n_DtoH, graph, dependencies, std::size(dependencies), &copy_params, context));
    }

    CUgraphExec graphexec;
    checkError(cuGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));

    return graphexec;
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

        std::unique_ptr<VSFrameRef, const freeFrame_t &> dst { nullptr, vsapi->freeFrame };
        if (radius) {
            dst.reset(
                vsapi->newVideoFrame(
                    d->vi->format, d->vi->width,
                    d->vi->height * 2 * temporal_width,
                    src, core)
            );
            for (int i = 0; i < d->vi->format->numPlanes; ++i) {
                if (!d->process[i]) {
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

            vsapi->setFilterError(("BM3D_RTC: " + error_message).c_str(), frameCtx);

            return nullptr;
        };

        float * const h_res = resource.h_res;
        CUstream stream = resource.stream;
        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);

        checkError(cuCtxPushCurrent(d->context));

        if (d->chroma) {
            int width = vsapi->getFrameWidth(src, 0);
            int height = vsapi->getFrameHeight(src, 0);
            int s_pitch = vsapi->getStride(src, 0);
            int s_stride = s_pitch / sizeof(float);
            int width_bytes = width * sizeof(float);

            CUgraphExec graphexec = resource.graphexecs[0];

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

            checkError(cuGraphLaunch(graphexec, stream));

            checkError(cuStreamSynchronize(stream));

            float * h_dst = h_res;
            for (int plane = 0; plane < std::ssize(d->process); ++plane) {
                if (!d->process[plane]) {
                    h_dst += d_stride * height * 2 * temporal_width;
                    continue;
                }

                float * dstp = reinterpret_cast<float *>(
                    vsapi->getWritePtr(dst.get(), plane));

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

                CUgraphExec graphexec = resource.graphexecs[plane];

                float * h_src = h_res;
                for (int i = 0; i < num_input_frames; ++i) {
                    vs_bitblt(
                        h_src, d_pitch,
                        vsapi->getReadPtr(srcs[i].get(), plane), s_pitch,
                        width_bytes, height
                    );
                    h_src += d_stride * height;
                }

                checkError(cuGraphLaunch(graphexec, stream));

                checkError(cuStreamSynchronize(stream));

                float * dstp = reinterpret_cast<float *>(
                    vsapi->getWritePtr(dst.get(), plane));

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

        checkError(cuCtxPopCurrent(nullptr));

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        if (radius) {
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

    auto device = d->device;

    cuCtxPushCurrent(d->context);

    delete d;

    cuCtxPopCurrent(nullptr);

    cuDevicePrimaryCtxRelease(device);
}

static void VS_CC BM3DCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d { std::make_unique<BM3DData>() };

    const auto set_error = [&](const std::string & error_message) {
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
    }
    for (int i = 0; i < std::ssize(sigma); ++i) {
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

    for (int i = 0; i < std::ssize(d->bm_error_s); ++i) {
        auto _temp = vsapi->propGetData(in, "bm_error_s", i, &error);
        auto temp = std::string{ _temp ? _temp : "" };
        if (error) {
            temp = (i == 0) ? "ssd" : d->bm_error_s[i - 1];
        } else {
            std::transform(
                temp.begin(), temp.end(), temp.begin(),
                [](unsigned char c){ return std::tolower(c); }
            );
            if (
                temp != "ssd" && temp != "sad" &&
                temp != "zssd" && temp != "zsad" &&
                temp != "ssd/norm") {
                return set_error("invalid \'bm_error_s\': "  + temp);
            }
            if (temp == "ssd/norm") {
                temp = "ssd_norm";
            }
        }
        d->bm_error_s[i] = std::move(temp);
    }

    for (int i = 0; i < std::ssize(d->transform_2d_s); ++i) {
        auto _temp = vsapi->propGetData(in, "transform_2d_s", i, &error);
        auto temp = std::string{ _temp ? _temp : "" };
        if (error) {
            temp = (i == 0) ? "dct" : d->transform_2d_s[i - 1];
        } else {
            std::for_each(temp.begin(), temp.end(), [](char & c){c = std::tolower(c);});
            if (temp != "dct" && temp != "haar" && temp != "wht" && temp != "bior1.5") {
                return set_error("invalid \'transform_2d_s\': " + temp);
            }
            if (temp == "bior1.5") {
                temp = "bior1_5";
            }
        }
        d->transform_2d_s[i] = std::move(temp);
    }

    for (int i = 0; i < std::ssize(d->transform_1d_s); ++i) {
        auto _temp = vsapi->propGetData(in, "transform_1d_s", i, &error);
        auto temp = std::string{ _temp ? _temp : "" };
        if (error) {
            temp = (i == 0) ? "dct" : d->transform_1d_s[i - 1];
        } else {
            std::transform(
                temp.begin(), temp.end(), temp.begin(),
                [](unsigned char c){ return std::tolower(c); }
            );
            if (temp != "dct" && temp != "haar" && temp != "wht" && temp != "bior1.5") {
                return set_error("invalid \'transform_1d_s\': "  + temp);
            }
            if (temp == "bior1.5") {
                temp = "bior1_5";
            }
        }
        d->transform_1d_s[i] = std::move(temp);
    }

    const float extractor = [&](){
        int temp = int64ToIntS(vsapi->propGetInt(in, "extractor_exp", 0, &error));
        if (error) {
            return 0.0f;
        }
        return (temp ? std::ldexp(1.0f, temp) : 0.0f);
    }();

    d->semaphore.current.store(num_copy_engines - 1, std::memory_order::relaxed);

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
        if (0 <= device_id && device_id < device_count) {
            checkError(cuDeviceGet(&d->device, device_id));
        } else {
            return set_error("invalid device ID (" + std::to_string(device_id) + ")");
        }

        checkError(cuDevicePrimaryCtxRetain(&d->context, d->device));
        checkError(cuCtxPushCurrent(d->context));

        d->resources.reserve(num_copy_engines);

        const int max_width { d->process[0] ? width : width >> d->vi->format->subSamplingW };
        const int max_height { d->process[0] ? height : height >> d->vi->format->subSamplingH };

        const int num_planes { chroma ? 3 : 1 };
        const int temporal_width = 2 * radius + 1;

        size_t d_pitch;
        int d_stride;
        CUfunction functions[3];
        for (int i = 0; i < num_copy_engines; ++i) {
            Resource<CUdeviceptr, cuMemFree> d_src {};
            if (i == 0) {
                checkError(cuMemAllocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float),
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height, 4
                ));
                d_stride = static_cast<int>(d_pitch / sizeof(float));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(cuMemAlloc(&d_src.data,
                    (final_ ? 2 : 1) * num_planes * temporal_width * max_height * d_pitch));
            }

            Resource<CUdeviceptr, cuMemFree> d_res {};
            checkError(cuMemAlloc(&d_res.data,
                num_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<float *, cuMemFreeHost> h_res {};
            checkError(cuMemAllocHost(reinterpret_cast<void **>(&h_res.data),
                num_planes * temporal_width * 2 * max_height * d_pitch));

            Resource<CUstream, cuStreamDestroy> stream {};
            checkError(cuStreamCreate(&stream.data, CU_STREAM_NON_BLOCKING));

            std::array<Resource<CUgraphExec, cuGraphExecDestroy>, 3> graphexecs {};
            if (chroma) {
                if (i == 0) {
                    const auto result = compile(
                        width, height, d_stride,
                        sigma[0], block_step[0], bm_range[0],
                        radius, ps_num[0], ps_range[0],
                        true, sigma[1], sigma[2],
                        final_,
                        d->bm_error_s[0],
                        d->transform_2d_s[0], d->transform_1d_s[0],
                        extractor,
                        d->device
                    );

                    if (std::holds_alternative<CUmodule>(result)) {
                        d->modules[0] = std::get<CUmodule>(result);
                    } else {
                        return set_error(std::get<std::string>(result));
                    }

                    checkError(cuModuleGetFunction(&functions[0], d->modules[0], "bm3d"));
                }

                const auto result = get_graphexec(
                    d_res, d_src, h_res,
                    width, height, d_stride,
                    block_step[0], radius,
                    true, final_, d->context, functions[0]
                );

                if (std::holds_alternative<CUgraphExec>(result)) {
                    graphexecs[0] = std::get<CUgraphExec>(result);
                } else {
                    return set_error(std::get<std::string>(result));
                }
            } else {
                auto subsamplingW = d->vi->format->subSamplingW;
                auto subsamplingH = d->vi->format->subSamplingH;

                for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                    if (d->process[plane]) {
                        int plane_width { plane == 0 ? width : width >> subsamplingW };
                        int plane_height { plane == 0 ? height : height >> subsamplingH };

                        if (i == 0) {
                            const auto result = compile(
                                plane_width, plane_height, d_stride,
                                sigma[plane], block_step[plane], bm_range[plane],
                                radius, ps_num[plane], ps_range[plane],
                                false, 0.0f, 0.0f, final_,
                                d->bm_error_s[plane],
                                d->transform_2d_s[plane], d->transform_1d_s[plane],
                                extractor,
                                d->device
                            );

                            if (std::holds_alternative<CUmodule>(result)) {
                                d->modules[plane] = std::get<CUmodule>(result);
                            } else {
                                return set_error(std::get<std::string>(result));
                            }

                            checkError(cuModuleGetFunction(
                                &functions[plane], d->modules[plane], "bm3d"));
                        }

                        const auto result = get_graphexec(
                            d_res, d_src, h_res,
                            plane_width, plane_height, d_stride,
                            block_step[plane], radius,
                            false, final_, d->context, functions[plane]
                        );

                        if (std::holds_alternative<CUgraphExec>(result)) {
                            graphexecs[plane] = std::get<CUgraphExec>(result);
                        } else {
                            return set_error(std::get<std::string>(result));
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

        checkError(cuCtxPopCurrent(nullptr));
    }

    vsapi->createFilter(
        in, out, "BM3D",
        BM3DInit, BM3DGetFrame, BM3DFree,
        fmParallel, 0, d.release(), core
    );
}

struct VAggregateData {
    VSNodeRef * node;

    VSNodeRef * src_node;
    const VSVideoInfo * src_vi;

    std::array<bool, 3> process; // sigma != 0

    int radius;

    std::unordered_map<std::thread::id, float *> buffer;
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

        auto thread_id = std::this_thread::get_id();

        float * buffer;
        {
            try {
                buffer = d->buffer.at(thread_id);
            } catch (const std::out_of_range &) {
                assert(d->process[0] || d->src_vi->numFrames > 1);

                const int max_height {
                    d->process[0] ?
                    vsapi->getFrameHeight(src_frame, 0) :
                    vsapi->getFrameHeight(src_frame, 1)
                };
                const int max_pitch {
                    d->process[0] ?
                    vsapi->getStride(src_frame, 0) :
                    vsapi->getStride(src_frame, 1)
                };
                buffer = reinterpret_cast<float *>(std::malloc(2 * max_height * max_pitch));
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

                memset(buffer, 0, 2 * plane_height * plane_stride * sizeof(float));

                for (int i = 0; i < 2 * d->radius + 1; ++i) {
                    auto agg_src = reinterpret_cast<const float *>(vsapi->getReadPtr(vbm3d_frames[i], plane));
                    agg_src += (2 * d->radius - i) * 2 * plane_height * plane_stride;

                    float * agg_dst = buffer;

                    for (int y = 0; y < 2 * plane_height; ++y) {
                        for (int x = 0; x < plane_width; ++x) {
                            agg_dst[x] += agg_src[x];
                        }
                        agg_src += plane_stride;
                        agg_dst += plane_stride;
                    }
                }

                auto dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst_frame, plane));
                Aggregation(dstp, plane_stride, buffer, plane_stride, plane_width, plane_height);
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

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(d->node);
    d->src_node = vsapi->propGetNode(in, "src", 0, nullptr);
    d->src_vi = vsapi->getVideoInfo(d->src_node);

    d->radius = (vi->height / d->src_vi->height - 2) / 4;

    d->process.fill(false);
    int num_planes_args = vsapi->propNumElements(in, "planes");
    for (int i = 0; i < num_planes_args; ++i) {
        int plane = vsapi->propGetInt(in, "planes", i, nullptr);
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

static void VS_CC BM3Dv2Create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {

    std::array<bool, 3> process;
    process.fill(true);

    int num_sigma_args = vsapi->propNumElements(in, "sigma");
    for (int i = 0; i < std::min(3, num_sigma_args); ++i) {
        auto sigma = vsapi->propGetFloat(in, "sigma", i, nullptr);
        if (sigma < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        }
    }

    bool skip = true;
    auto src = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto src_vi = vsapi->getVideoInfo(src);
    for (int i = 0; i < src_vi->format->numPlanes; ++i) {
        skip &= !process[i];
    }
    if (skip) {
        vsapi->propSetNode(out, "clip", src, paReplace);
        vsapi->freeNode(src);
        return ;
    }

    int error;
    int radius = vsapi->propGetInt(in, "radius", 0, &error);
    if (error) {
        radius = 0;
    }

    auto map = vsapi->invoke(myself, "BM3D", in);
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

    auto map2 = vsapi->invoke(myself, "VAggregate", map);
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

    myself = plugin;

    configFunc(
        "com.wolframrhodium.bm3dcuda_rtc", "bm3dcuda_rtc",
        "BM3D algorithm implemented in CUDA (NVRTC)",
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
        "bm_error_s:data[]:opt;"
        "transform_2d_s:data[]:opt;"
        "transform_1d_s:data[]:opt;"
    };

    registerFunc("BM3D", bm3d_args, BM3DCreate, nullptr, plugin);

    registerFunc(
        "VAggregate",
        "clip:clip;"
        "src:clip;"
        "planes:int[];",
        VAggregateCreate, nullptr, plugin);

    registerFunc("BM3Dv2", bm3d_args, BM3Dv2Create, nullptr, plugin);
}
