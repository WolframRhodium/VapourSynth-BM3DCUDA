/*
 * Avisynth wrapper for BM3DCPU
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

#include "bm3d_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <avisynth.h>

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

    std::array<float, 3> sigma;
    std::array<int, 3> block_step;
    std::array<int, 3> bm_range;
    std::array<int, 3> ps_num;
    std::array<int, 3> ps_range;

    int radius;
    bool chroma;
    bool process[3]; // sigma != 0

    // final estimation
    bool final_() const {
        return static_cast<bool>(ref_node);
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
    const int center = radius;
    int temporal_width = 2 * radius + 1;
    const std::vector src_frames = [&](){
        std::vector<PVideoFrame> temp;
        temp.reserve(temporal_width);
        for (int i = -radius; i <= radius; ++i) {
            int clamped_n = std::clamp(n + i, 0, vi.num_frames - 1);
            temp.push_back(child->GetFrame(clamped_n, env));
        }
        return temp;
    }();
    const std::vector ref_frames = [&](){
        std::vector<PVideoFrame> temp;
        if (final_()) {
            temp.reserve(temporal_width);
            for (int i = -radius; i <= radius; ++i) {
                int clamped_n = std::clamp(n + i, 0, vi.num_frames - 1);
                temp.push_back(ref_node->GetFrame(clamped_n, env));
            }
        }
        return temp;
    }();
    const PVideoFrame & src_frame = src_frames[center];

    PVideoFrame dst = env->NewVideoFrameP(vi, const_cast<PVideoFrame *>(&src_frame));

    const auto cast_fp = []<typename T>(T * p) {
        if constexpr (std::is_const_v<T>)
            return reinterpret_cast<const float *>(p);
        else
            return reinterpret_cast<float *>(p);
    };

    if (chroma) {
        // planar YUV444PS input
        constexpr bool chroma = true;

        std::vector srcps = [&](){
            std::vector<const float *> temp;
            temp.reserve(3 * temporal_width);
            for (int plane = 0; plane < 3; ++plane) {
                for (auto & frame : src_frames) {
                    temp.push_back(cast_fp(frame->GetReadPtr(planes_id[plane])));
                }
            }
            return temp;
        }();

        std::array<float * __restrict, 3> dstps {
            const_cast<float * __restrict>(cast_fp(dst->GetWritePtr(planes_id[0]))),
            const_cast<float * __restrict>(cast_fp(dst->GetWritePtr(planes_id[1]))),
            const_cast<float * __restrict>(cast_fp(dst->GetWritePtr(planes_id[2]))),
        };

        const int width = src_frame->GetRowSize(PLANAR_Y) / sizeof(float);
        const int height = src_frame->GetHeight(PLANAR_Y);
        const int stride = src_frame->GetPitch(PLANAR_Y) / sizeof(float);

        float * buffer = [&]() -> float * {
            if (radius == 0) {
                auto p = env->Allocate(
                    sizeof(float) * stride * height * 2 * num_planes(chroma), 
                    32, 
                    AVS_POOLED_ALLOC
                );
                return cast_fp(p);
            } else {
                return nullptr;
            }
        }();

        if (radius == 0) {
            memset(buffer, 0, sizeof(float) * stride * height * 2 * num_planes(chroma));
        } else {
            for (const auto & dstp : dstps) {
                memset(dstp, 0, sizeof(float) * stride * height * 2 * temporal_width);
            }
        }

        if (!final_()) {
            constexpr bool final_ = false;
            if (radius == 0) {
                constexpr bool temporal = false;
                bm3d<temporal, chroma, final_>(
                    dstps, stride, srcps.data(), nullptr, 
                    width, height, 
                    sigma, block_step[0], bm_range[0], 
                    radius, ps_num[0], ps_range[0], 
                    buffer);
            } else {
                constexpr bool temporal = true;
                bm3d<temporal, chroma, final_>(
                    dstps, stride, srcps.data(), nullptr, 
                    width, height, 
                    sigma, block_step[0], bm_range[0], 
                    radius, ps_num[0], ps_range[0], 
                    nullptr);
            }
        } else {
            constexpr bool final_ = true;
            std::vector refps = [&](){
                std::vector<const float *> temp;
                temp.reserve(3 * temporal_width);
                for (int plane = 0; plane < 3; ++plane) {
                    for (auto & frame : ref_frames) {
                        temp.push_back(cast_fp(frame->GetReadPtr(planes_id[plane])));
                    }
                }
                return temp;
            }();
            if (radius == 0) {
                constexpr bool temporal = false;
                bm3d<temporal, chroma, final_>(
                    dstps, stride, srcps.data(), refps.data(), 
                    width, height, 
                    sigma, block_step[0], bm_range[0], 
                    radius, ps_num[0], ps_range[0], 
                    buffer);
            } else {
                constexpr bool temporal = true;
                bm3d<temporal, chroma, final_>(
                    dstps, stride, srcps.data(), refps.data(), 
                    width, height, 
                    sigma, block_step[0], bm_range[0], 
                    radius, ps_num[0], ps_range[0], 
                    nullptr);
            }
        }

        if (buffer) {
            env->Free(buffer);
        }
    } else {
        constexpr bool chroma = false;

        for (unsigned plane = 0; plane < std::size(planes_id); plane++) {
            if (!process[plane]) {
                continue;
            }

            std::vector srcps = [&](){
                std::vector<const float *> temp;
                temp.reserve(temporal_width);
                for (auto & frame : src_frames) {
                    temp.push_back(cast_fp(frame->GetReadPtr(planes_id[plane])));
                }
                return temp;
            }();

            std::array<float * __restrict, 1> dstps { 
                cast_fp(dst->GetWritePtr(planes_id[plane]))
            };

            const int width = src_frame->GetRowSize(planes_id[plane]) / sizeof(float);
            const int height = src_frame->GetHeight(planes_id[plane]);
            const int stride = src_frame->GetPitch(planes_id[plane]) / sizeof(float);

            float * const buffer = [&]() -> float * {
                if (radius == 0) {
                    auto p = env->Allocate(
                        sizeof(float) * stride * height * 2 * num_planes(chroma), 
                        32, 
                        AVS_POOLED_ALLOC);
                    return cast_fp(p);
                } else {
                    return nullptr;
                }
            }();

            if (radius == 0) {
                memset(buffer, 0, sizeof(float) * stride * height * 2 * num_planes(chroma));
            } else {
                for (const auto & dstp : dstps) {
                    memset(dstp, 0, sizeof(float) * stride * height * 2 * temporal_width);
                }
            }

            const std::array plane_sigma { sigma[plane] };

            if (!final_()) {
                constexpr bool final_ = false;
                if (radius == 0) {
                    constexpr bool temporal = false;
                    bm3d<temporal, chroma, final_>(
                        dstps, stride, srcps.data(), nullptr, 
                        width, height, 
                        plane_sigma, block_step[plane], bm_range[plane], 
                        radius, ps_num[plane], ps_range[plane], 
                        buffer);
                } else {
                    constexpr bool temporal = true;
                    bm3d<temporal, chroma, final_>(
                        dstps, stride, srcps.data(), nullptr, 
                        width, height, 
                        plane_sigma, block_step[plane], bm_range[plane], 
                        radius, ps_num[plane], ps_range[plane], 
                        nullptr);
                }
            } else {
                constexpr bool final_ = true;
                std::vector refps = [&](){
                    std::vector<const float *> temp;
                    temp.reserve(temporal_width);
                    for (auto & frame : ref_frames) {
                        temp.push_back(cast_fp(frame->GetReadPtr(planes_id[plane])));
                    }
                    return temp;
                }();
                if (radius == 0) {
                    constexpr bool temporal = false;
                    bm3d<temporal, chroma, final_>(
                        dstps, stride, srcps.data(), refps.data(), 
                        width, height, 
                        plane_sigma, block_step[plane], bm_range[plane], 
                        radius, ps_num[plane], ps_range[plane], 
                        buffer);
                } else {
                    constexpr bool temporal = true;
                    bm3d<temporal, chroma, final_>(
                        dstps, stride, srcps.data(), refps.data(), 
                        width, height, 
                        plane_sigma, block_step[plane], bm_range[plane], 
                        radius, ps_num[plane], ps_range[plane], 
                        nullptr);
                }
            }

            if (buffer) {
                env->Free(buffer);
            }
        }
    }

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
        env->ThrowError("BM3D_CPU: only 32bit float planar Y/YUV/RGB input supported");
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
        env->ThrowError("BM3D_CPU: Unknown sample type");
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
            env->ThrowError("BM3D_CPU: \"ref\" must be of the same format as \"clip\"");
        }
    }

    auto array_loader = []<typename T>(const AVSValue & arg, T default_value) {
        std::array<T, 3> ret;
        if (!arg.Defined()) {
            ret.fill(default_value);
        } else if (arg.IsArray()) {
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
        } else {
            if constexpr (std::is_same_v<T, float>) {
                ret.fill(static_cast<float>(arg.AsFloat(default_value)));
            } else if (std::is_same_v<T, int>) {
                ret.fill(arg.AsInt(default_value));
            }
        }
        return ret;
    };

    sigma = array_loader(args[2], 3.0f);
    for (unsigned i = 0; i < std::size(sigma); ++i) {
        if (sigma[i] < 0.0f) {
            env->ThrowError("BM3D_CPU: \"sigma\" must be non-negative");
        }
        if (sigma[i] < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        } else {
            process[i] = true;

            // assumes grayscale input, hard_thr = 2.7
            sigma[i] *= (3.0f / 4.0f) / 255.0f * 64.0f * (final_() ? 1.0f : 2.7f);
        }
    }

    block_step = array_loader(args[3], 8);
    for (const auto & x : block_step) {
        if (x <= 0 || x > 8) {
            env->ThrowError("BM3D_CPU: \"block_step\" must be in range [1, 8]");
        }
    }

    bm_range = array_loader(args[4], 8);
    for (const auto & x : bm_range) {
        if (x <= 0 || x > 8) {
            env->ThrowError("BM3D_CPU: \"bm_range\" must be in range [1, 8]");
        }
    }

    radius = args[5].AsInt(0);
    if (radius < 0) {
        env->ThrowError("BM3D_CPU: \"radius\" must be non-negative");
    } else if (radius > 0) {
        vi.height *= 2 * (2 * radius + 1);
    }

    ps_num = array_loader(args[6], 2);
    for (const auto & x : ps_num) {
        if (x <= 0 || x > 8) {
            env->ThrowError("BM3D_CPU: \"ps_num\" must be in range [1, 8]");
        }
    }

    ps_range = array_loader(args[7], 4);
    for (const auto & x : ps_range) {
        if (x <= 0) {
            env->ThrowError("BM3D_CPU: \"ps_range\" must be positive");
        }
    }

    chroma = args[8].AsBool(false);
    if (chroma && vi.pixel_type != VideoInfo::CS_YUV444PS) {
        env->ThrowError("BM3D_CPU: clip format must be YUV444PS when \"chroma\" is true");
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

    env->AddFunction("BM3D_CPU", 
        "c[ref]c"
        "[sigma]f[block_step]i[bm_range]"
        "i[radius]i[ps_num]i[ps_range]i[chroma]b"
        , BM3DFilter::Create, nullptr);

   return "BM3D algorithm (AVX2 version)";
}
