/*
* VAggregate for BM3D_*_AVS
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

class VAggregateFilter : public GenericVideoFilter {
    std::vector<int> planes_id;
    int radius;

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
    int temporal_width = 2 * radius + 1;

    const std::vector srcs = [&](){
        std::vector<PVideoFrame> temp;

        temp.reserve(temporal_width);

        for (int i = -radius; i <= radius; ++i) {
            int clamped_n = std::clamp(n + i, 0, vi.num_frames - 1);
            temp.push_back(child->GetFrame(clamped_n, env));
        }

        return temp;
    }();

    PVideoFrame dst = env->NewVideoFrameP(vi, const_cast<PVideoFrame *>(&srcs[radius]));

    std::vector<float> buffer;
    buffer.reserve(2 * vi.height * vi.width);

    for (unsigned plane = 0; plane < std::size(planes_id); ++plane) {
        int width {
            vi.IsYUV() && plane != 0 ? 
            vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U) : 
            vi.width
        };
        int height {
            vi.IsYUV() && plane != 0 ? 
            vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U) : 
            vi.height
        };

        memset(buffer.data(), 0, 2 * height * vi.width * sizeof(float));

        for (int i = 0; i < temporal_width; ++i) {
            int stride = srcs[i]->GetPitch(planes_id[plane]) / sizeof(float);

            const float * agg_src {
                reinterpret_cast<const float *>(srcs[i]->GetReadPtr(planes_id[plane])) + 
                (temporal_width - 1 - i) * 2 * height * stride
            };
            float * agg_dst = buffer.data();

            for (int y = 0; y < 2 * height; ++y) {
                for (int x = 0; x < width; ++x) {
                    agg_dst[x] += agg_src[x];
                }
                agg_dst += vi.width;
                agg_src += stride;
            }
        }

        auto dstp = reinterpret_cast<float *>(dst->GetWritePtr(planes_id[plane]));
        Aggregation(
            dstp, dst->GetPitch(planes_id[plane]) / sizeof(float), 
            buffer.data(), vi.width, 
            width, height);
    }

    return dst;
}

VAggregateFilter::VAggregateFilter(AVSValue args, IScriptEnvironment* env)
    : GenericVideoFilter(args[0].AsClip())
{
    env->CheckVersion(8);

    if (
        vi.BitsPerComponent() != 32 || 
        !vi.IsPlanar() || 
        !(vi.IsY() || vi.IsYUV() || vi.IsRGB())
        ) {
        env->ThrowError("BM3D_VAggregate: only 32bit float planar Y/YUV/RGB input supported");
    }

    if (vi.IsY()) {
        planes_id = { PLANAR_Y };
    } else if (vi.IsYUV()) {
        planes_id = { PLANAR_Y, PLANAR_U, PLANAR_V };
    } else if (vi.IsRGB()) {
        planes_id = { PLANAR_R, PLANAR_G, PLANAR_B };
    } else {
        env->ThrowError("BM3D_VAggregate: Unknown sample type");
    }

    radius = args[1].AsInt(0);
    if (radius <= 0) {
        env->ThrowError("BM3D_VAggregate: \"radius\" must be positive");
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

    env->AddFunction("BM3D_VAggregate", 
        "c[radius]i", 
        VAggregateFilter::Create, nullptr);

    return "BM3D_VAggregate";
}
