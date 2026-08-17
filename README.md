# VapourSynth-BM3DCUDA

Copyright© 2021 WolframRhodium

BM3D denoising filter for VapourSynth, implemented in CUDA.

## Description

- Please check [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D).

- The `_rtc` version compiles GPU code at runtime, which might runs faster than standard version at the cost of a slight overhead.

- The `cpu` version is implemented in AVX and AVX2 intrinsics, serves as a reference implementation on CPU. However, _bitwise identical_ outputs are not guaranteed across CPU and CUDA implementations.

## Requirements

- CPU with AVX support.

- CUDA-enabled GPU(s) of [compute capability](https://developer.nvidia.com/cuda-gpus) 5.0 or higher (Maxwell+).

- GPU driver 450 or newer.

The minimum requirement on compute capability is 3.5, which requires manual compilation (specifying nvcc flag `-gencode arch=compute_35,code=sm_35`).

The `cpu` version does not require any external libraries but requires AVX2 support on CPU in addition.

## Parameters

```python3
{bm3dcuda, bm3dcuda_rtc, bm3dcpu}.BM3D(clip clip[, clip ref=None, float[] sigma=3.0, int[] block_step=8, int[] bm_range=9, int radius=0, int[] ps_num=2, int[] ps_range=4, bint chroma=False, int device_id=0, bool fast=True, int extractor_exp=0])
```

CUDA and CUDA RTC provide execution selection on `BM3Dv2`:

```python3
{bm3dcuda, bm3dcuda_rtc}.BM3Dv2(clip[, ..., temporal_mode="rolling", rolling_chunk=4, rolling_cache_chunks=1, rolling_cache_limit=16])
```

- clip:

    The input clip. Must be of 32 bit float format. Each plane is denoised separately if `chroma` is set to `False`. Data of unprocessed planes is undefined for the public intermediate-frame path; rolling `BM3Dv2` copies them from the source. Frame properties of the output clip are copied from it.

- ref:

    The reference clip. Must be of the same format, width, height, number of frames as `clip`.

    Used in block-matching and as the reference in empirical Wiener filtering, i.e. `bm3d.Final` / `bm3d.VFinal`:

    ```python3
    basic = core.{bm3dcpu, bm3dcuda, bm3dcuda_rtc}.BM3D(src, radius=0)
    final = core.{bm3d...}.BM3D(src, ref=basic, radius=0)

    vbasic = core.{bm3d...}.BM3D(src, radius=radius_nonzero).bm3d.VAggregate(radius=radius_nonzero)
    vfinal = core.{bm3d...}.BM3D(src, ref=vbasic, radius=r).bm3d.VAggregate(radius=r)
    
    # alternatively, using the v2 interface
    basic_or_vbasic = core.{bm3dcpu, bm3dcuda, bm3dcuda_rtc}.BM3Dv2(src, radius=r)
    final_or_vfinal = core.{bm3d...}.BM3Dv2(src, ref=basic_or_vbasic, radius=r)
    ```

    corresponds to the followings (ignoring color space handling and other differences in implementation), respectively

    ```python3
    basic = core.bm3d.Basic(clip)
    final = core.bm3d.Final(basic, ref=basic)

    vbasic = core.bm3d.VBasic(src, radius=r).bm3d.VAggregate(radius=r, sample=1)
    vfinal = core.bm3d.VFinal(src, ref=vbasic, radius=r).bm3d.VAggregate(radius=r)
    ```

- sigma:
    The strength of denoising for each plane.

    The strength is similar (but not strictly equal) as `VapourSynth-BM3D` due to differences in implementation. (coefficient normalization is not implemented, for example)

    Default `[3,3,3]`.

- block_step, bm_range, radius, ps_num, ps_range:

    Same as those in `VapourSynth-BM3D`.

    If `chroma` is set to `True`, only the first value is in effect.

    Otherwise an array of values may be specified for each plane (except `radius`).
    
    **Note**: It is generally not recommended to take a large value of `ps_num` as current implementations do not take duplicate block-matching candidates into account during temporary searching, which may leads to regression in denoising quality. This issue is not present in `VapourSynth-BM3D`.

    **Note2**: Lowering the value of "block_step" will be useful in reducing blocking artifacts at the cost of slower processing.

- chroma:

    CBM3D algorithm. `clip` must be of `YUV444PS` format.

    Y channel is used in block-matching of chroma channels.

    Default `False`.

- device_id:

    Set GPU to be used.

    Default `0`.

- fast:

    Multi-threaded copy between CPU and GPU at the expense of 4x memory consumption.

    Default `True`.

- extractor_exp:

    Used for deterministic (bitwise) output. This parameter is not present in the `cpu` version since the implementation always produces deterministic output.

    [Pre-rounding](https://ieeexplore.ieee.org/document/6545904) is employed for associative floating-point summation.

    The value should be a positive integer not less than 3, and may need to be higher depending on the source video and filter parameters.

    Default `0`. (non-determinism)

- temporal_mode, rolling_chunk, rolling_cache_chunks, rolling_cache_limit:

    `temporal_mode` selects temporal execution for `radius > 0`. `"rolling"`
    is the default and uses one CUDA stream and one resource set to build
    aligned output chunks. `"legacy"` composes public `BM3D` and
    `VAggregate`. These are the only accepted temporal modes. `fast` is
    accepted in rolling mode for API compatibility but has no effect.

    `rolling_chunk` sets the rolling chunk size in the range `[1, 64]` and
    defaults to `4`. It may only be supplied with `temporal_mode="rolling"`.
    Rolling mode requires `radius > 0`.

    `rolling_cache_chunks` sets the number of completed chunks retained in the
    rolling output cache, in the range `[1, 64]`, and defaults to `1`. It may
    only be supplied with `temporal_mode="rolling"`. A value greater than one
    keeps an LRU cache of final VapourSynth frames for random access; it does
    not allocate another CUDA resource, stream, or graph and does not add GPU
    work on cache hits. The tradeoff is host frame memory proportional to
    `rolling_chunk * rolling_cache_chunks` (plus normal VapourSynth frame
    overhead).

    `rolling_cache_limit` bounds adaptive cache growth in the range `[1, 64]`
    and defaults to `16`. When `rolling_cache_chunks` is omitted, rolling
    grows its cache up to this limit after repeated chunk reuse. Supplying
    `rolling_cache_chunks` explicitly keeps a fixed cache size unless a larger
    `rolling_cache_limit` is also requested.

## Notes

- `bm3d.VAggregate` should be called after temporal filtering, as in `VapourSynth-BM3D`. Alternatively, you may use the `BM3Dv2()` interface for both spatial and temporal denoising in one step.

- For CUDA and CUDA RTC, `BM3Dv2(radius > 0)` uses rolling temporal BM3D and
  final GPU aggregation by default. The public `BM3D(radius > 0)` and
  `VAggregate` interfaces retain their existing intermediate-frame behavior.

- CUDA rolling chunks are aligned by
  `floor(frame / rolling_chunk) * rolling_chunk`. The filter retains one
  completed chunk of final VapourSynth frames by default. Requests inside that
  cache are hits; a request outside it recomputes the aligned chunk. Set
  `rolling_cache_chunks > 1` to retain several completed chunks in an LRU
  cache and reduce repeated work for bounded random-access working sets. A
  partial final chunk uses replicated end frames internally and caches only
  valid output frames.

- Rolling uploads one `rolling_chunk + 4 * radius` source slab (and one
  reference slab for final estimation), computes
  `rolling_chunk + 2 * radius` temporal centers in order, and transfers only
  normalized output frames to the host. Larger chunks amortize halo work but
  increase GPU accumulators, pinned staging memory, retained frame memory, and
  latency for a random-access miss.

- For mostly linear rendering, `rolling_chunk=16, rolling_cache_chunks=1`
  favors throughput. For interactive access or a small downstream temporal
  window, `rolling_chunk=4, rolling_cache_chunks=2` is a lower-latency starting
  point. Fully unpredictable random access can still cause rolling chunk
  misses; increase `rolling_cache_limit` when the working set is bounded.

- RTC rolling uses one device resource, one stream, and one Driver API graph.
  A completed chunk is retained in a frame cache by default; cache misses
  stage the source/reference halo, serialize graph execution, and insert the
  result into the bounded LRU cache. `rolling_cache_chunks` changes only the
  number of retained final chunks.
  `fast` is accepted for API compatibility but does not change rolling's
  resource count. Partial final chunks cache only valid output frames.

- `VAggregate(planes=...)` accepts each plane at most once. Plane indices must be non-negative and smaller than the number of planes in `src`.

- The `_rtc` version has three additional experimental parameters:

    - bm_error_s: (string)

        Specify cost for block similarity measurement.

        Currently implemented costs: 
        `SSD` (Sum of Squared Differences), 
        `SAD` (Sum of Absolute Differences), 
        `ZSSD` (Zero-mean SSD), 
        `ZSAD` (Zero-mean SAD), 
        `SSD/NORM`.

        Default `SSD`.

    - transform_2d_s/transform_1d_s: (string)

        Specify type of transform.

        Currently implemented transforms: 
        `DCT` (Discrete Cosine Transform), 
        `Haar` (Haar Transform), 
        `WHT` (Walsh–Hadamard Transform), 
        `Bior1.5` (transform based on a bi-orthogonal spline wavelet).

        Default `DCT`.

    These features are not implemented in the standard version due to performance and binary size concerns.

## Statistics

GPU memory consumptions:

`(ref ? 4 : 3) * (chroma ? 3 : 1) * (fast ? 4 : 1) * (2 * radius + 1) * size_of_a_single_frame`

For the rolling CUDA/RTC `BM3Dv2(radius > 0)` path, one stream holds a
source slab, one `2 * radius + 1`-slice scratch result, and
`rolling_chunk` weighted accumulators. Pinned host memory holds the source slab
and normalized chunk outputs. The filter additionally retains
`rolling_cache_chunks * rolling_chunk` normal VapourSynth output frames (or
fewer for the final partial chunk).

## Compilation
- The CMake configuration of `BM3DCUDA_RTC` links to NVRTC static library by default, which requires CUDA 11.5 or later.

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_FLAGS="--threads 0 --use_fast_math -Wno-deprecated-gpu-targets" -D CMAKE_CUDA_ARCHITECTURES="50;61-real;75-real;86"

cmake --build build --config Release
```
