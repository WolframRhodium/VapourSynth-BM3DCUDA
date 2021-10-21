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

- clip:

    The input clip. Must be of 32 bit float format. Each plane is denoised separately if `chroma` is set to `False`. Data of unprocessed planes is undefined. Frame properties of the output clip are copied from it.

- ref:

    The reference clip. Must be of the same format, width, height, number of frames as `clip`.

    Used in block-matching and as the reference in empirical Wiener filtering, i.e. `bm3d.Final` / `bm3d.VFinal`:

    ```python3
    basic = core.{bm3dcpu, bm3dcuda, bm3dcuda_rtc}.BM3D(clip, radius=0)
    final = core.{bm3d...}.BM3D(basic, ref=src, radius=0)

    vbasic = core.{bm3d...}.BM3D(src, radius=radius_nonzero).bm3d.VAggregate(radius=radius_nonzero)
    vfinal = core.{bm3d...}.BM3D(src, ref=vbasic, radius=r).bm3d.VAggregate(radius=r)
    ```

    corresponds to the followings (ignoring color space handling and other differences in implementation), respectively

    ```python3
    basic = core.bm3d.Basic(clip)
    final = core.bm3d.Final(basic, ref=src)

    vbasic = core.bm3d.VBasic(src, radius=r).bm3d.VAggregate(radius=r)
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

    **Note2**: Lowering the value of "blocking_step" will be useful in reducing blocking artifacts at the cost of slower processing.

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

## Notes

- `bm3d.VAggregate` should be called after temporal filtering, as in `VapourSynth-BM3D`.

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

## Compilation
- The CMake configuration of `BM3DCUDA_RTC` links to NVRTC static library by default, which requires CUDA 11.5 or later.

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_FLAGS="--threads 0 --use_fast_math -Wno-deprecated-gpu-targets" -D CMAKE_CUDA_ARCHITECTURES="50;61-real;75-real;86"

cmake --build build --config Release
```
