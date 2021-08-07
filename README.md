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

The `_rtc` version requires compute capability 3.5 or higher, GPU driver 465 or newer and has dependencies on `nvrtc64_112_0.dll/libnvrtc.so.11.2` and `nvrtc-builtins64_114.dll/libnvrtc-builtins.so.11.4.50`.

The `cpu` version does not require any external libraries but requires AVX2 support on CPU in addition.

## Parameters

```python3
{bm3dcuda, bm3dcuda_rtc, bm3dcpu}.BM3D(clip clip[, clip ref=None, float[] sigma=3.0, int[] block_step=8, int[] bm_range=9, int radius=0, int[] ps_num=2, int[] ps_range=4, bint chroma=False, int device_id=0, bool fast=True, int extractor_exp=0])
```

- clip:

    The input clip. Must be of 32 bit float format. Each plane is denoised separately if `chroma` is set to `False`. Data of unprocessed planes is undefined.

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

- The standard version and the `_rtc` version has an experimental parameter:
    - **_unsafe_**: (bool)

        Performs unsafe memory optimization that reduces memory consumption of V-BM3D. It is generally non-deterministic and is in conflict with parameter `extractor_exp`.

        `bm3d.VAggregate` **is not required to be called explicitly in this mode**, as it is handled by the plugin itself.

        Default to false.

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

## Compilation on Linux

### Standard version
- g++ 11 (or higher) and nvcc 11.4.1 is required.

- Unused nvcc flags may be removed. [Documentation for -gencode](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-generate-code)

```
cd source

nvcc kernel.cu -o kernel.o -c --use_fast_math --std=c++17 -gencode arch=compute_50,code=\"sm_50,compute_50\" -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=\"sm_86,compute_86\" -t 0

g++-11 source.cpp kernel.o -o libbm3dcuda.so -shared -fPIC -I/usr/local/cuda-11.4/include -L/usr/local/cuda-11.4/lib64 -lcudart_static --std=c++20 -march=native -O3
```

### RTC version
- g++ 11 or clang 12 (or higher) is required.

```
cd rtc_source

g++-11 source.cpp -o libbm3dcuda_rtc.so -shared -fPIC -I /usr/local/cuda-11.4/include -L /usr/local/cuda-11.4/lib64 -lnvrtc -lcuda -Wl,-rpath,/usr/local/cuda-11.4/lib64 --std=c++20 -march=native -O3
```

### CPU version
```
cd cpu_source

g++ source.cpp -o libbm3dcpu.so -shared -fPIC --std=c++17 -march=native -O3 -ffast-math
```
