# VapourSynth-BM3DCUDA

Copyright© 2021 WolframRhodium

BM3D denoising filter for VapourSynth, implemented in CUDA

## Description

Please check [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D).

## Requirements

- CPU with AVX support.

- CUDA-enabled GPU(s) of [compute capability](https://developer.nvidia.com/cuda-gpus) 5.0 or higher.

- GPU driver 450 or newer.

The minimum requirement on compute capability is 3.0, which requires manual compilation (specifying nvcc flag `-gencode arch=compute_30,code=sm_30`).

## Parameters

```python3
bm3dcuda.BM3D(clip clip[, clip ref=None, float[] sigma=3.0, int[] block_step=8, int[] bm_range=9, int radius=0, int[] ps_num=2, int[] ps_range=4, bint chroma=False, int device_id=0, bool fast=True])
```

- clip:<br />
    The input clip. Must be of 32 bit float format. Each plane is denoised separately if `chroma` is set to `False`.

- ref:<br />
    The reference clip. Must be of the same format, width, height, number of frames as `clip`.<br />
    Used in block-matching and as the reference in empirical Wiener filtering, i.e. `bm3d.Final / bm3d.VFinal`.

- sigma:<br />
    The strength of denoising for each plane.<br />
    The strength is similar (but not strictly equal) as `VapourSynth-BM3D` due to differences in implementation. (coefficient normalization is not implemented, for example)<br />
    Default `[3,3,3]`.

- block_step, bm_range, radius, ps_num, ps_range:<br />
    Same as those in `VapourSynth-BM3D`.<br />
    If `chroma` is set to `True`, only the first value is in effect.<br />
    Otherwise an array of values may be specified for each plane.

- chroma:<br />
    CBM3D algorithm. `clip` must be of `YUV444PS` format.<br />
    Y channel is used in block-matching of chroma channels.
    Default `False`.

- device_id:<br />
    Set GPU to be used.<br />
    Default `0`.

- fast:<br />
    Multi-threaded copy between CPU and GPU at the expense of 4x memory consumption.<br />
    Default `True`.

## Notes

- `bm3d.VAggregate` should be called after temporal filtering, as in `VapourSynth-BM3D`.

## Statistics

GPU memory consumptions:<br />
`(ref ? 4 : 3) * (chroma ? 3 : 1) * (fast ? 4 : 1) * (2 * radius + 1) * size_of_a_single_frame`

## Compilation on Linux

- g++ 11 (or higher) is required to compile `source.cpp`, while nvcc 11.3 only supports g++ 10 or older.

- Unused nvcc flags may be removed. [Documentation for -gencode](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-generate-code)

```
cd source

nvcc kernel.cu -c --use_fast_math --std=c++17 -gencode arch=compute_50,code=\"sm_50,compute_50\" -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=\"sm_86,compute_86\" -t 0 --compiler-bindir g++-10 -o kernel.o

g++-11 source.cpp kernel.o -shared -fPIC -I/usr/local/cuda-11.3/include -I/usr/local/include -L/usr/local/cuda-11.3/lib64 -lcudart_static --std=c++20 -march=native -O3 -o libbm3dcuda.so
```
