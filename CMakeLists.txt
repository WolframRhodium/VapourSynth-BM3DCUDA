cmake_minimum_required(VERSION 3.20)

project(VapourSynth-BM3DCUDA VERSION 2.6 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")
set(USE_NVRTC_STATIC ON CACHE BOOL "Whether to use NVRTC static library")
set(ENABLE_CPU ON CACHE BOOL "Enable CPU build")
set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA and CUDA RTC build")

if (ENABLE_CUDA)
    add_subdirectory(source)
    add_subdirectory(rtc_source)
endif()

if (ENABLE_CPU)
    add_subdirectory(cpu_source)
endif()

