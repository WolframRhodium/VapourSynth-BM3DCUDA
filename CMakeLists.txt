cmake_minimum_required(VERSION 3.20)

project(VapourSynth-BM3DCUDA VERSION 2.6 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

set(USE_NVRTC_STATIC ON CACHE BOOL "Whether to use NVRTC static library")

add_subdirectory(source)
add_subdirectory(rtc_source)
add_subdirectory(cpu_source)
