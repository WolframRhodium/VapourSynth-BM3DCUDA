cmake_minimum_required(VERSION 3.20)

project(VapourSynth-BM3DCUDA VERSION 2.6 LANGUAGES CXX)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

set(USE_NVRTC_STATIC ON CACHE BOOL "Whether to use NVRTC static library")

option(ENABLE_VAPOURSYNTH "Enable VapourSynth build" ON)
set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

option(ENABLE_AVISYNTHPLUS "Enable AviSynth+ build" OFF)
set(AVISYNTHPLUS_INCLUDE_DIRECTORY "" CACHE PATH "Path to AviSynth+ headers")

add_subdirectory(source)
add_subdirectory(rtc_source)
add_subdirectory(cpu_source)

if (ENABLE_AVISYNTHPLUS)
    add_subdirectory(vaggregate)
endif()
