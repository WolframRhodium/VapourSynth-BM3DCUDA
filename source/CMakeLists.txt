cmake_minimum_required(VERSION 3.20)

project(BM3DCUDA VERSION 2.6 LANGUAGES CXX CUDA)

if (ENABLE_VAPOURSYNTH OR ENABLE_AVISYNTHPLUS)
    add_library(bm3dcuda_kernel OBJECT kernel.cu)
    set_target_properties(bm3dcuda_kernel PROPERTIES
        CUDA_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
endif()

if (ENABLE_VAPOURSYNTH)
    add_library(bm3dcuda_source OBJECT source.cpp)
    target_include_directories(bm3dcuda_source PRIVATE
        ${VAPOURSYNTH_INCLUDE_DIRECTORY}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(bm3dcuda_source PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)

    add_library(bm3dcuda SHARED)
    target_link_libraries(bm3dcuda bm3dcuda_source bm3dcuda_kernel)
endif()

if (ENABLE_AVISYNTHPLUS)
    add_library(bm3dcuda_source_avs OBJECT source_avs.cpp)
    target_include_directories(bm3dcuda_source_avs PRIVATE
        ${AVISYNTHPLUS_INCLUDE_DIRECTORY}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(bm3dcuda_source_avs PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)

    add_library(bm3dcuda_avs SHARED)
    target_link_libraries(bm3dcuda_avs bm3dcuda_source_avs bm3dcuda_kernel)
endif()
