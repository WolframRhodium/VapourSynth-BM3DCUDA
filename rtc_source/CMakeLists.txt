cmake_minimum_required(VERSION 3.20)

project(BM3DCUDA_RTC VERSION 2.5 LANGUAGES CXX)

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(WARNING "NVRTC is not available for 32-bit system")
else()
    FIND_PACKAGE(CUDAToolkit REQUIRED)

    add_library(bm3dcuda_rtc SHARED source.cpp)
    target_include_directories(bm3dcuda_rtc PRIVATE 
        ${VAPOURSYNTH_INCLUDE_DIRECTORY} 
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(bm3dcuda_rtc PROPERTIES 
        CXX_EXTENSIONS OFF
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)
    target_link_libraries(bm3dcuda_rtc PRIVATE CUDA::cuda_driver CUDA::nvrtc)
endif()
