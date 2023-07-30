cmake_minimum_required(VERSION 3.16)

project(BM3DHIP VERSION 2.6 LANGUAGES CXX)

find_package(HIP REQUIRED)

add_library(bm3dhip_source OBJECT source.cpp)
target_include_directories(bm3dhip_source PRIVATE 
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${hip_INCLUDE_DIRS})
target_compile_definitions(bm3dhip_source PRIVATE __HIP_PLATFORM_HCC__ __HIP_PLATFORM_AMD__)
set_target_properties(bm3dhip_source PROPERTIES 
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_library(bm3dhip_kernel OBJECT kernel.hip.cpp)
set_target_properties(bm3dhip_kernel PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)
target_compile_options(bm3dhip_kernel PRIVATE
    # https://www.llvm.org/docs/AMDGPUUsage.html
    -x hip -mno-wavefrontsize64 -mcumode -ffast-math -munsafe-fp-atomics
    --offload-arch=gfx1030 --offload-arch=gfx1032 --offload-arch=gfx1100)

add_library(bm3dhip SHARED)
target_link_libraries(bm3dhip bm3dhip_source bm3dhip_kernel hip::device)

install(TARGETS bm3dhip
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
