cmake_minimum_required(VERSION 3.20)

project(BM3DCUDA VERSION 2.6 LANGUAGES CXX CUDA)

add_library(bm3dcuda_source OBJECT source.cpp)
target_include_directories(bm3dcuda_source PRIVATE 
    ${VAPOURSYNTH_INCLUDE_DIRECTORY}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(bm3dcuda_source PROPERTIES 
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_library(bm3dcuda_kernel OBJECT kernel.cu)
set_target_properties(bm3dcuda_kernel PROPERTIES
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

add_library(bm3dcuda SHARED)
target_link_libraries(bm3dcuda bm3dcuda_source bm3dcuda_kernel)

install(TARGETS bm3dcuda
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
