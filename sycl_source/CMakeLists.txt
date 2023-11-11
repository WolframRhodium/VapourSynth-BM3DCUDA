if(WIN32)
    cmake_minimum_required(VERSION 3.23)
else()
    cmake_minimum_required(VERSION 3.20.5)
endif()

project(BM3DSYCL LANGUAGES CXX)

find_package(IntelSYCL REQUIRED CONFIG)

add_library(bm3dsycl SHARED source.cpp kernel.cpp)

target_include_directories(bm3dsycl PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})

set_target_properties(bm3dsycl PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_sycl_to_target(
    TARGET bm3dsycl
    SOURCES source.cpp kernel.cpp)

# for aot compilation for gpus
target_compile_options(bm3dsycl PRIVATE -fsycl -fsycl-targets=spir64,spir64_gen)

install(TARGETS bm3dsycl
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
