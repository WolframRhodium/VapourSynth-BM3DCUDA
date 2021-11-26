cmake_minimum_required(VERSION 3.20)

project(BM3DCPU VERSION 2.6 LANGUAGES CXX)

add_library(bm3dcpu SHARED source.cpp)
target_include_directories(bm3dcpu PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})
set_target_properties(bm3dcpu PROPERTIES 
    CXX_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))

    target_compile_options(bm3dcpu PRIVATE "-mavx2;-mfma;-mpopcnt")

elseif (((CMAKE_CXX_COMPILER_ID STREQUAL "Intel") OR (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")) AND
        (CMAKE_SYSTEM_NAME STREQUAL "Linux"))

    target_compile_options(bm3dcpu PRIVATE "-march=core-avx2")

elseif ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR
        (((CMAKE_CXX_COMPILER_ID STREQUAL "Intel") OR (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")) AND
         (CMAKE_SYSTEM_NAME STREQUAL "Windows")))

    target_compile_options(bm3dcpu PRIVATE "/arch:AVX2")

endif()

install(TARGETS bm3dcpu
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
