cmake_minimum_required(VERSION 3.20)

project(BM3DCPU VERSION 2.5 LANGUAGES CXX)

if (ENABLE_VAPOURSYNTH)
    add_library(bm3dcpu SHARED source.cpp)
    target_include_directories(bm3dcpu PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})
    set_target_properties(bm3dcpu PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)

    if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM") OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "Intel") AND (CMAKE_SYSTEM_NAME STREQUAL "Linux"))

        target_compile_options(bm3dcpu PRIVATE "-mavx2")

    elseif ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR
            ((CMAKE_CXX_COMPILER_ID STREQUAL "Intel") AND (CMAKE_SYSTEM_NAME STREQUAL "Windows")))

        target_compile_options(bm3dcpu PRIVATE "/arch:AVX2")

    endif()
endif()

if (ENABLE_AVISYNTHPLUS)
    add_library(bm3dcpu_avs SHARED source_avs.cpp)
    target_include_directories(bm3dcpu_avs PRIVATE ${AVISYNTHPLUS_INCLUDE_DIRECTORY})
    set_target_properties(bm3dcpu_avs PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)

    if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM") OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "Intel") AND (CMAKE_SYSTEM_NAME STREQUAL "Linux"))

        target_compile_options(bm3dcpu_avs PRIVATE "-mavx2")

    elseif ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR
            ((CMAKE_CXX_COMPILER_ID STREQUAL "Intel") AND (CMAKE_SYSTEM_NAME STREQUAL "Windows")))

        target_compile_options(bm3dcpu_avs PRIVATE "/arch:AVX2")

    endif()
endif()
