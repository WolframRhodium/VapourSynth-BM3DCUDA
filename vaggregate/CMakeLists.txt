cmake_minimum_required(VERSION 3.20)

project(BM3D_VAggregate VERSION 2.6 LANGUAGES CXX)

if (ENABLE_AVISYNTHPLUS)
    add_library(bm3d_vaggregate_avs SHARED source_avs.cpp)
    target_include_directories(bm3d_vaggregate_avs PRIVATE ${AVISYNTHPLUS_INCLUDE_DIRECTORY})
    set_target_properties(bm3d_vaggregate_avs PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON)

    install(TARGETS bm3d_vaggregate_avs
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
