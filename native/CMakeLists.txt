cmake_minimum_required(VERSION 3.16)
project(auto_wall_geometry LANGUAGES CXX)

add_library(auto_wall_geometry SHARED trace_skeleton.cpp)
target_compile_features(auto_wall_geometry PRIVATE cxx_std_17)
set_target_properties(auto_wall_geometry PROPERTIES PREFIX "")
set(output "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/_native")
foreach(kind LIBRARY RUNTIME ARCHIVE)
    set_target_properties(auto_wall_geometry PROPERTIES
        ${kind}_OUTPUT_DIRECTORY "${output}"
        ${kind}_OUTPUT_DIRECTORY_RELEASE "${output}")
endforeach()
if(MSVC)
    target_compile_options(auto_wall_geometry PRIVATE /W4 /EHsc)
    set_property(TARGET auto_wall_geometry PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_compile_options(auto_wall_geometry PRIVATE -Wall -Wextra -Wpedantic)
endif()
