# Minimum CMake version required, we'll just use the latest version.
cmake_minimum_required(VERSION 3.22)
# Project name, version and description
project(test VERSION 1.0 DESCRIPTION "Polarity Test")

set(sources ${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/* )

find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)
find_package(SDL2 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_executable(${PROJECT_NAME} ${sources})

target_link_libraries( ${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${Armadillo_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)

# Set C++ version
set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
)

# Set Compiler flags
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
            /W4
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
            -Wall -Wextra -O2 -march=native
    )
endif()