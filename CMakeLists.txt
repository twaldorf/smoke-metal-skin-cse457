cmake_minimum_required(VERSION 3.21)
project(rtiow1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CUDA_PATH /opt/cuda/bin)
set(EXE rtiow1)


# search for pkg-config
include (FindPkgConfig)
if(NOT PKG_CONFIG_FOUND)
    message (FATAL_ERROR "pkg-config not found")
endif ()

find_package(CUDAToolkit QUIET)
message("CUDA found: " ${CUDAToolkit_FOUND})
if(CUDAToolkit_FOUND)
    option(CUDA_COMPILE "CUDA Support" ON)
else()
    option(CUDA_COMPILE "CUDA Support" OFF)
endif()
message("CUDA support: " ${CUDA_COMPILE})
if(CUDA_COMPILE)
    add_definitions(-DUSE_CUDA)
endif()


set(SOURCE
        main.cpp src/vec3.cpp src/colour.cpp
        src/ray.cpp src/png.cpp
        src/sphere.cpp src/hittable_list.cpp
        src/camera.cpp src/utl.hpp src/vec3.hpp
        src/colour.hpp src/ray.hpp src/png.hpp
        src/hittable.hpp src/sphere.hpp src/hittable_list.hpp
        src/camera.hpp src/material.cpp src/material.hpp src/util.cpp)

# check for libpng
pkg_check_modules (LIBPNG libpng16 REQUIRED)
if (NOT LIBPNG_FOUND)
    message(FATAL_ERROR "You don't seem to have libpng16 development libraries installed")
else ()
    include_directories (${LIBPNG_INCLUDE_DIRS})
    link_directories (${LIBPNG_LIBRARY_DIRS})
    link_libraries (${LIBPNG_LIBRARIES})
endif ()

add_executable(${EXE} ${SOURCE})

if(CUDA_COMPILE)
    message(${CUDAToolkit_LIBRARY_DIR})
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS} PRIVATE CUDA::cudart CUDA::cuda_driver)
else()
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS})
endif()
