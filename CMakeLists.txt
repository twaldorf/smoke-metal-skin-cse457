cmake_minimum_required(VERSION 3.21)
project(rtiow1 LANGUAGES CXX)

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
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_definitions(-DUSE_CUDA)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()


set(SOURCE
        main.cpp src/vec3.cpp src/colour.cpp
        src/ray.cpp src/png.cpp
        src/sphere.cpp src/hitable_list.cpp
        src/camera.cpp src/util.hpp src/vec3.hpp
        src/colour.hpp src/ray.hpp src/png.hpp
        src/hitable.hpp src/sphere.hpp src/hitable_list.hpp
        src/camera.hpp src/material.cpp src/material.hpp src/util.cpp src/cuda.cu src/cuda.hpp src/world_gen.cpp src/world_gen.hpp)

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
    set_property(TARGET ${EXE} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${EXE} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CUDA)
    target_compile_options(${EXE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --gpu-architecture=sm_89
            -Xptxas=-v
            -rdc=true
            #--gpu-code=compute_80,sm_80,sm_89
            >)
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS} PRIVATE CUDA::cudart CUDA::cuda_driver)
else()
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS})
endif()
