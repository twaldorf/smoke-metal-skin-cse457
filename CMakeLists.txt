cmake_minimum_required(VERSION 3.21)
project(rtiow1 LANGUAGES CXX)

set(CUDA_PATH /opt/cuda/bin)
set(EXE rtiow1)

# search for pkg-config
include (FindPkgConfig)
if(NOT PKG_CONFIG_FOUND)
    message (FATAL_ERROR "pkg-config not found")
endif ()

find_package(Boost REQUIRED)
if (NOT Boost_FOUND)
    message(FATAL_ERROR "You don't seem to have boost development libraries installed")
else ()
    include_directories (${BOOST_INCLUDE_DIRS})
    link_directories (${BOOST_LIBRARY_DIRS})
    link_libraries (${BOOST_LIBRARIES})
endif ()

# check for libpng
pkg_check_modules (LIBPNG libpng16 REQUIRED)
if(NOT LIBPNG_FOUND)
    message(FATAL_ERROR "You don't seem to have libpng16 development libraries installed")
else()
    include_directories (${LIBPNG_INCLUDE_DIRS})
    link_directories (${LIBPNG_LIBRARY_DIRS})
    link_libraries (${LIBPNG_LIBRARIES})
endif()

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
    set(CMAKE_CUDA_STANDARD 20)
    add_definitions(-DUSE_CUDA)
endif()
set(CMAKE_CXX_STANDARD 20)


set(SOURCE
        main.cpp src/vec3.cpp
        src/ray.cpp src/png.cpp
        src/sphere.cpp src/hitable_list.cpp
        src/camera.cpp src/util.hpp src/vec3.hpp
        src/ray.hpp src/png.hpp
        src/hitable.hpp src/sphere.hpp src/hitable_list.hpp
        src/camera.hpp src/material.cpp src/material.hpp src/util.cpp
        src/gpu/gpu_render.cu src/gpu/gpu_render.hpp src/world_gen.cpp src/world_gen.hpp)

if(CUDA_COMPILE)
    set(CUDA_SOURCE src/gpu/gpu_camera.cu src/gpu/gpu_camera.hpp
            src/gpu/gpu_render.cu src/gpu/gpu_render.hpp src/gpu/gpu_hitable.hpp
            src/gpu/gpu_hitable_list.cu src/gpu/gpu_hitable_list.hpp
            src/gpu/gpu_material.cu src/gpu/gpu_material.hpp
            src/gpu/gpu_ray.cu src/gpu/gpu_ray.hpp
            src/gpu/gpu_sphere.cu src/gpu/gpu_sphere.hpp
            src/gpu/gpu_vec3.cu src/gpu/gpu_vec3.hpp
            src/gpu/gpu_util.cu src/gpu/gpu_util.hpp)
    add_executable(${EXE} ${SOURCE} ${CUDA_SOURCE})
    message(${CUDAToolkit_LIBRARY_DIR})
    set_source_files_properties(${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
    target_compile_options(${EXE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -arch=sm_52
            -gencode=arch=compute_52,code=sm_52
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_80,code=sm_80
            -gencode=arch=compute_86,code=sm_86
            -gencode=arch=compute_87,code=sm_87
            -gencode=arch=compute_89,code=sm_89
            -gencode=arch=compute_89,code=compute_89
            -m64
            -Xptxas=-v
            -rdc=true
            #--gpu-code=compute_75,compute_80,sm_75,sm_80,sm_89
            >)
    set_property(TARGET ${EXE} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${EXE} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS} ${Boost_LIBRARIES} PRIVATE CUDA::cudart CUDA::cuda_driver)
else()
    add_executable(${EXE} ${SOURCE})
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS} ${Boost_LIBRARIES})
endif()
