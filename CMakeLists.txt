cmake_minimum_required(VERSION 3.21)
project(rtiow1 LANGUAGES C CXX)

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
    #change if different on local env
    enable_language(CUDA)
    set(OptiX_INSTALL_DIR /opt/optix)
    set(owl_dir ${PROJECT_SOURCE_DIR}/owl)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
    add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
    include_directories(${OWL_INCLUDES})
    include(embed_ptx)
    find_package(OptiX REQUIRED)
    message("OPTIX found: " ${OptiX_FOUND})
    option(CUDA_COMPILE "CUDA Support" ON)
    set(OPTIX_FORCE_DEPRECATED_LAUNCHER 1)
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

#we want to compile this as CUDA if
set(ENTRY main.cpp)
set(SOURCE
        src/vec3.cpp
        src/ray.cpp src/png.cpp
        src/sphere.cpp src/hitable_list.cpp
        src/camera.cpp src/util.hpp src/vec3.hpp
        src/ray.hpp src/png.hpp
        src/hitable.hpp src/sphere.hpp src/hitable_list.hpp
        src/camera.hpp src/material.cpp src/material.hpp src/util.cpp
        src/gpu/gpu_render.cu src/gpu/gpu_render.cuh src/world_gen.cpp
        src/world_gen.hpp src/constant_medium.cpp src/tri.cpp src/tri.hpp
        src/render.cpp src/render.hpp src/tri.cpp src/tri.hpp src/texture.cpp)
set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CXX)

if(CUDA_COMPILE)
    set(OPTIX_SOURCE src/gpu/optix/optix_render.cu)
    set_source_files_properties(${OPTIX_SOURCE} PROPERTIES LANGUAGE CUDA)
    embed_ptx(OUTPUT_TARGET rt_ptx
            PTX_LINK_LIBRARIES owl::owl
            SOURCES ${OPTIX_SOURCE})
    set(CUDA_SOURCE src/gpu/gpu_camera.cu src/gpu/gpu_camera.cuh
            src/gpu/gpu_render.cu src/gpu/gpu_render.cuh src/gpu/gpu_hitable.cuh
            src/gpu/gpu_hitable_list.cu src/gpu/gpu_hitable_list.cuh
            src/gpu/gpu_material.cu src/gpu/gpu_material.cuh
            src/gpu/gpu_ray.cu src/gpu/gpu_ray.cuh
            src/gpu/gpu_sphere.cu src/gpu/gpu_sphere.cuh
            src/gpu/gpu_vec3.cu src/gpu/gpu_vec3.cuh
            src/gpu/gpu_util.cu src/gpu/gpu_util.cuh
            src/gpu/gpu_constant_medium.cu src/gpu/gpu_constant_medium.cuh)
    set(OPTIX_HOST src/gpu/optix/optix_host.cpp src/gpu/optix/tiny_obj_loader.hpp)
    set_source_files_properties(${ENTRY} ${OPTIX_HOST} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXE} ${ENTRY} ${SOURCE} ${CUDA_SOURCE} ${OPTIX_HOST})
    message(${CUDAToolkit_LIBRARY_DIR})
    set_source_files_properties(${ENTRY} ${OPTIX_HOST} ${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
    target_compile_options(${EXE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            #--generate-line-info
            #-arch=sm_89
            #-arch=sm_52
            #-gencode=arch=compute_52,code=sm_52
            #-gencode=arch=compute_60,code=sm_60
            #-gencode=arch=compute_61,code=sm_61
            #-gencode=arch=compute_70,code=sm_70
            #-gencode=arch=compute_75,code=sm_75
            #-gencode=arch=compute_80,code=sm_80
            #-gencode=arch=compute_86,code=sm_86
            #-gencode=arch=compute_87,code=sm_87
            -gencode=arch=compute_89,code=sm_89
            #-gencode=arch=compute_89,code=compute_89
            -m64
            #-Xptxas=-v
            -rdc=true
            #--gpu-code=compute_75,compute_80,sm_75,sm_80,sm_89
            >)
    set_property(TARGET ${EXE} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${EXE} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS} ${Boost_LIBRARIES} PRIVATE stb_image rt_ptx owl::owl CUDA::cudart CUDA::cuda_driver)
else()
    set_source_files_properties(${ENTRY} PROPERTIES LANGUAGE CXX)
    add_executable(${EXE} ${ENTRY} ${SOURCE})
    target_link_libraries(${EXE} ${LIBPNG_LINK_FLAGS} ${Boost_LIBRARIES})
endif()
