#ifndef RTIOW1_SRC_WORLD_GEN_HPP_
#define RTIOW1_SRC_WORLD_GEN_HPP_

#include "hitable_list.hpp"

#ifdef USE_CUDA
//in cuda.hpp
#else
hitable_list random_scene();
#endif

#endif //RTIOW1_SRC_WORLD_GEN_HPP_
