#ifndef RTIOW1_SRC_UTIL_HPP_
#define RTIOW1_SRC_UTIL_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <curand_kernel.h>

#ifdef USE_CUDA
#define FLOAT float
#else
#define FLOAT double
#endif

//forward declarations
class vec3;

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const FLOAT pi = 3.1415926535897932385;

#ifdef USE_CUDA
__device__ FLOAT degrees_to_radians(FLOAT degrees);

__device__ FLOAT random_float(curandState *rand_state);

__device__ FLOAT random_float(FLOAT min, FLOAT max, curandState *rand_state);

FLOAT clamp(FLOAT x, FLOAT min, FLOAT max);
#else
FLOAT degrees_to_radians(FLOAT degrees);
const FLOAT infinity = std::numeric_limits<FLOAT>::infinity();

FLOAT random_float();

FLOAT random_float(FLOAT min, FLOAT max);

FLOAT clamp(FLOAT x, FLOAT min, FLOAT max);
#endif
#endif
