#ifndef RTIOW1_SRC_UTIL_HPP_
#define RTIOW1_SRC_UTIL_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <random>

#ifdef USE_CUDA
#include <curand_kernel.h>
#define FLOAT float
#else
#define FLOAT double
#endif

//forward declarations
class vec3;

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const FLOAT pi = 3.14159265358979323846;

#ifdef USE_CUDA
__device__ FLOAT gpu_degrees_to_radians(FLOAT degrees);

__device__ FLOAT gpu_random_float(curandState *rand_state);

__device__ FLOAT gpu_random_float(FLOAT min, FLOAT max, curandState *rand_state);

#endif
FLOAT degrees_to_radians(FLOAT degrees);
const FLOAT infinity = std::numeric_limits<FLOAT>::infinity();

FLOAT random_float();

FLOAT random_float(FLOAT min, FLOAT max);

FLOAT clamp(FLOAT x, FLOAT min, FLOAT max);

double random_double();

#endif
