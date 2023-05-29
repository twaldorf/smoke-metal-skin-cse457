#ifndef RTIOW1_SRC_GPU_GPU_UTIL_HPP_
#define RTIOW1_SRC_GPU_GPU_UTIL_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <curand_kernel.h>
#define FLOAT float

//forward declarations
class vec3;

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const FLOAT gpu_pi = 3.14159265358979323846;

__device__ FLOAT gpu_degrees_to_radians(FLOAT degrees);

__device__ FLOAT gpu_random_float(curandState *rand_state);

__device__ FLOAT gpu_random_float(FLOAT min, FLOAT max, curandState *rand_state);

//class gpu_camera;
//class gpu_hitable_list;
//
//struct gpu_renderInfo
//{
//	gpu_camera cam;
//	const gpu_hitable_list& world;
//	int blockX;
//	int blockY;
//	int blockSize;
//	int image_height;
//	int image_width;
//	int samples;
//	int max_depth;
//};

#endif //RTIOW1_SRC_GPU_GPU_UTIL_HPP_
