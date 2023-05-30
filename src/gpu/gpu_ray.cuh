#ifndef RTIOW1_SRC_GPU_GPU_RAY_CUH_
#define RTIOW1_SRC_GPU_GPU_RAY_CUH_

#include "gpu_vec3.cuh"
#include <curand_kernel.h>

class gpu_ray {
 public:
	__device__ gpu_ray() = default;
	__device__ gpu_ray(const gpu_vec3& origin, const gpu_vec3& direction) : gpu_orig(origin), gpu_dir(direction) {}
	__device__ gpu_point3 origin() const { return gpu_orig; }
	__device__ gpu_vec3 direction() const { return gpu_dir; }
	__device__ gpu_point3 at(float t) const { return gpu_orig + t*gpu_dir; }

 public:
	gpu_point3 gpu_orig;
	gpu_vec3 gpu_dir;

};

class gpu_hitable;
__device__ gpu_colour gpu_ray_colour(const gpu_ray& r, gpu_hitable **world, curandState *rand_state, int max_depth);

#endif //RTIOW1_SRC_GPU_GPU_RAY_CUH_
