#ifndef RTIOW1_SRC_RAY_HPP_
#define RTIOW1_SRC_RAY_HPP_

#include "vec3.hpp"

//forward declaration to avoid circular dependencies
class hitable;
class gpu_hitable;

#ifdef USE_CUDA
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

__device__ gpu_colour gpu_ray_colour(const gpu_ray& r, gpu_hitable **world, curandState *rand_state, int max_depth);
#endif

class ray {
 public:
	ray() = default;
	ray(const vec3& origin, const vec3& direction) : orig(origin), dir(direction) {}
	point3 origin() const { return orig; }
	vec3 direction() const { return dir; }
	point3 at(float t) const { return orig + t*dir; }

 public:
	point3 orig;
	vec3 dir;

};

colour ray_colour(const ray& r, const hitable& world, int max_depth);

#endif //RTIOW1_SRC_RAY_HPP_
