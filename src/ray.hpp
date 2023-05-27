#ifndef RTIOW1_SRC_RAY_HPP_
#define RTIOW1_SRC_RAY_HPP_

#include <curand_kernel.h>
#include "vec3.hpp"

//forward declaration to avoid circular dependencies
class hitable;

class ray {
 public:
	#ifdef USE_CUDA
	__device__ ray() = default;
	__device__ ray(const vec3& origin, const vec3& direction) : orig(origin), dir(direction) {}
	__device__ point3 origin() const { return orig; }
	__device__ vec3 direction() const { return dir; }
	__device__ point3 at(float t) const { return orig + t*dir; }
	#else
	ray() = default;
	ray(const point3& origin, const vec3& direction)
	: orig(origin), dir(direction) {}

	point3 origin() const { return orig; }
	vec3 direction() const { return dir; }

	point3 at(double t) const {
		return orig + t*dir;
	}
	#endif

 public:
	point3 orig;
	vec3 dir;
};
#ifdef USE_CUDA
__device__ colour ray_colour(const ray& r, hitable **world, curandState *rand_state, int max_depth);
#else
colour ray_colour(const ray& r, const hitable& world, int max_depth);
#endif

#endif //RTIOW1_SRC_RAY_HPP_
