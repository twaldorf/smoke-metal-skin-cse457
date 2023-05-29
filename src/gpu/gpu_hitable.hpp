#ifndef RTIOW1_SRC_GPU_GPU_HITABLE_HPP_
#define RTIOW1_SRC_GPU_GPU_HITABLE_HPP_

#include "gpu_vec3.hpp"
#include "gpu_ray.hpp"
#include "gpu_material.hpp"

class material;

#include <curand_kernel.h>
class gpu_material;
class gpu_ray;

//data about each intersection
struct gpu_hit_record {
	//point and normal of the intersection
	gpu_point3 p;
	gpu_vec3 normal;
	//material info
	gpu_material *mat_ptr;

	//magnitude of the ray at intersection
	FLOAT t;
	//flag to check if we are entering or exiting an object
	bool front_face;

	__device__ inline void set_face_normal(const gpu_ray& r, const gpu_vec3& outward_normal)
	{
		front_face = gpu_dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class gpu_hitable {
 public:
	__device__ virtual bool hit(const gpu_ray& r, FLOAT t_min, FLOAT t_max, gpu_hit_record& rec) const = 0;
};

#endif //RTIOW1_SRC_GPU_GPU_HITABLE_HPP_
