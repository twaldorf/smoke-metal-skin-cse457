#ifndef RTIOW1_SRC_HITABLE_HPP_
#define RTIOW1_SRC_HITABLE_HPP_

#include "ray.hpp"
#include "material.hpp"

#ifdef USE_CUDA
#include <curand_kernel.h>
class gpu_material;
class material;
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
#endif

struct hit_record {
	//point and normal of the intersection
	point3 p;
	vec3 normal;
	//material info
	shared_ptr<material> mat_ptr;

	//magnitude of the ray at intersection
	FLOAT t;
	//flag to check if we are entering or exiting an object
	bool front_face;

	//check if entering or exiting and object and flips the normal if we are exiting an object
	inline void set_face_normal(const ray& r, const vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

//all geometric objects need to implement this function
class hitable {
 public:
	virtual bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const = 0;
};

#endif //RTIOW1_SRC_HITABLE_HPP_
