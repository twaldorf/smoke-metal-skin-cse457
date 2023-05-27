#ifndef RTIOW1_SRC_HITABLE_HPP_
#define RTIOW1_SRC_HITABLE_HPP_

#include "ray.hpp"
#ifdef USE_CUDA
#include <curand_kernel.h>
#endif

class material;

//data about each intersection
struct hit_record {
	//point and normal of the intersection
	point3 p;
	vec3 normal;
	//material info
	#ifdef USE_CUDA
	material *mat_ptr;
	#else
	shared_ptr<material> mat_ptr;
	#endif
	//magnitude of the ray at intersection
	FLOAT t;
	//flag to check if we are entering or exiting an object
	bool front_face;

	#ifdef USE_CUDA
	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
	#else
	//check if entering or exiting and object and flips the normal if we are exiting an object
	inline void set_face_normal(const ray& r, const vec3& outward_normal)
	{
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
	#endif
};

//all geometric objects need to implement this function
class hitable {
 public:
	#ifdef USE_CUDA
	__device__ virtual bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const = 0;
	#else
	virtual bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const = 0;
	#endif
};

#endif //RTIOW1_SRC_HITABLE_HPP_
