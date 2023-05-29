#ifndef RTIOW1_SRC_HITABLE_HPP_
#define RTIOW1_SRC_HITABLE_HPP_

#include "vec3.hpp"
#include "ray.hpp"
#include "material.hpp"

class material;

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
