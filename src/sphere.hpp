#ifndef RTIOW1_SRC_SPHERE_HPP_
#define RTIOW1_SRC_SPHERE_HPP_

#include "hitable.hpp"
#include "vec3.hpp"

class sphere : public hitable {
 public:
	#ifdef USE_CUDA
	__device__ sphere() {}
	__device__ sphere(point3 cen, FLOAT r, material* m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;
	material *mat_ptr;
	#else
	sphere(point3 cen, FLOAT r, shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {};

	bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;
	shared_ptr<material> mat_ptr;
	#endif

	point3 center;
	FLOAT radius;

};

#endif //RTIOW1_SRC_SPHERE_HPP_
