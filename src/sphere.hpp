#ifndef RTIOW1_SRC_SPHERE_HPP_
#define RTIOW1_SRC_SPHERE_HPP_

#include "hitable.hpp"
#include "vec3.hpp"

#ifdef USE_CUDA
class gpu_sphere : public gpu_hitable
{
 public:
	__device__ gpu_sphere() = default;
	__device__ gpu_sphere(gpu_point3 cen, FLOAT r, gpu_material *m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ bool hit(const gpu_ray& r, FLOAT t_min, FLOAT t_max, gpu_hit_record& rec) const override;
	gpu_material* mat_ptr{};

	gpu_point3 center;
	FLOAT radius{};
};
#endif

class sphere : public hitable {
 public:
	sphere(point3 cen, FLOAT r, shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {};

	bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;
	shared_ptr<material> mat_ptr;


	point3 center;
	FLOAT radius;

};

#endif //RTIOW1_SRC_SPHERE_HPP_
