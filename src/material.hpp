#ifndef RTIOW1_SRC_MATERIAL_HPP_
#define RTIOW1_SRC_MATERIAL_HPP_

#include "util.hpp"
#include "ray.hpp"
#include "hitable.hpp"

struct hit_record;
struct gpu_hit_record;

#ifdef USE_CUDA
class gpu_material {
 public:
	__device__ virtual bool scatter(
		const gpu_ray& r_in, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState *rand_state
	) const = 0;
};

class gpu_lambertian : public gpu_material {
 public:
	__device__ explicit gpu_lambertian(const gpu_colour& a) : albedo(a) {}
	__device__ bool scatter(const gpu_ray& r, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState *rand_state) const override;
 public:
	gpu_colour albedo;
};

class gpu_metal : public gpu_material {
 public:
	__device__ gpu_metal(const gpu_colour& a, FLOAT f) : albedo(a), fuzz(f < 1 ? f : 1) {}
	__device__ bool scatter(const gpu_ray& r_in, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState *rand_state) const override;
 public:
	gpu_colour albedo;
	FLOAT fuzz;
};

class gpu_dielectric : public gpu_material {
 public:
	__device__	explicit gpu_dielectric(FLOAT index_of_refraction) : ir(index_of_refraction) {}
	__device__ bool scatter(const gpu_ray& r_in, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState *rand_state) const override;

 public:
	FLOAT ir;

 private:
	__device__ static FLOAT reflectance(FLOAT cosine, FLOAT ref_idx);
};
#endif

//all materials must implement a scatter function
class material {
 public:
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered
		) const = 0;
};

class lambertian : public material {
 public:
	explicit lambertian(const colour& a) : albedo(a) {}
	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;
 public:
	colour albedo;
};

class metal : public material {
 public:
	metal(const colour& a, FLOAT f) : albedo(a), fuzz(f < 1 ? f : 1) {}
	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;
 public:
	colour albedo;
	FLOAT fuzz;
};

class dielectric : public material {
 public:
	explicit dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;
 public:
	FLOAT ir;

 private:
	static FLOAT reflectance(FLOAT cosine, FLOAT ref_idx);
};

#endif //RTIOW1_SRC_MATERIAL_HPP_
