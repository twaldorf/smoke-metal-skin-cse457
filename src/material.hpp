#ifndef RTIOW1_SRC_MATERIAL_HPP_
#define RTIOW1_SRC_MATERIAL_HPP_

#include "util.hpp"
#include "ray.hpp"
#include "hitable.hpp"

struct hit_record;

//all materials must implement a scatter function
class material {
 public:
	#ifdef USE_CUDA
	__device__ virtual bool scatter(
		const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, curandState *rand_state
	) const = 0;
	#else
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered
		) const = 0;
	#endif
};

class lambertian : public material {
 public:
	#ifdef USE_CUDA
	__device__ explicit lambertian(const colour& a) : albedo(a) {}
	__device__ bool scatter(const ray& r, const hit_record& rec, colour& attenuation, ray& scattered, curandState *rand_state) const override;
	#else
	explicit lambertian(const colour& a) : albedo(a) {}
	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;
	#endif
 public:
	colour albedo;
};

class metal : public material {
 public:
	#ifdef USE_CUDA
	__device__ metal(const colour& a, FLOAT f) : albedo(a), fuzz(f < 1 ? f : 1) {}
	__device__ bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, curandState *rand_state) const override;
	#else
	metal(const colour& a, FLOAT f) : albedo(a), fuzz(f < 1 ? f : 1) {}
	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;
	#endif
 public:
	colour albedo;
	FLOAT fuzz;
};

class dielectric : public material {
 public:
	#ifdef USE_CUDA
	__device__	explicit dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	__device__ bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered, curandState *rand_state) const override;
#else
	explicit dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;
	#endif
 public:
	FLOAT ir;

 private:
	#ifdef USE_CUDA
	__device__ static FLOAT reflectance(FLOAT cosine, FLOAT ref_idx);
	#else
	static FLOAT reflectance(FLOAT cosine, FLOAT ref_idx);
	#endif
};

#endif //RTIOW1_SRC_MATERIAL_HPP_
