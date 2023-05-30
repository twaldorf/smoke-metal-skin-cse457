#ifndef RTIOW1_SRC_GPU_GPU_MATERIAL_CUH_
#define RTIOW1_SRC_GPU_GPU_MATERIAL_CUH_

#include "gpu_util.cuh"
#include "gpu_ray.cuh"
#include "gpu_hitable.cuh"
#include "gpu_vec3.cuh"

struct hit_record;
struct gpu_hit_record;

class gpu_ray;

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
	__device__ explicit gpu_dielectric(FLOAT index_of_refraction) : ir(index_of_refraction) {}
	__device__ bool scatter(const gpu_ray& r_in, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState *rand_state) const override;

 public:
	FLOAT ir;

 private:
	__device__ static FLOAT reflectance(FLOAT cosine, FLOAT ref_idx);
};

class gpu_isotropic : public gpu_material {
 public:
	__device__ explicit gpu_isotropic(gpu_colour* c) : albedo(c) {}

	__device__ bool scatter(const gpu_ray& r_in, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState *rand_state) const override;

 public:
	gpu_colour* albedo;
};

#endif //RTIOW1_SRC_GPU_GPU_MATERIAL_CUH_
