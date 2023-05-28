#include "material.hpp"

#ifdef USE_CUDA
__device__ bool gpu_lambertian::scatter(const gpu_ray& r, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState* rand_state) const
{
	gpu_vec3 target = rec.p + rec.normal + gpu_random_in_unit_sphere(rand_state);
	scattered = gpu_ray(rec.p, target-rec.p);
	attenuation = albedo;
	return true;
}

__device__ bool gpu_metal::scatter(const gpu_ray& r_in, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState* rand_state) const
{
	gpu_vec3 reflected = gpu_reflect(gpu_unit_vector(r_in.direction()), rec.normal);
	scattered = gpu_ray(rec.p, reflected + fuzz*gpu_random_in_unit_sphere(rand_state));
	attenuation = albedo;
	return (gpu_dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool gpu_dielectric::scatter(const gpu_ray& r_in, const gpu_hit_record& rec, gpu_colour& attenuation, gpu_ray& scattered, curandState* rand_state) const
{
	attenuation = gpu_colour(1.0, 1.0, 1.0);

	// n1/n2, n1 =1 so 1/n2 or n2/1
	FLOAT refraction_ratio = rec.front_face ? (1.0f/ir) : ir;

	gpu_vec3 unit_direction = gpu_unit_vector(r_in.direction());
	FLOAT cos_theta = fmin(gpu_dot(-unit_direction, rec.normal), 1.0f);
	FLOAT sin_theta = sqrt(1.0f - cos_theta*cos_theta); //trig identity

	//check for total internal reflection
	bool cannot_refract = refraction_ratio * sin_theta > 1.0;
	gpu_vec3 direction;

	//total internal reflection
	if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(rand_state))
		direction = gpu_reflect(unit_direction, rec.normal);
		//normal refraction
	else
		direction = gpu_refract(unit_direction, rec.normal, refraction_ratio);

	scattered = gpu_ray(rec.p, direction);
	return true;
}

__device__ FLOAT gpu_dielectric::reflectance(float cosine, float ref_idx)
{
	FLOAT r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
	r0 *= r0;
	return r0 + (1.0f-r0)*pow((1.0f-cosine), 5.0f);
}

#endif
//implements lambertian materials
bool lambertian::scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const
{
	auto scatter_direction = rec.normal + random_unit_vector();

	if(scatter_direction.near_zero())
		scatter_direction = rec.normal;

	scattered = ray(rec.p, scatter_direction);
	attenuation = albedo;
	return true;
}

//implements metal materials
bool metal::scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const
{
	vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
	//ray is scattered/reflected based on how fuzzy the metal is, 0 fuzz is mirror-like
	scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
	attenuation = albedo;
	return (dot(scattered.direction(), rec.normal) > 0);
}

//implements dielectric materials
bool dielectric::scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const
{
	attenuation = colour(1.0, 1.0, 1.0);

	// n1/n2, n1 =1 so 1/n2 or n2/1
	FLOAT refraction_ratio = rec.front_face ? (1.0/ir) : ir;

	vec3 unit_direction = unit_vector(r_in.direction());
	FLOAT cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
	FLOAT sin_theta = sqrt(1.0 - cos_theta*cos_theta); //trig identity

	//check for total internal reflection
	bool cannot_refract = refraction_ratio * sin_theta > 1.0;
	vec3 direction;

	//total internal reflection
	if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float())
		direction = reflect(unit_direction, rec.normal);
	//normal refraction
	else
		direction = refract(unit_direction, rec.normal, refraction_ratio);

	scattered = ray(rec.p, direction);
	return true;
}

FLOAT dielectric::reflectance(FLOAT cosine, FLOAT ref_idx)
{
	//Schlick's approximation
	// https://link.springer.com/chapter/10.1007/978-1-4842-7185-8_9 for full equation
	auto r0 = (1-ref_idx) / (1+ref_idx);
	r0 *= r0;
	return r0 + (1-r0)*pow((1-cosine), 5);
}

