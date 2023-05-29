#include "material.hpp"

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

