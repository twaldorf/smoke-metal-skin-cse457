#ifndef RTIOW1_SRC_MATERIAL_HPP_
#define RTIOW1_SRC_MATERIAL_HPP_

#include "utl.hpp"
#include "ray.hpp"
#include "hittable.hpp"

struct hit_record;

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
	metal(const colour& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;

 public:
	colour albedo;
	double fuzz;
};

class dielectric : public material {
 public:
	explicit dielectric(double index_of_refraction) : ir(index_of_refraction) {}

	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;

 public:
	double ir;

 private:
	static double reflectance(double cosine, double ref_idx);
};

#endif //RTIOW1_SRC_MATERIAL_HPP_
