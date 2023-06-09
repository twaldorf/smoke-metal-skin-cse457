#ifndef RTIOW1_SRC_MATERIAL_HPP_
#define RTIOW1_SRC_MATERIAL_HPP_

#include "util.hpp"
#include "ray.hpp"
#include "hitable.hpp"
#include "vec3.hpp"
#include "texture.hpp"

struct hit_record;

//all materials must implement a scatter function
class material {
 public:
    virtual colour emitted(FLOAT u, FLOAT v, const point3& p) const {
        return colour(0,0,0);
    }
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered
		) const = 0;
};

class lambertian : public material {
 public:
    lambertian(const colour& a) : albedo(make_shared<solid_colour>(a)) {}
    lambertian(shared_ptr<texture> a) : albedo(a) {}
	bool scatter(const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered) const override;
 public:
    shared_ptr<texture> albedo;
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

class isotropic : public material {
public:
    isotropic(colour c) : albedo(make_shared<solid_colour>(c)) {}
    isotropic(shared_ptr<texture> a) : albedo(a) {}

    virtual bool scatter(
            const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered
    ) const override;

    shared_ptr<texture> albedo;
};

class diffuse_light : public material  {
public:
    diffuse_light(shared_ptr<texture> a) : emit(a) {}
    diffuse_light(colour c) : emit(make_shared<solid_colour>(c)) {}

    virtual bool scatter(
            const ray& r_in, const hit_record& rec, colour& attenuation, ray& scattered
    ) const override {
        return false;
    }

    virtual colour emitted(FLOAT u, FLOAT v, const point3& p) const override;

public:
    shared_ptr<texture> emit;
};

#endif //RTIOW1_SRC_MATERIAL_HPP_
