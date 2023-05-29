#include "sphere.hpp"

//test if ray hits sphere
bool sphere::hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const
{
	// math to detect if a value on the ray can solve x^2+y^2+z^2=r^2
	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius*radius;

	auto discriminant = half_b*half_b - a*c;
	if (discriminant < 0)
		return false;
	auto sqrtd = sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range.
	auto root = (-half_b - sqrtd) / a;
	if (root < t_min || t_max < root) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || t_max < root)
		return false;
	}

	//return info about the intersection
	rec.t = root;
	rec.p = r.at(rec.t);
	vec3 outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;

	return true;
}
