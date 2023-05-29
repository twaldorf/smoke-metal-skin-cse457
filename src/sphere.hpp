#ifndef RTIOW1_SRC_SPHERE_HPP_
#define RTIOW1_SRC_SPHERE_HPP_

#include "hitable.hpp"
#include "vec3.hpp"

class sphere : public hitable {
 public:
	sphere(point3 cen, FLOAT r, shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {};

	bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;
	shared_ptr<material> mat_ptr;


	point3 center;
	FLOAT radius;

};

#endif //RTIOW1_SRC_SPHERE_HPP_
