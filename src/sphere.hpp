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

private:
    static void get_sphere_uv(const point3& p, FLOAT& u, FLOAT& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }
};

#endif //RTIOW1_SRC_SPHERE_HPP_
