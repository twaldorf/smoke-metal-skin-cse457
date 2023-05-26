#ifndef RTIOW1_SRC_CAMERA_HPP_
#define RTIOW1_SRC_CAMERA_HPP_

//#include <curand_kernel.h>
#include "ray.hpp"
#include "utl.hpp"

class camera {
 public:
	camera(point3 lookfrom,
		point3 lookat,
		vec3 vup,
		double vfov,
		double aspect_ratio,
		double aperture,
		double focus_dist
		);

	ray get_ray(double s, double t) const;

 private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	double lens_radius;
};

#endif //RTIOW1_SRC_CAMERA_HPP_