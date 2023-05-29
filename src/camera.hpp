#ifndef RTIOW1_SRC_CAMERA_HPP_
#define RTIOW1_SRC_CAMERA_HPP_

#include "vec3.hpp"
#include "ray.hpp"
#include "util.hpp"

class camera {
 public:
	camera(point3 lookfrom,
		point3 lookat,
		vec3 vup,
		FLOAT vfov,
		FLOAT aspect_ratio,
		FLOAT aperture,
		FLOAT focus_dist
		);

	ray get_ray(FLOAT s, FLOAT t) const;

 private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	FLOAT lens_radius;
};

#endif //RTIOW1_SRC_CAMERA_HPP_