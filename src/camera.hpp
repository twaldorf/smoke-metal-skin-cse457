#ifndef RTIOW1_SRC_CAMERA_HPP_
#define RTIOW1_SRC_CAMERA_HPP_

#include "ray.hpp"
#include "util.hpp"


#ifdef USE_CUDA
#include <curand_kernel.h>
#endif


class camera {
 public:
	#ifdef USE_CUDA
	__device__ camera(point3 lookfrom,
		point3 lookat,
		vec3 vup,
		FLOAT vfov,
		FLOAT aspect_ratio,
		FLOAT aperture,
		FLOAT focus_dist
	);

	__device__ ray get_ray(FLOAT s, FLOAT t, curandState *rand_state);
	#else
	camera(point3 lookfrom,
		point3 lookat,
		vec3 vup,
		FLOAT vfov,
		FLOAT aspect_ratio,
		FLOAT aperture,
		FLOAT focus_dist
		);

	ray get_ray(double s, double t) const;
	#endif

 private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	FLOAT lens_radius;
};

#endif //RTIOW1_SRC_CAMERA_HPP_