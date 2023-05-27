#include "camera.hpp"

#ifdef USE_CUDA
__device__ camera::camera(point3 lookfrom,
	point3 lookat,
	vec3 vup,
	FLOAT vfov,
	FLOAT aspect_ratio,
	FLOAT aperture,
	FLOAT focus_dist)
{
	FLOAT theta = degrees_to_radians(vfov);
	FLOAT h = tan(theta/2.0f);
	FLOAT viewport_height = 2.0f*h;
	FLOAT viewport_width = aspect_ratio * viewport_height;

	w = unit_vector(lookfrom-lookat);
	u = unit_vector(cross(vup, w));
	v = cross(w, u);

	lower_left_corner = origin  - h*focus_dist*u - h*focus_dist*v - focus_dist*w;
	horizontal = viewport_width*focus_dist*u;
	vertical = viewport_height*focus_dist*v;

	lens_radius = aperture / 2.0f;
}

__device__ ray camera::get_ray(FLOAT s, FLOAT t, curandState *rand_state)
{
	vec3 rd = lens_radius*random_in_unit_disk(rand_state);
	vec3 offset = u * rd.x() + v * rd.y();
	return {origin+offset, lower_left_corner + s*horizontal + t*vertical - origin - offset};
}
#else
camera::camera(point3 lookfrom,
		point3 lookat,
		vec3 vup,
		FLOAT vfov,
		FLOAT aspect_ratio,
		FLOAT aperture,
		FLOAT focus_dist)
{
	auto theta = degrees_to_radians(vfov);
	auto h = tan(theta/2);
	auto viewport_height = 2.0*h;
	auto viewport_width = aspect_ratio * viewport_height;

	w = unit_vector(lookfrom-lookat);
	u = unit_vector(cross(vup, w));
	v = cross(w, u);

	origin = lookfrom;
	horizontal = focus_dist * viewport_width * u;
	vertical = focus_dist * viewport_height * v;
	lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

	lens_radius = aperture / 2;
}

ray camera::get_ray(FLOAT s, FLOAT t) const
{
	vec3 rd = lens_radius * random_in_unit_disk();
	vec3 offset = u * rd.x() + v * rd.y();

	return {origin+offset, lower_left_corner + s*horizontal + t*vertical - origin - offset};
}
#endif