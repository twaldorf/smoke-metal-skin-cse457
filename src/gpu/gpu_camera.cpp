#include "gpu_camera.hpp"

__device__ gpu_camera::gpu_camera(gpu_point3 lookfrom,
	gpu_point3 lookat,
	gpu_vec3 vup,
	FLOAT vfov,
	FLOAT aspect_ratio,
	FLOAT aperture,
	FLOAT focus_dist)
{
	FLOAT theta = gpu_degrees_to_radians(vfov);
	FLOAT half_height = tan(theta/2.0f);
	FLOAT half_width = aspect_ratio * half_height;
	origin = lookfrom;
	w = gpu_unit_vector(lookfrom-lookat);
	u = gpu_unit_vector(gpu_cross(vup, w));
	v = gpu_cross(w, u);


	lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
	horizontal = 2.0f*half_width*focus_dist*u;
	vertical = 2.0f*half_height*focus_dist*v;

	lens_radius = aperture / 2.0f;
}

__device__ gpu_ray gpu_camera::get_ray(FLOAT s, FLOAT t, curandState *rand_state)
{
	gpu_vec3 rd = lens_radius*gpu_random_in_unit_disk(rand_state);
	gpu_vec3 offset = u * rd.x() + v * rd.y();
	return {origin+offset, lower_left_corner + s*horizontal + t*vertical - origin - offset};
}