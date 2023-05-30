#include <cfloat>
#include "gpu_constant_medium.cuh"

__device__ bool gpu_constant_medium::hit(const gpu_ray& r, FLOAT t_min, FLOAT t_max, gpu_hit_record& rec, curandState* rand_state) const {
	gpu_hit_record rec1, rec2;

	if (!boundary->hit(r, FLT_MIN, FLT_MAX, rec1, rand_state))
		return false;

	if (!boundary->hit(r, rec1.t+0.0001f, FLT_MAX, rec2, rand_state))
		return false;

	if (rec1.t < t_min)
		rec1.t = t_min;
	if (rec2.t > t_max)
		rec2.t = t_max;

	if (rec1.t >= rec2.t)
		return false;

	if (rec1.t < 0)
		rec1.t = 0;

	const auto ray_length = r.direction().length();
	const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
	const auto hit_distance = neg_inv_density * log(gpu_random_float(rand_state));

	if (hit_distance > distance_inside_boundary)
		return false;

	rec.t = rec1.t + hit_distance / ray_length;
	rec.p = r.at(rec.t);

	rec.normal = gpu_vec3(1,0,0);  // arbitrary
	rec.front_face = true;     // also arbitrary
	rec.mat_ptr = phase_function;

	return true;
}