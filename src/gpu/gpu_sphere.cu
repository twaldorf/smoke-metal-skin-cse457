#include "gpu_sphere.cuh"
#include "gpu_ray.cuh"

__device__ bool gpu_sphere::hit(const gpu_ray& r, FLOAT t_min, FLOAT t_max, gpu_hit_record& rec, curandState* rand_state) const
{
	gpu_vec3f oc = r.origin() - center;
	FLOAT a = gpu_dot(r.direction(), r.direction());
	FLOAT half_b = gpu_dot(oc, r.direction());
	FLOAT c = oc.length_squared() - radius*radius;
	FLOAT discriminant = half_b*half_b - a*c;

	if (discriminant < 0)
		return false;

	FLOAT sqrtd = sqrt(discriminant);
	FLOAT root = (-half_b - sqrtd) / a;

	if (root < t_min || t_max < root) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || t_max < root)
			return false;
	}

	//return info about the intersection
	rec.t = root;
	rec.p = r.at(rec.t);
	gpu_vec3f outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;

	return true;
}