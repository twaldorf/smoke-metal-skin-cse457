#include "gpu_vec3.hpp"
#include <curand_kernel.h>

//various vector utility function
__device__ gpu_vec3 gpu_vec3::random(curandState *rand_state)
{
	return { curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state)};
}

__device__ gpu_vec3 gpu_vec3::random(FLOAT min, FLOAT max, curandState *rand_state)
{
	return { gpu_random_float(min, max, rand_state), gpu_random_float(min, max, rand_state), gpu_random_float(min, max, rand_state)};
}

__device__ FLOAT gpu_vec3::length_squared() const {
	return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
}

__device__ FLOAT gpu_vec3::length() const
{
	return sqrt(this->length_squared());
}

__device__ gpu_vec3 operator+(const gpu_vec3 &u, const gpu_vec3 &v)
{
	return {u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}

__device__ gpu_vec3 operator-(const gpu_vec3 &u, const gpu_vec3 &v)
{
	return {u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
}

__device__ gpu_vec3 operator*(const gpu_vec3 &u, const gpu_vec3 &v)
{
	return {u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
}

__device__ gpu_vec3 operator*(FLOAT t, const gpu_vec3 &v)
{
	return {t*v.e[0], t*v.e[1], t*v.e[2]};
}

__device__ gpu_vec3 operator*(const gpu_vec3 &v, FLOAT t)
{
	return t * v;
}

__device__ gpu_vec3 operator/(gpu_vec3 v, FLOAT t)
{
	return (1/t) * v;
}

__device__ FLOAT gpu_dot(const gpu_vec3 &u, const gpu_vec3 &v)
{
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

__device__ gpu_vec3 gpu_cross(const gpu_vec3 &u, const gpu_vec3 &v)
{
	return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
			u.e[2] * v.e[0] - u.e[0] * v.e[2],
			u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

__device__ gpu_vec3 gpu_unit_vector(gpu_vec3 v)
{
	return v / v.length();
}

//diffuse 1
__device__ gpu_vec3 gpu_random_in_unit_sphere(curandState *rand_state)
{
	gpu_vec3 p;
	do {
		p = 2.0f*gpu_vec3(curand_uniform(rand_state),curand_uniform(rand_state),curand_uniform(rand_state)) - gpu_vec3(1,1,1);
	} while (p.length_squared() >= 1.0f);
	return p;
}

//diffuse 2
__device__ gpu_vec3 gpu_random_unit_vector(curandState *rand_state)
{
	return gpu_unit_vector(gpu_random_in_unit_sphere(rand_state));
}

//diffuse 3
__device__ gpu_vec3 gpu_random_in_hemisphere(const gpu_vec3& normal, curandState *rand_state)
{
	gpu_vec3 in_unit_sphere = gpu_random_in_unit_sphere(rand_state);
	if(gpu_dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

__device__ bool gpu_vec3::near_zero() const
{
	// Return true if the vector is close to zero in all dimensions.
	const auto tolerance = 1e-8;
	return (fabs(e[0]) < tolerance) && (fabs(e[1]) < tolerance) && (fabs(e[2]) < tolerance);
}

//calculate direction of a reflected ray
__device__ gpu_vec3 gpu_reflect(const gpu_vec3& v, const gpu_vec3& n)
{
	//length of b is v*u then take that and multiply by n to "give it direction"
	return v - 2*gpu_dot(v,n)*n;
}

//calculate a refracted ray's direction
__device__ gpu_vec3 gpu_refract(const gpu_vec3& uv, const gpu_vec3& n, FLOAT eta_i_over_eta_t)
{
	FLOAT cos_theta = fmin(gpu_dot(-uv, n), 1.0f);
	gpu_vec3 r_out_perp = eta_i_over_eta_t * (uv + cos_theta*n);
	gpu_vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;

	return r_out_perp + r_out_parallel;

}

//randomly generate a point in a 2d disk with a radius of 1 (a unit circle)
__device__ gpu_vec3 gpu_random_in_unit_disk(curandState *rand_state)
{
	gpu_vec3 p;
	do {
		p = 2.0f*gpu_vec3(curand_uniform(rand_state),curand_uniform(rand_state),0) - gpu_vec3(1,1,0);
	} while (p.length_squared() >= 1.0f);
	return p;
}