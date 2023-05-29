#ifndef RTIOW1_SRC_GPU_GPU_VEC3_HPP_
#define RTIOW1_SRC_GPU_GPU_VEC3_HPP_

#include <cmath>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "gpu_util.hpp"

using std::sqrt;

class gpu_vec3
{
 public:
	__host__ __device__ gpu_vec3() : e{ 0, 0, 0 } {}
	__host__ __device__ gpu_vec3(FLOAT e0, FLOAT e1, FLOAT e2) : e{ e0, e1, e2 } {}

	__host__ __device__ FLOAT x() const
	{
		return e[0];
	}
	__host__ __device__ FLOAT y() const
	{
		return e[1];
	}
	__host__ __device__ FLOAT z() const
	{
		return e[2];
	}

	__device__ gpu_vec3 operator-() const
	{
		return { -e[0], -e[1], -e[2] };
	}
	__device__ FLOAT operator[](int i) const
	{
		return e[i];
	}
	__device__ FLOAT& operator[](int i)
	{
		return e[i];
	}

	__device__ gpu_vec3& operator+=(const gpu_vec3& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	__device__ gpu_vec3& operator*=(const FLOAT t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	__device__ gpu_vec3& operator/=(const FLOAT t)
	{
		return *this *= 1 / t;
	}

	__device__ bool near_zero() const;

	__device__ static gpu_vec3 random(curandState* rand_state);

	__device__ static gpu_vec3 random(FLOAT min, FLOAT max, curandState* rand_state);

	__device__ FLOAT length_squared() const;

	__device__ FLOAT length() const;

	FLOAT e[3];
};

__device__ gpu_vec3 operator+(const gpu_vec3 &u, const gpu_vec3 &v);
__device__ gpu_vec3 operator-(const gpu_vec3 &u, const gpu_vec3 &v);
__device__ gpu_vec3 operator*(const gpu_vec3 &u, const gpu_vec3 &v);
__device__ gpu_vec3 operator*(FLOAT t, const gpu_vec3 &v);
__device__ gpu_vec3 operator*(const gpu_vec3 &v, FLOAT t);
__device__ gpu_vec3 operator/(gpu_vec3 v, FLOAT t);

__device__ FLOAT gpu_dot(const gpu_vec3 &u, const gpu_vec3 &v);
__device__ gpu_vec3 gpu_cross(const gpu_vec3 &u, const gpu_vec3 &v);
__device__ gpu_vec3 gpu_unit_vector(gpu_vec3 v);

__device__ gpu_vec3 gpu_random_in_unit_sphere(curandState *rand_state);
__device__ gpu_vec3 gpu_random_unit_vector(curandState *rand_state);
__device__ gpu_vec3 gpu_random_in_hemisphere(const gpu_vec3& normal, curandState *rand_state);
__device__ gpu_vec3 gpu_reflect(const gpu_vec3& v, const gpu_vec3& n);
__device__ gpu_vec3 gpu_refract(const gpu_vec3& uv, const gpu_vec3& n, FLOAT etai_over_etat);
__device__ gpu_vec3 gpu_random_in_unit_disk(curandState *rand_state);

using gpu_point3 = gpu_vec3;   // 3D point
using gpu_colour = gpu_vec3;


#endif //RTIOW1_SRC_GPU_GPU_VEC3_HPP_
