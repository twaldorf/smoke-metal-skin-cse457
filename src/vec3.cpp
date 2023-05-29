#include "vec3.hpp"

//various vector utility function
vec3 vec3::random()
{
	return { random_float(), random_float(), random_float()};
}

vec3 vec3::random(FLOAT min, FLOAT max)
{
	return { random_float(min, max), random_float(min, max), random_float(min, max)};
}

FLOAT vec3::length_squared() const {
	return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
}

FLOAT vec3::length() const
{
	return sqrt(this->length_squared());
}

std::ostream& operator<<(std::ostream &out, const vec3 &v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

vec3 operator+(const vec3 &u, const vec3 &v)
{
	return {u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}

vec3 operator-(const vec3 &u, const vec3 &v)
{
	return {u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
}

vec3 operator*(const vec3 &u, const vec3 &v)
{
	return {u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
}

vec3 operator*(FLOAT t, const vec3 &v)
{
	return {t*v.e[0], t*v.e[1], t*v.e[2]};
}

vec3 operator*(const vec3 &v, FLOAT t)
{
	return t * v;
}

vec3 operator/(vec3 v, FLOAT t)
{
	return (1/t) * v;
}

FLOAT dot(const vec3 &u, const vec3 &v)
{
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

vec3 cross(const vec3 &u, const vec3 &v)
{
	return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

vec3 unit_vector(vec3 v)
{
	return v / v.length();
}

//diffuse 1
vec3 random_in_unit_sphere()
{
	while(true)
	{
		auto p = vec3::random(-1,1);
		if(p.length_squared() >= 1)
			continue;
		return p;
	}
}

//diffuse 2
vec3 random_unit_vector()
{
	return unit_vector(random_in_unit_sphere());
}

//diffuse 3
vec3 random_in_hemisphere(const vec3& normal)
{
	vec3 in_unit_sphere = random_in_unit_sphere();
	if(dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

bool vec3::near_zero() const
{
	// Return true if the vector is close to zero in all dimensions.
	const auto tolerance = 1e-8;
	return (fabs(e[0]) < tolerance) && (fabs(e[1]) < tolerance) && (fabs(e[2]) < tolerance);
}

//calculate direction of a reflected ray
vec3 reflect(const vec3& v, const vec3& n)
{
	//length of b is v*u then take that and multiply by n to "give it direction"
	return v - 2*dot(v,n)*n;
}

//calculate a refracted ray's direction
vec3 refract(const vec3& uv, const vec3& n, FLOAT eta_i_over_eta_t)
{
	auto cos_theta = fmin(dot(-uv, n), 1.0);
	vec3 r_out_perp = eta_i_over_eta_t * (uv + cos_theta*n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;

	return r_out_perp + r_out_parallel;

}

//randomly generate a point in a 2d disk with a radius of 1 (a unit circle)
vec3 random_in_unit_disk()
{
	while(true)
	{
		auto p = vec3(random_float(-1, 1), random_float(-1, 1), 0);
		if(p.length_squared() >= 1)
			continue;
		return p;
	}
}
