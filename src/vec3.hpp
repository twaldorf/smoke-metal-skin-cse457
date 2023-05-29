#ifndef RTIOW1__VEC3_HPP_
#define RTIOW1__VEC3_HPP_

#include <cmath>
#include <iostream>
#include "util.hpp"

using std::sqrt;

class vec3 {
	public:
	vec3() : e{0,0,0} {}
	vec3(FLOAT e0, FLOAT e1, FLOAT e2) : e{e0, e1, e2} {}

	FLOAT x() const { return e[0]; }
	FLOAT y() const { return e[1]; }
	FLOAT z() const { return e[2]; }

	vec3 operator-() const { return {-e[0], -e[1], -e[2]}; }
	FLOAT operator[](int i) const { return e[i]; }
	FLOAT& operator[](int i) { return e[i]; }

	vec3& operator+=(const vec3 &v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	vec3& operator*=(const FLOAT t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	vec3& operator/=(const FLOAT t) {
		return *this *= 1/t;
	}

	bool near_zero() const;

	static vec3 random();

	static vec3 random(FLOAT min, FLOAT max);

	FLOAT length_squared() const;

	FLOAT length() const;

	FLOAT e[3];
};


std::ostream& operator<<(std::ostream &out, const vec3 &v);
vec3 operator+(const vec3 &u, const vec3 &v);
vec3 operator-(const vec3 &u, const vec3 &v);
vec3 operator*(const vec3 &u, const vec3 &v);
vec3 operator*(FLOAT t, const vec3 &v);
vec3 operator*(const vec3 &v, FLOAT t);
vec3 operator/(vec3 v, FLOAT t);

FLOAT dot(const vec3 &u, const vec3 &v);
vec3 cross(const vec3 &u, const vec3 &v);
vec3 unit_vector(vec3 v);

vec3 random_in_unit_sphere();
vec3 random_unit_vector();
vec3 random_in_hemisphere(const vec3& normal);
vec3 reflect(const vec3& v, const vec3& n);
vec3 refract(const vec3& uv, const vec3& n, FLOAT etai_over_etat);
vec3 random_in_unit_disk();

using point3 = vec3;   // 3D point
using colour = vec3;    // RGB color

#endif
