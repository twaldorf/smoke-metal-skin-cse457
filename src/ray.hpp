#ifndef RTIOW1_SRC_RAY_HPP_
#define RTIOW1_SRC_RAY_HPP_

#include "vec3.hpp"

//forward declaration to avoid circular dependencies
class hitable;

class ray {
 public:
	ray() = default;
	ray(const vec3& origin, const vec3& direction) : orig(origin), dir(direction) {}
	point3 origin() const { return orig; }
	vec3 direction() const { return dir; }
	point3 at(float t) const { return orig + t*dir; }

 public:
	point3 orig;
	vec3 dir;

};

colour ray_colour(const ray& r, const hitable& world, int max_depth);

#endif //RTIOW1_SRC_RAY_HPP_
