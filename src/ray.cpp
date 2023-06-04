#include <cfloat>
#include "ray.hpp"
#include "vec3.hpp"
#include "hitable.hpp"
#include "material.hpp"

//get the colour of the ray
colour ray_colour(const ray& r, const hitable& world, int depth)
{
	//hit_detail records details of the intersection
	hit_record rec;

	//prevents recursion beyond max_depth
	if(depth <= 0)
		return {0,0,0};

	if(world.hit(r, 0.001, infinity, rec))
	{
		ray scattered;
		colour attenuation;
        colour emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

        if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return emitted;

        return emitted + attenuation * ray_colour(scattered, world, depth-1);
	}

	//sky
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5*(unit_direction.y() + 1.0);
	return (1.0-t)*colour(1.0, 1.0, 1.0) + t*colour(0.5, 0.7, 1.0);

    // void background, TODO: add feature flag to world object
    // return {0,0,0};
}
