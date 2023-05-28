#include <cfloat>
#include "ray.hpp"
#include "colour.hpp"
#include "vec3.hpp"
#include "hitable.hpp"
#include "material.hpp"

#ifdef USE_CUDA
__device__ gpu_colour gpu_ray_colour(const gpu_ray& r, gpu_hitable **world, curandState *rand_state, int depth)
{
	//we are looping instead of using recursion
	gpu_ray currentRay = r;
	gpu_colour currentAttenuation = gpu_colour(1.0,1.0,1.0);

	//prevents recursion beyond max_depth
	for(int i = 0; i < depth; i++)
	{
		gpu_hit_record rec;

		if((*world)->hit(currentRay, 0.001f, FLT_MAX, rec))
		{
			gpu_ray scattered;
			gpu_colour attenuation;

			if(rec.mat_ptr->scatter(currentRay, rec, attenuation, scattered, rand_state))
			{
				currentAttenuation = currentAttenuation * attenuation;
				currentRay = scattered;
			}
			else
			{
				return {0.0, 0.0, 0.0};
			}
		}
		else
		{
			gpu_vec3 unit_direction;
			unit_direction = gpu_unit_vector(currentRay.direction());
			FLOAT t = 0.5f*(unit_direction.y() + 1.0f);
			gpu_vec3 c = (1.0f-t)*gpu_colour(1.0f, 1.0f, 1.0f) + t*gpu_colour(0.5f, 0.7f, 1.0f);
			return currentAttenuation * c;
		}
	}
	//exceed recursion
	return {0.0,0.0,0.0};
}
#endif

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
		if(rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * ray_colour(scattered, world, depth-1);
		return {0,0,0};
	}

	//sky
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5*(unit_direction.y() + 1.0);
	return (1.0-t)*colour(1.0, 1.0, 1.0) + t*colour(0.5, 0.7, 1.0);
}
