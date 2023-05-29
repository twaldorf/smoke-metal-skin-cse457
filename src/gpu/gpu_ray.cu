#include "gpu_ray.hpp"

#include <cfloat>
#include "gpu_ray.hpp"
#include "gpu_vec3.hpp"
#include "gpu_hitable.hpp"
#include "gpu_material.hpp"

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