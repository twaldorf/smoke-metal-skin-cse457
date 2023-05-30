#include "gpu_hitable_list.cuh"

__device__ bool gpu_hitable_list::hit(const gpu_ray& r, float t_min, float t_max, gpu_hit_record& rec, curandState* rand_state) const
{
	gpu_hit_record tempRecord;
	bool hitAnything = false;
	FLOAT closest_so_far = t_max;

	for(int i = 0; i < list_size; i++)
	{
		if(list[i]->hit(r, t_min, closest_so_far, tempRecord, rand_state))
		{
			hitAnything = true;
			closest_so_far = tempRecord.t;
			rec = tempRecord;
		}
	}
	return hitAnything;
}