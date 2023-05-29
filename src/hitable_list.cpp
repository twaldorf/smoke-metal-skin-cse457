#include "hitable_list.hpp"

bool hitable_list::hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const
{
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;

	//loop through every object and test if it hits
	for(const auto& object : objects)
	{
		//test intersection, ignore if intersection is father away then the closest
		//intersection we have already found
		if(object->hit(r, t_min, closest_so_far, temp_rec))
		{
			//save data about the hit
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}
