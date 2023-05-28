#ifndef RTIOW1_SRC_HITABLE_LIST_HPP_
#define RTIOW1_SRC_HITABLE_LIST_HPP_

#include <memory>
#include <vector>
#include "hitable.hpp"

using std::shared_ptr;
using std::make_shared;

#ifdef USE_CUDA
class gpu_hitable_list : public gpu_hitable
{
 public:
	__device__ gpu_hitable_list() = default;
	__device__ gpu_hitable_list(gpu_hitable** l, int n)
	{
		list = l;
		list_size = n;
	}
	gpu_hitable** list;
	int list_size;

	__device__ bool hit(const gpu_ray& r, float t_min, float t_max, gpu_hit_record& rec) const;
};
#endif

class hitable_list : public hitable {
 public:
	hitable_list() = default;
	explicit hitable_list(shared_ptr<hitable> object) { add(object); }
	void clear() { objects.clear(); }
	void add(shared_ptr<hitable> object) { objects.push_back(object); }

	bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;


 public:
	std::vector<shared_ptr<hitable>> objects;
};

#endif