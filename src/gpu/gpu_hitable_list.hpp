#ifndef RTIOW1_SRC_GPU_GPU_HITABLE_LIST_HPP_
#define RTIOW1_SRC_GPU_GPU_HITABLE_LIST_HPP_

#include <memory>
#include <vector>
#include "gpu_hitable.hpp"

using std::shared_ptr;
using std::make_shared;

class gpu_hitable_list : public gpu_hitable
{
 public:
	__device__ gpu_hitable_list() = default;
	__device__ gpu_hitable_list(gpu_hitable** l, int n)
	{
		list = l;
		list_size = n;
	}
	gpu_hitable** list{};
	int list_size{};

	__device__ bool hit(const gpu_ray& r, float t_min, float t_max, gpu_hit_record& rec) const;
};

#endif //RTIOW1_SRC_GPU_GPU_HITABLE_LIST_HPP_
