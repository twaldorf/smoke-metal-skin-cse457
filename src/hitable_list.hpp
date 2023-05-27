#ifndef RTIOW1_SRC_HITABLE_LIST_HPP_
#define RTIOW1_SRC_HITABLE_LIST_HPP_

#include <memory>
#include <vector>
#include "hitable.hpp"

using std::shared_ptr;
using std::make_shared;

class hitable_list : public hitable {
 public:
	#ifdef USE_CUDA
	__device__ hitable_list() = default;
	__device__ hitable_list(hitable **l, int n) {list = l; list_size = n;}
	hitable **list;
	int list_size;

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	#else
	hitable_list() = default;
	explicit hitable_list(shared_ptr<hitable> object) { add(object); }
	void clear() { objects.clear(); }
	void add(shared_ptr<hitable> object) { objects.push_back(object); }

	bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;
	#endif

 public:
	#ifndef USE_CUDA
	std::vector<shared_ptr<hitable>> objects;
	#endif
};

#endif