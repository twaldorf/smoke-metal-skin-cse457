#ifndef RTIOW1_SRC_GPU_GPU_CONSTANT_MEDIUM_CUH_
#define RTIOW1_SRC_GPU_GPU_CONSTANT_MEDIUM_CUH_

#include "gpu_material.cuh"
#include "gpu_hitable.cuh"

class gpu_constant_medium : public gpu_hitable {
 public:
	__device__  gpu_constant_medium(gpu_hitable* b, FLOAT d, gpu_colour* c)
		: boundary(b),
		  neg_inv_density(-1/d),
		  phase_function(new gpu_isotropic(c)) {};

	__device__ bool hit(const gpu_ray& r, FLOAT t_min, FLOAT t_max, gpu_hit_record& rec, curandState* rand_state) const override;

//    virtual bool bounding_box(double time0, double time1, aabb&& output_box) const override;

	gpu_hitable* boundary;
	gpu_material* phase_function;
	FLOAT neg_inv_density;
};

#endif //RTIOW1_SRC_GPU_GPU_CONSTANT_MEDIUM_CUH_
