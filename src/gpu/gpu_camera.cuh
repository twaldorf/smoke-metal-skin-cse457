#ifndef RTIOW1_SRC_GPU_GPU_CAMERA_CUH_
#define RTIOW1_SRC_GPU_GPU_CAMERA_CUH_

#include "gpu_vec3.cuh"
#include "gpu_ray.cuh"
#include "gpu_util.cuh"

#include <curand_kernel.h>

class gpu_camera {
 public:
	__device__ gpu_camera(gpu_point3 lookfrom,
		gpu_point3 lookat,
		gpu_vec3 vup,
		FLOAT vfov,
		FLOAT aspect_ratio,
		FLOAT aperture,
		FLOAT focus_dist
	);

	__device__ gpu_ray get_ray(FLOAT s, FLOAT t, curandState *rand_state);

 private:
	gpu_point3 origin;
	gpu_point3 lower_left_corner;
	gpu_vec3 horizontal;
	gpu_vec3 vertical;
	gpu_vec3 u, v, w;
	FLOAT lens_radius;

};

#endif //RTIOW1_SRC_GPU_GPU_CAMERA_CUH_
