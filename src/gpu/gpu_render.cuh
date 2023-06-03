#ifndef RTIOW1_SRC_CUDA_HPP_
#define RTIOW1_SRC_CUDA_HPP_

#include <curand_kernel.h>
#include "gpu_camera.cuh"
#include "gpu_hitable.cuh"
#include "gpu_vec3.cuh"
#include "../util.hpp"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
__global__ void rand_init(curandState *rand_state);
__global__ void gpu_render_init(int max_x, int max_y, curandState *rand_state);
__global__ void gpu_render(gpu_vec3 *fb, int image_width, int image_height, int samples_per_pixel, gpu_camera **cam, gpu_hitable **world, curandState *rand_state, int max_depth);
__global__ void create_world(gpu_hitable **obj_list, gpu_hitable **world, gpu_camera **camera, int nx, int ny, curandState *rand_state);
__global__ void free_world(gpu_hitable **obj_list, gpu_hitable **world, gpu_camera **camera);
__host__ void start_gpu_render(gpu_colour *fb, screenInfo screen);

#endif //RTIOW1_SRC_CUDA_HPP_
