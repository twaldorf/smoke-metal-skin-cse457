#ifndef RTIOW1_SRC_CUDA_HPP_
#define RTIOW1_SRC_CUDA_HPP_

#ifdef USE_CUDA
#include <curand_kernel.h>
#include "camera.hpp"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define RND (curand_uniform(&local_rand_state))

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
__global__ void rand_init(curandState *rand_state);
__global__ void render_init(int max_x, int max_y, curandState *rand_state);
__global__ void render(vec3 *fb, int image_width, int image_height, int samples_per_pixel, camera **cam, hitable **world, curandState *rand_state, int max_depth);
__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state);
__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera);

#endif

#endif //RTIOW1_SRC_CUDA_HPP_
