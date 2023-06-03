#include "gpu_util.cuh"

__device__ FLOAT gpu_degrees_to_radians(FLOAT degrees)
{
	return degrees * gpu_pi / 180.0f;
}

#define RND (curand_uniform(&local_rand_state))
__device__ FLOAT gpu_random_float(curandState *rand_state)
{
	//higher values=more fuzzy
	curandState local_rand_state = *rand_state;
	return curand_uniform(&local_rand_state);
}

__device__ FLOAT gpu_random_float(FLOAT min, FLOAT max, curandState *rand_state)
{
	// Returns a random real in [min,max).
	curandState local_rand_state = *rand_state;
	return min + (max - min) * gpu_random_float(&local_rand_state);
}
