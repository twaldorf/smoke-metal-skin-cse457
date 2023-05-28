#include "util.hpp"
#include "material.hpp"
#include "sphere.hpp"

#ifdef USE_CUDA
__device__ FLOAT gpu_degrees_to_radians(FLOAT degrees)
{
	return degrees * pi / 180.0f;
}

#define RND (curand_uniform(&local_rand_state))
__device__ FLOAT gpu_random_float(curandState *rand_state)
{
	//higher values=more fuzzy
	curandState local_rand_state = *rand_state;
	curand_uniform(&local_rand_state);
}

__device__ FLOAT gpu_random_float(FLOAT min, FLOAT max, curandState *rand_state)
{
	// Returns a random real in [min,max).
	curandState local_rand_state = *rand_state;
	return min + (max - min) * gpu_random_float(&local_rand_state);
}
#endif

FLOAT degrees_to_radians(FLOAT degrees)
{
	return degrees * pi / 180.0f;
}

FLOAT random_float()
{
	//higher values=more fuzzy
	static std::uniform_real_distribution<FLOAT> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

FLOAT random_float(FLOAT min, FLOAT max)
{
	// Returns a random real in [min,max).
	return min + (max - min) * random_float();
}

FLOAT clamp(FLOAT x, FLOAT min, FLOAT max)
{
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}
