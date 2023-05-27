#include "util.hpp"
#include "material.hpp"
#include "sphere.hpp"

#ifdef USE_CUDA
__device__ FLOAT degrees_to_radians(FLOAT degrees)
{
	return degrees * pi / 180.0f;
}

__device__ FLOAT random_float(curandState *rand_state)
{
	//higher values=more fuzzy
	curand_uniform(rand_state);
}

__device__ FLOAT random_double(FLOAT min, FLOAT max, curandState *rand_state)
{
	// Returns a random real in [min,max).
	return min + (max - min) * random_float(rand_state);
}

FLOAT clamp(FLOAT x, FLOAT min, FLOAT max)
{
	if (x < min)
		return min;
	if (x > max)
		return max;
	return x;
}
#else
FLOAT degrees_to_radians(FLOAT degrees)
{
	return degrees * pi / 180.0;
}

FLOAT random_float()
{
	//higher values=more fuzzy
	static std::uniform_real_distribution<FLOAT> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

FLOAT random_double(FLOAT min, FLOAT max)
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
#endif