#include "util.hpp"
#include "material.hpp"
#include "sphere.hpp"

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

double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}
