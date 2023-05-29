#ifndef RTIOW1_SRC_UTIL_HPP_
#define RTIOW1_SRC_UTIL_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#define FLOAT float

//forward declarations
class vec3;

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const FLOAT pi = 3.14159265358979323846;

FLOAT degrees_to_radians(FLOAT degrees);
const FLOAT infinity = std::numeric_limits<FLOAT>::infinity();

FLOAT random_float();

FLOAT random_float(FLOAT min, FLOAT max);

FLOAT clamp(FLOAT x, FLOAT min, FLOAT max);

double random_double();

#endif
