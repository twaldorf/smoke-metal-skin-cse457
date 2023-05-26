#ifndef RTIOW1_SRC_UTL_HPP_
#define RTIOW1_SRC_UTL_HPP_

#include <cmath>
#include <limits>
#include <memory>
#include <random>

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

double degrees_to_radians(double degrees);

double random_double();

double random_double(double min, double max);

double clamp(double x, double min, double max);

#endif
