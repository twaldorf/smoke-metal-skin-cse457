#ifndef RTIOW1_SRC_GPU_OPTIX_OPTIX_MATERIALS_CUH_
#define RTIOW1_SRC_GPU_OPTIX_OPTIX_MATERIALS_CUH_

#include "owl/common/math/vec.h"
using namespace owl;

struct Lambertian {
	vec3f albedo;
};
struct Metal {
	vec3f albedo;
	float fuzz;
};
struct Dielectric {
	float ref_idx;
};
struct Isotropic {
	vec3f albedo;
};
#endif //RTIOW1_SRC_GPU_OPTIX_OPTIX_MATERIALS_CUH_
