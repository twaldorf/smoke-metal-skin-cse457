#ifndef RTIOW1_SRC_GPU_GPU_OPTIX_UTIL_CUH_
#define RTIOW1_SRC_GPU_GPU_OPTIX_UTIL_CUH_

#include <owl/owl.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>
#include "optix_materials.cuh"

using namespace owl;
typedef owl::common::LCG<4> Random;

typedef enum {
	/*! ray could get properly bounced, and is still alive */
	rayGotBounced,
	/*! ray could not get scattered, and should get cancelled */
	rayGotCancelled,
	/*! ray didn't hit anything, and went into the environment */
	rayDidntHitAnything
} ScatterEvent;

/*! "per ray data" (PRD) for our sample's rays. In the simple example, there is only
  one ray type, and it only ever returns one thing, which is a color (everything else
  is handled through the recursion). In addition to that return type, rays have to
  carry recursion state, which in this case are recursion depth and random number state */
struct PerRayData
{
	Random random;
	struct {
		ScatterEvent	scatterEvent;
		vec3f		scattered_origin;
		vec3f		scattered_direction;
		vec3f		attenuation;
	} out;
};

#define RANDVEC3F vec3f(rnd(),rnd(),rnd())

__device__ float schlick(float cosine, float ref_idx);
__device__ bool refract(const vec3f& v, const vec3f& n, float ni_over_nt, vec3f &refracted);
__device__ vec3f reflect(const vec3f &v, const vec3f &n);
__device__ vec3f randomPointOnUnitDisc(Random &random);
__device__ vec3f randomPointInUnitSphere(Random &rnd);
__device__ bool scatter(const Lambertian &lambertian, const vec3f &P, vec3f N, PerRayData &prd);
__device__ bool scatter(const Dielectric &dielectric, const vec3f &P, vec3f N, PerRayData &prd);
__device__ bool scatter(const Metal &metal, const vec3f &P, vec3f N, PerRayData &prd);
__device__ bool scatter(const Isotropic &isotropic, const vec3f &P, vec3f N, PerRayData &prd);

#endif //RTIOW1_SRC_GPU_GPU_OPTIX_UTIL_CUH_
