#ifndef RTIOW1_SRC_GPU_GPU_OPTIX_UTIL_CUH_
#define RTIOW1_SRC_GPU_GPU_OPTIX_UTIL_CUH_

#include <owl/owl.h>
#include <owl/common/math/AffineSpace.h>
#include <owl/common/math/random.h>
#include <optix_device.h>
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


inline __device__ float schlick(float cosine, float ref_idx)
{
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
}

inline __device__ bool refract(const vec3f& v, const vec3f& n, float ni_over_nt, vec3f &refracted)
{
	vec3f uv = normalize(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
	if (discriminant > 0.f) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
		return true;
	}
	else
		return false;
}

inline __device__ vec3f reflect(const vec3f &v, const vec3f &n)
{
	return v - 2.0f*dot(v, n)*n;
}

inline __device__ float randomFloat(Random &random)
{
	return random();
}

inline __device__ vec3f randomPointOnUnitDisc(Random &random)
{
	vec3f p;
	do {
		p = 2.0f*vec3f(random(), random(), 0.f) - vec3f(1.f, 1.f, 0.f);
	} while (dot(p, p) >= 1.0f);
	return p;
}


inline __device__ vec3f randomPointInUnitSphere(Random &rnd)
{
	vec3f p;
	do {
		p = 2.0f*RANDVEC3F - vec3f(1, 1, 1);
	} while (dot(p,p) >= 1.0f);
	return p;
}

inline __device__ bool scatter(const Lambertian &lambertian, const vec3f &P, vec3f N, PerRayData &prd)
{
	vec3i ld = optixGetLaunchDimensions();
	vec3i li = optixGetLaunchIndex();

	vec3i center = vec3i(ld.x/2, ld.y/2, ld.z/2);
	if(li == center)
		printf("Coordinates: %d %d\n", li.x, li.y);

	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = optixGetWorldRayDirection();

	if (dot(N,dir)  > 0.f)
		N = -N;
	N = normalize(N);

	const vec3f target
		= P + (N + randomPointInUnitSphere(prd.random));


	// return scattering event
	prd.out.scattered_origin = P;
	prd.out.scattered_direction = (target-P);
	prd.out.attenuation = lambertian.albedo;
	return true;
}

inline __device__ bool scatter(const Dielectric &dielectric, const vec3f &P, vec3f N, PerRayData &prd)
{
	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = normalize((vec3f)optixGetWorldRayDirection());

	N = normalize(N);
	vec3f outward_normal;
	vec3f reflected = reflect(dir,N);
	float ni_over_nt;
	prd.out.attenuation = vec3f(1.0f, 1.0f, 1.0f);
	vec3f refracted;
	float reflect_prob;
	float cosine;

	if (dot(dir,N) > 0.f) {
		outward_normal = -N;
		ni_over_nt = dielectric.ref_idx;
		cosine = dot(dir, N);// / vec3f(dir).length();
		cosine = sqrtf(1.f - dielectric.ref_idx*dielectric.ref_idx*(1.0f-cosine*cosine));
	}
	else {
		outward_normal = N;
		ni_over_nt = 1.0f / dielectric.ref_idx;
		cosine = -dot(dir, N);// / vec3f(dir).length();
	}
	if (refract(dir, outward_normal, ni_over_nt, refracted))
		reflect_prob = schlick(cosine, dielectric.ref_idx);
	else
		reflect_prob = 1.f;

	prd.out.scattered_origin = P;
	if (prd.random() < reflect_prob)
		prd.out.scattered_direction = reflected;
	else
		prd.out.scattered_direction = refracted;

	return true;
}

inline __device__ bool scatter(const Metal &metal, const vec3f &P, vec3f N, PerRayData &prd)
{
	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = optixGetWorldRayDirection();

	if (dot(N,dir)  > 0.f)
		N = -N;
	N = normalize(N);

	vec3f reflected = reflect(normalize(dir),N);
	prd.out.scattered_origin = P;
	prd.out.scattered_direction = (reflected+metal.fuzz*randomPointInUnitSphere(prd.random));
	prd.out.attenuation = metal.albedo;
	return (dot(prd.out.scattered_direction, N) > 0.f);
}

inline __device__ bool scatter(const Isotropic &isotropic, const vec3f &P, vec3f N, PerRayData &prd)
{
	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = optixGetWorldRayDirection();

	if (dot(N,dir)  > 0.f)
		N = -N;
	N = normalize(N);

	// return scattering event
	prd.out.scattered_origin = P;
	prd.out.scattered_direction = randomPointInUnitSphere(prd.random);
	prd.out.attenuation = isotropic.albedo;
	return true;
}

#endif //RTIOW1_SRC_GPU_GPU_OPTIX_UTIL_CUH_
