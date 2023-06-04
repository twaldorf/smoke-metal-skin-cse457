#ifndef RTIOW1_SRC_GPU_OPTIX_OPTIX_RENDER_CUH_
#define RTIOW1_SRC_GPU_OPTIX_OPTIX_RENDER_CUH_

#include <optix_device.h>
#include "owl/owl.h"
#include "optix_geometry.cuh"
#include "optix_util.cuh"

#define SAMPLES 1000

using namespace owl;

template<typename SphereGeomType>
__device__ void boundsProg(const void *geomData, box3f &primBounds, const int primID);

OPTIX_BOUNDS_PROGRAM(MetalSpheres)(const void *geomData, box3f &primBounds, const int primID)
{
	boundsProg<MetalSpheresGeom>(geomData,primBounds,primID);
}

OPTIX_BOUNDS_PROGRAM(LambertianSpheres)(const void *geomData, box3f &primBounds, const int primID)
{
	boundsProg<LambertianSpheresGeom>(geomData,primBounds,primID);
}

OPTIX_BOUNDS_PROGRAM(DielectricSpheres)(const void *geomData, box3f &primBounds, const int primID)
{
	boundsProg<DielectricSpheresGeom>(geomData,primBounds,primID);
}

OPTIX_BOUNDS_PROGRAM(IsotropicSpheres)(const void *geomData, box3f &primBounds, const int primID)
{
	boundsProg<IsotropicSpheresGeom>(geomData,primBounds,primID);
}


// ==================================================================
// intersect programs - still all the same, since they don't use the
// material, either
// ==================================================================

template<typename SpheresGeomType>
__device__ void intersectProg();

OPTIX_INTERSECT_PROGRAM(MetalSpheres)()
{
	intersectProg<MetalSpheresGeom>();
}

OPTIX_INTERSECT_PROGRAM(LambertianSpheres)()
{
	intersectProg<LambertianSpheresGeom>();
}

OPTIX_INTERSECT_PROGRAM(DielectricSpheres)()
{
	intersectProg<DielectricSpheresGeom>();
}

OPTIX_INTERSECT_PROGRAM(IsotropicSpheres)()
{
	intersectProg<IsotropicSpheresGeom>();
}

// ==================================================================
// plumbing for closest hit, templated over geometry type so we can
// re-use the same code for different materials
// ==================================================================

// ----------- sphere+material -----------
template<typename SpheresGeomType>
__device__ void closestHitSpheres();

// ----------- "box+material" -----------
template<typename BoxesGeomType>
__device__ void closestHitBoxes();

// ==================================================================
// actual closest-hit program instantiations for geom+material types
// ==================================================================

// ---------------------- spheres ----------------------
OPTIX_CLOSEST_HIT_PROGRAM(MetalSpheres)()
{
	closestHitSpheres<MetalSpheresGeom>();
}

OPTIX_CLOSEST_HIT_PROGRAM(LambertianSpheres)()
{
	closestHitSpheres<LambertianSpheresGeom>();
}

OPTIX_CLOSEST_HIT_PROGRAM(DielectricSpheres)()
{
	closestHitSpheres<DielectricSpheresGeom>();
}

// ---------------------- boxes ----------------------
OPTIX_CLOSEST_HIT_PROGRAM(MetalBoxes)()
{
	closestHitBoxes<MetalBoxesGeom>();
}

OPTIX_CLOSEST_HIT_PROGRAM(LambertianBoxes)()
{
	closestHitBoxes<LambertianBoxesGeom>();
}

OPTIX_CLOSEST_HIT_PROGRAM(DielectricBoxes)()
{
	closestHitBoxes<DielectricBoxesGeom>();
}

// ==================================================================
// miss and raygen
// ==================================================================

__device__ vec3f missColor(const Ray &ray);

OPTIX_MISS_PROGRAM(miss)()
{
	/* nothing to do */
}

__device__ vec3f tracePath(const RayGenData &self, owl::Ray &ray, PerRayData &prd);


OPTIX_RAYGEN_PROGRAM(rayGen)()
{
	const RayGenData &self = owl::getProgramData<RayGenData>();
	const vec2i pixelID = owl::getLaunchIndex();

	const int pixelIdx = pixelID.x + self.fbSize.x * (self.fbSize.y - 1 - pixelID.y);

	PerRayData prd;
	prd.random.init(pixelID.x, pixelID.y);

	vec3f color = 0.f;
	for (int sampleID=0; sampleID < SAMPLES; sampleID++)
	{
		owl::Ray ray;

		const vec2f pixelSample(prd.random(), prd.random());
		const vec2f screen = (vec2f(pixelID) + pixelSample) / vec2f(self.fbSize);
		const vec3f origin = self.camera.origin; // + lens_offset
		const vec3f direction = self.camera.lower_left_corner
								+ screen.u * self.camera.horizontal
								+ screen.v * self.camera.vertical
								- self.camera.origin;

		ray.origin = origin;
		ray.direction = normalize(direction);

		color += tracePath(self, ray, prd);
	}

	self.fbPtr[pixelIdx] = owl::make_rgba(color * (1.f / SAMPLES));
}

#endif //RTIOW1_SRC_GPU_OPTIX_OPTIX_RENDER_CUH_
