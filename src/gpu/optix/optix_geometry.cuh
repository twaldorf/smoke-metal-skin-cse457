#ifndef RTIOW1_SRC_GPU_OPTIX_OPTIX_GEOMETRY_CUH_
#define RTIOW1_SRC_GPU_OPTIX_OPTIX_GEOMETRY_CUH_

#include <optix_device.h>
#include "optix_materials.cuh"

// ==================================================================
/* the raw geometric shape of a sphere, without material - this is
   what goes into intersection and bounds programs */
// ==================================================================
struct Sphere {
	vec3f center;
	float radius;
};

// ==================================================================
/* the four actual primitive types created by fusing material data
   and geometry data */
// ==================================================================

struct MetalSphere {
	Sphere sphere;
	Metal  material;
};
struct DielectricSphere {
	Sphere sphere;
	Dielectric material;
};
struct LambertianSphere {
	Sphere sphere;
	Lambertian material;
};
struct IsotropicSphere {
	Sphere sphere;
	Isotropic material;
};

// ==================================================================
/* the three actual "Geoms" that each consist of multiple prims of
   same type (this is what optix6 would have called the "geometry
   instance" */
// ==================================================================

struct MetalSpheresGeom {
	/* for spheres geometry we store one full "sphere+material" record
	   per sphere */
	MetalSphere *prims;
};
struct DielectricSpheresGeom {
	/* for spheres geometry we store one full "sphere+material" record
	   per sphere */
	DielectricSphere *prims;
};
struct LambertianSpheresGeom {
	/* for spheres geometry we store one full "sphere+material" record
	   per sphere */
	LambertianSphere *prims;
};
struct IsotropicSpheresGeom {
	/* for spheres geometry we store one full "sphere+material" record
	   per sphere */
	IsotropicSphere *prims;
};

struct MetalBoxesGeom {
	/*! for our boxes geometry we use triangles for the geometry, so the
	  materials will actually be shared among every group of 12
	  triangles */
	Metal *perBoxMaterial;
	/* the vertex and index arrays for the triangle mesh */
	vec3f *vertex;
	vec3i *index;
};
struct DielectricBoxesGeom {
	/*! for our boxes geometry we use triangles for the geometry, so the
	  materials will actually be shared among every group of 12
	  triangles */
	Dielectric *perBoxMaterial;
	/* the vertex and index arrays for the triangle mesh */
	vec3f *vertex;
	vec3i *index;
};
struct LambertianBoxesGeom {
	/*! for our boxes geometry we use triangles for the geometry, so the
	  materials will actually be shared among every group of 12
	  triangles */
	Lambertian *perBoxMaterial;
	/* the vertex and index arrays for the triangle mesh */
	vec3f *vertex;
	vec3i *index;
};
struct IsotropicBoxesGeom {
	/*! for our boxes geometry we use triangles for the geometry, so the
	  materials will actually be shared among every group of 12
	  triangles */
	Isotropic *perBoxMaterial;
	/* the vertex and index arrays for the triangle mesh */
	vec3f *vertex;
	vec3i *index;
};

// ==================================================================
/* and finally, input for raygen and miss programs */
// ==================================================================
struct RayGenData
{
	uint32_t *fbPtr;
	vec2i  fbSize;
	uint32_t samples;
	OptixTraversableHandle world;

	struct {
		vec3f origin;
		vec3f lower_left_corner;
		vec3f horizontal;
		vec3f vertical;
	} camera;
};

struct MissProgData
{
	/* nothing in this example */
};

#endif //RTIOW1_SRC_GPU_OPTIX_OPTIX_GEOMETRY_CUH_
