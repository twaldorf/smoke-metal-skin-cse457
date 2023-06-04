#ifndef RTIOW1_SRC_GPU_OPTIX_OPTIX_HOST_HPP_
#define RTIOW1_SRC_GPU_OPTIX_OPTIX_HOST_HPP_

// The Ray Tracing in One Weekend scene, but with cubes substituted for some
// spheres. This program shows how different geometric types in a single scene
// are handled.
// public owl API
#include "owl/owl.h"
// our device-side data structures
#include "optix_geometry.cuh"
#include "owl/common/math/AffineSpace.h"
#include "../../util.hpp"

#include <random>

//MUST BE NAMED SAME AS DEVICE CODE FILE
extern "C" char optix_render_ptx[];

struct diBoxes{
	std::vector<vec3f> vertices;
	std::vector<vec3i> indices;
	std::vector<Dielectric> materials;
} ;

struct meBoxes {
	std::vector<vec3f> vertices;
	std::vector<vec3i> indices;
	std::vector<Metal> materials;
};

struct laBoxes {
	std::vector<vec3f> vertices;
	std::vector<vec3i> indices;
	std::vector<Lambertian> materials;
};

struct isBoxes {
	std::vector<vec3f> vertices;
	std::vector<vec3i> indices;
	std::vector<Isotropic> materials;
};

struct OptixWorld {
	std::vector<DielectricSphere> dielectricSpheres;
	std::vector<LambertianSphere> lambertianSpheres;
	std::vector<MetalSphere>      metalSpheres;
	std::vector<IsotropicSphere>  isotropicSpheres;
	diBoxes dielectricBoxes;
	meBoxes metalBoxes;
	laBoxes lambertianBoxes;
	isBoxes isotropicBoxes;
};

struct camInfo
{
	vec2i fbSize;
	vec3f lookFrom;
	vec3f lookAt;
	vec3f lookUp;
	float fovy;
};

inline float rnd()
{
	static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
	static std::uniform_real_distribution<float> dis(0.f, 1.f);
	return dis(gen);
}

inline vec3f rnd3f() { return vec3f(rnd(),rnd(),rnd()); }

inline vec3f randomPointInUnitSphere()
{
	vec3f p;
	do {
		p = 2.f*vec3f(rnd(),rnd(),rnd()) - vec3f(1.f);
	} while (dot(p,p) >= 1.f);
	return p;
}

template<typename BoxArray, typename Material>
void addRandomBox(BoxArray &boxes, const vec3f &center, const float size, const Material &material);

void createScene(OptixWorld* world);

void optixRender(screenInfo screen);

#endif //RTIOW1_SRC_GPU_OPTIX_OPTIX_HOST_HPP_
