// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include "optix_host.hpp"

template<typename BoxArray, typename Material>
void addRandomBox(BoxArray& boxes, const vec3f& center, const float size, const Material& material)
{
	const int NUM_VERTICES = 8;
	static const vec3f unitBoxVertices[NUM_VERTICES] =
		{
			{ -1.f, -1.f, -1.f },
			{ +1.f, -1.f, -1.f },
			{ +1.f, +1.f, -1.f },
			{ -1.f, +1.f, -1.f },
			{ -1.f, +1.f, +1.f },
			{ +1.f, +1.f, +1.f },
			{ +1.f, -1.f, +1.f },
			{ -1.f, -1.f, +1.f },
		};

	const int NUM_INDICES = 12;
	static const vec3i unitBoxIndices[NUM_INDICES] =
		{
			{ 0, 2, 1 }, //face front
			{ 0, 3, 2 },
			{ 2, 3, 4 }, //face top
			{ 2, 4, 5 },
			{ 1, 2, 5 }, //face right
			{ 1, 5, 6 },
			{ 0, 7, 4 }, //face left
			{ 0, 4, 3 },
			{ 5, 4, 7 }, //face back
			{ 5, 7, 6 },
			{ 0, 6, 7 }, //face bottom
			{ 0, 1, 6 }
		};

	const vec3f U = normalize(randomPointInUnitSphere());
	owl::affine3f xfm = owl::frame(U);
	xfm = owl::affine3f(owl::linear3f::rotate(U, rnd())) * xfm;
	xfm = owl::affine3f(owl::linear3f::scale(.7f * size)) * xfm;
	xfm = owl::affine3f(owl::affine3f::translate(center)) * xfm;

	const int startIndex = (int)boxes.vertices.size();
	for(int i = 0; i < NUM_VERTICES; i++)
		boxes.vertices.push_back(owl::xfmPoint(xfm, unitBoxVertices[i]));
	for(int i = 0; i < NUM_INDICES; i++)
		boxes.indices.push_back(unitBoxIndices[i] + vec3i(startIndex));
	boxes.materials.push_back(material);
}

void createScene(OptixWorld* world)
{
	world->lambertianSpheres.push_back({ Sphere{ vec3f(0.0f, -1000.0f, -1.0f), 1000.0f },
										 Lambertian{ vec3f(0.5f, 0.5f, 0.5f) }});

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			float choose_mat = rnd();
			float choose_shape = rnd();
			vec3f center(a + rnd(), 0.2f, b + rnd());
			if (choose_mat < 0.7f)
			{
				if (choose_shape > .5f)
				{
					addRandomBox(world->lambertianBoxes, center, 0.2f,
						Lambertian{ rnd3f() * rnd3f() });
				}
				else
					world->lambertianSpheres.push_back({ Sphere{ center, 0.2f },
														 Lambertian{ rnd3f() * rnd3f() }});
			}
			else if (choose_mat < 0.8f)
			{
				if (choose_shape > .5f)
				{
					addRandomBox(world->metalBoxes, center, .2f,
						Metal{ 0.5f * (1.f + rnd3f()), 0.5f * rnd() });
				}
				else
					world->metalSpheres.push_back({ Sphere{ center, 0.2f },
													Metal{ 0.5f * (1.0f + rnd3f()), 0.5f * rnd() }});
			}
			else if (choose_mat < 0.9f)
			{
				if (choose_shape > .5f)
				{
					addRandomBox(world->isotropicBoxes, center, 0.2f,
						Isotropic{ rnd3f() * rnd3f() });
				}
				else
					world->isotropicSpheres.push_back({ Sphere{ center, 0.2f },
													Isotropic{rnd3f() * rnd3f() }});
			}
			else
			{
				if (choose_shape > .5f)
				{
					addRandomBox(world->dielectricBoxes, center, .2f,
						Dielectric{ 1.5f });
				}
				else
					world->dielectricSpheres.push_back({ Sphere{ center, 0.2f },
														 Dielectric{ 1.5f }});
			}
		}
	}
	world->dielectricSpheres.push_back({ Sphere{ vec3f(0.0f, 1.0f, 0.0f), 1.0f },
										 Dielectric{ 1.5f }});
//	world->lambertianSpheres.push_back({ Sphere{ vec3f(-4.0f, 1.0f, 0.0f), 1.0f },
//										 Lambertian{ vec3f(0.4f, 0.2f, 0.1f) }});
	world->isotropicSpheres.push_back({ Sphere{ vec3f(-4.0f, 1.0f, 0.0f), 1.0f },
										 Isotropic{ vec3f(0.9f, 0.1f, 0.1f) }});
	world->metalSpheres.push_back({ Sphere{ vec3f(4.0f, 1.0f, 0.0f), 1.0f },
									Metal{ vec3f(0.7f, 0.6f, 0.5f), 0.0f }});
}

void optixRender(screenInfo screen)
{
	// ##################################################################
	// pre-owl host-side set-up
	// ##################################################################
	clock_t start, stop;
	double timer_seconds;

	OptixWorld objectList;
	vec2i fbSize(screen.image_width, screen.image_height);
	vec3f lookFrom(13, 2, 3);
	vec3f lookAt(0, 0, 0);
	vec3f lookUp(0.0f, 1.0f, 0.0f);
	float fovy = 20.0f;
	camInfo optixCamera = { fbSize, lookFrom, lookAt, lookUp, fovy };

	createScene(&objectList);

	// ##################################################################
	// init owl
	// ##################################################################

	OWLContext context = owlContextCreate(nullptr, 1);
	OWLModule  module  = owlModuleCreate(context, optix_render_ptx);


	// ##################################################################
	// set up all the *GEOMETRY* graph we want to render
	// ##################################################################

	// -------------------------------------------------------
	// declare *sphere* geometry type(s)
	// -------------------------------------------------------

	// ----------- metal -----------
	OWLVarDecl metalSpheresGeomVars[] = {
		{ "prims", OWL_BUFPTR, OWL_OFFSETOF(MetalSpheresGeom, prims) },
		{ /* sentinel to mark end of list */ }
	};
	OWLGeomType metalSpheresGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_USER,
		sizeof(MetalSpheresGeom), metalSpheresGeomVars, -1);
	owlGeomTypeSetClosestHit(metalSpheresGeomType, 0, module, "MetalSpheres");
	owlGeomTypeSetIntersectProg(metalSpheresGeomType, 0, module, "MetalSpheres");
	owlGeomTypeSetBoundsProg(metalSpheresGeomType, module, "MetalSpheres");

	// ----------- dielectric -----------
	OWLVarDecl dielectricSpheresGeomVars[] = {
		{ "prims", OWL_BUFPTR, OWL_OFFSETOF(DielectricSpheresGeom, prims) },
		{ /* sentinel to mark end of list */ }
	};
	OWLGeomType dielectricSpheresGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_USER,
		sizeof(DielectricSpheresGeom), dielectricSpheresGeomVars, -1);
	owlGeomTypeSetClosestHit(dielectricSpheresGeomType, 0, module,"DielectricSpheres");
	owlGeomTypeSetIntersectProg(dielectricSpheresGeomType, 0, module,"DielectricSpheres");
	owlGeomTypeSetBoundsProg(dielectricSpheresGeomType, module,"DielectricSpheres");

	// ----------- lambertian -----------
	OWLVarDecl lambertianSpheresGeomVars[] = {
		{ "prims", OWL_BUFPTR, OWL_OFFSETOF(LambertianSpheresGeom, prims) },
		{ /* sentinel to mark end of list */ }
	};
	OWLGeomType lambertianSpheresGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_USER,
		sizeof(LambertianSpheresGeom), lambertianSpheresGeomVars, -1);
	owlGeomTypeSetClosestHit(lambertianSpheresGeomType, 0, module, "LambertianSpheres");
	owlGeomTypeSetIntersectProg(lambertianSpheresGeomType, 0, module, "LambertianSpheres");
	owlGeomTypeSetBoundsProg(lambertianSpheresGeomType, module, "LambertianSpheres");

	OWLVarDecl isotropicSpheresGeomVars[] = {
		{ "prims", OWL_BUFPTR, OWL_OFFSETOF(IsotropicSpheresGeom, prims) },
		{ /* sentinel to mark end of list */ }
	};

	// ----------- isotropic -----------
	OWLGeomType isotropicSpheresGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_USER,
		sizeof(IsotropicSpheresGeom), isotropicSpheresGeomVars, -1);
	owlGeomTypeSetClosestHit(isotropicSpheresGeomType, 0, module, "IsotropicSpheres");
	owlGeomTypeSetIntersectProg(isotropicSpheresGeomType, 0, module, "IsotropicSpheres");
	owlGeomTypeSetBoundsProg(isotropicSpheresGeomType, module, "IsotropicSpheres");


	// -------------------------------------------------------
	// declare *boxes* geometry type(s)
	// -------------------------------------------------------

	// ----------- metal -----------
	OWLVarDecl metalBoxesGeomVars[] = {
		{ "perBoxMaterial", OWL_BUFPTR, OWL_OFFSETOF(MetalBoxesGeom, perBoxMaterial) },
		{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(MetalBoxesGeom, vertex) },
		{ "index", OWL_BUFPTR, OWL_OFFSETOF(MetalBoxesGeom, index) },
		{ /* sentinel to mark end of list */ }
	};
	OWLGeomType metalBoxesGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES,
														sizeof(MetalBoxesGeom), metalBoxesGeomVars, -1);
	owlGeomTypeSetClosestHit(metalBoxesGeomType, 0, module, "MetalBoxes");

	// ----------- dielectric -----------
	OWLVarDecl dielectricBoxesGeomVars[] = {
		{ "perBoxMaterial", OWL_BUFPTR, OWL_OFFSETOF(DielectricBoxesGeom, perBoxMaterial) },
		{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(DielectricBoxesGeom, vertex) },
		{ "index", OWL_BUFPTR, OWL_OFFSETOF(DielectricBoxesGeom, index) },
		{ /* sentinel to mark end of list */ }
	};
	OWLGeomType dielectricBoxesGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES,
															sizeof(DielectricBoxesGeom), dielectricBoxesGeomVars, -1);
	owlGeomTypeSetClosestHit(dielectricBoxesGeomType, 0, module,"DielectricBoxes");

	// ----------- lambertian -----------
	OWLVarDecl lambertianBoxesGeomVars[] = {
		{ "perBoxMaterial", OWL_BUFPTR, OWL_OFFSETOF(LambertianBoxesGeom, perBoxMaterial) },
		{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(LambertianBoxesGeom, vertex) },
		{ "index", OWL_BUFPTR, OWL_OFFSETOF(LambertianBoxesGeom, index) },
		{ /* sentinel to mark end of list */ }
	};
	OWLGeomType lambertianBoxesGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES,
															sizeof(LambertianBoxesGeom), lambertianBoxesGeomVars, -1);
	owlGeomTypeSetClosestHit(lambertianBoxesGeomType, 0, module, "LambertianBoxes");

	// ----------- isotropic -----------
	OWLVarDecl isotropicBoxesGeomVars[] = {
		{ "perBoxMaterial", OWL_BUFPTR, OWL_OFFSETOF(IsotropicBoxesGeom, perBoxMaterial) },
		{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(IsotropicBoxesGeom, vertex) },
		{ "index", OWL_BUFPTR, OWL_OFFSETOF(IsotropicBoxesGeom, index) },
		{ /* sentinel to mark end of list */ }
	};
	OWLGeomType isotropicBoxesGeomType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES,
		sizeof(IsotropicBoxesGeom), isotropicBoxesGeomVars, -1);
	owlGeomTypeSetClosestHit(isotropicBoxesGeomType, 0, module, "IsotropicBoxes");

	// -------------------------------------------------------
	// make sure to do that *before* setting up the geometry, since the
	// user geometry group will need the compiled bounds programs upon
	// accelBuild()
	// -------------------------------------------------------
	owlBuildPrograms(context);

	// ##################################################################
	// set up all the *GEOMS* we want to run that code on
	// ##################################################################

	// ====================== SPHERES ======================

	// ----------- metal -----------
	OWLBuffer metalSpheresBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(objectList.metalSpheres[0]),
														objectList.metalSpheres.size(), objectList.metalSpheres.data());
	OWLGeom metalSpheresGeom = owlGeomCreate(context, metalSpheresGeomType);
	owlGeomSetPrimCount(metalSpheresGeom, objectList.metalSpheres.size());
	owlGeomSetBuffer(metalSpheresGeom, "prims", metalSpheresBuffer);

	// ----------- lambertian -----------
	OWLBuffer lambertianSpheresBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(objectList.lambertianSpheres[0]),
														objectList.lambertianSpheres.size(), objectList.lambertianSpheres.data());
	OWLGeom lambertianSpheresGeom = owlGeomCreate(context, lambertianSpheresGeomType);
	owlGeomSetPrimCount(lambertianSpheresGeom, objectList.lambertianSpheres.size());
	owlGeomSetBuffer(lambertianSpheresGeom, "prims", lambertianSpheresBuffer);

	// ----------- dielectric -----------
	OWLBuffer dielectricSpheresBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(objectList.dielectricSpheres[0]),
														objectList.dielectricSpheres.size(), objectList.dielectricSpheres.data());
	OWLGeom dielectricSpheresGeom = owlGeomCreate(context, dielectricSpheresGeomType);
	owlGeomSetPrimCount(dielectricSpheresGeom, objectList.dielectricSpheres.size());
	owlGeomSetBuffer(dielectricSpheresGeom, "prims", dielectricSpheresBuffer);

	// ----------- isotropic -----------
	OWLBuffer isotropicSpheresBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(objectList.isotropicSpheres[0]),
														objectList.isotropicSpheres.size(), objectList.isotropicSpheres.data());
	OWLGeom isotropicSpheresGeom = owlGeomCreate(context, isotropicSpheresGeomType);
	owlGeomSetPrimCount(isotropicSpheresGeom, objectList.isotropicSpheres.size());
	owlGeomSetBuffer(isotropicSpheresGeom, "prims", isotropicSpheresBuffer);


	// ====================== BOXES ======================

	// ----------- metal -----------
	OWLBuffer metalMaterialsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(objectList.metalBoxes.materials[0]),
															objectList.metalBoxes.materials.size(),
															objectList.metalBoxes.materials.data());
	OWLBuffer metalVerticesBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, objectList.metalBoxes.vertices.size(),
														objectList.metalBoxes.vertices.data());
	OWLBuffer metalIndicesBuffer = owlDeviceBufferCreate(context, OWL_INT3, objectList.metalBoxes.indices.size(),
														objectList.metalBoxes.indices.data());
	OWLGeom metalBoxesGeom = owlGeomCreate(context, metalBoxesGeomType);
	owlTrianglesSetVertices(metalBoxesGeom, metalVerticesBuffer, objectList.metalBoxes.vertices.size(),
							sizeof(objectList.metalBoxes.vertices[0]), 0);
	owlTrianglesSetIndices(metalBoxesGeom, metalIndicesBuffer, objectList.metalBoxes.indices.size(),
							sizeof(objectList.metalBoxes.indices[0]), 0);
	owlGeomSetBuffer(metalBoxesGeom, "perBoxMaterial", metalMaterialsBuffer);
	owlGeomSetBuffer(metalBoxesGeom, "vertex", metalVerticesBuffer);
	owlGeomSetBuffer(metalBoxesGeom, "index", metalIndicesBuffer);

	// ----------- lambertian -----------
	OWLBuffer lambertianMaterialsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(objectList.lambertianBoxes.materials[0]),
																objectList.lambertianBoxes.materials.size(),
																objectList.lambertianBoxes.materials.data());
	OWLBuffer lambertianVerticesBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3,
																objectList.lambertianBoxes.vertices.size(),
																objectList.lambertianBoxes.vertices.data());
	OWLBuffer lambertianIndicesBuffer = owlDeviceBufferCreate(context, OWL_INT3,
																objectList.lambertianBoxes.indices.size(),
																objectList.lambertianBoxes.indices.data());
	OWLGeom lambertianBoxesGeom = owlGeomCreate(context, lambertianBoxesGeomType);
	owlTrianglesSetVertices(lambertianBoxesGeom, lambertianVerticesBuffer, objectList.lambertianBoxes.vertices.size(),
							sizeof(objectList.lambertianBoxes.vertices[0]), 0);
	owlTrianglesSetIndices(lambertianBoxesGeom, lambertianIndicesBuffer, objectList.lambertianBoxes.indices.size(),
							sizeof(objectList.lambertianBoxes.indices[0]), 0);
	owlGeomSetBuffer(lambertianBoxesGeom, "perBoxMaterial", lambertianMaterialsBuffer);
	owlGeomSetBuffer(lambertianBoxesGeom, "vertex", lambertianVerticesBuffer);
	owlGeomSetBuffer(lambertianBoxesGeom, "index", lambertianIndicesBuffer);

	// ----------- dielectric -----------
	OWLBuffer dielectricMaterialsBuffer = owlDeviceBufferCreate(context,
																OWL_USER_TYPE(objectList.dielectricBoxes.materials[0]),
																objectList.dielectricBoxes.materials.size(),
																objectList.dielectricBoxes.materials.data());
	OWLBuffer dielectricVerticesBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3,
																objectList.dielectricBoxes.vertices.size(),
																objectList.dielectricBoxes.vertices.data());
	OWLBuffer dielectricIndicesBuffer = owlDeviceBufferCreate(context, OWL_INT3,
																objectList.dielectricBoxes.indices.size(),
																objectList.dielectricBoxes.indices.data());
	OWLGeom dielectricBoxesGeom = owlGeomCreate(context, dielectricBoxesGeomType);
	owlTrianglesSetVertices(dielectricBoxesGeom, dielectricVerticesBuffer, objectList.dielectricBoxes.vertices.size(),
							sizeof(objectList.dielectricBoxes.vertices[0]), 0);
	owlTrianglesSetIndices(dielectricBoxesGeom, dielectricIndicesBuffer, objectList.dielectricBoxes.indices.size(),
							sizeof(objectList.dielectricBoxes.indices[0]), 0);
	owlGeomSetBuffer(dielectricBoxesGeom, "perBoxMaterial", dielectricMaterialsBuffer);
	owlGeomSetBuffer(dielectricBoxesGeom, "vertex", dielectricVerticesBuffer);
	owlGeomSetBuffer(dielectricBoxesGeom, "index", dielectricIndicesBuffer);

	// ----------- isotropic -----------
	OWLBuffer isotropicMaterialsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(objectList.isotropicBoxes.materials[0]),
																objectList.isotropicBoxes.materials.size(),
																objectList.isotropicBoxes.materials.data());
	OWLBuffer isotropicVerticesBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3,
																objectList.isotropicBoxes.vertices.size(),
																objectList.isotropicBoxes.vertices.data());
	OWLBuffer isotropicIndicesBuffer = owlDeviceBufferCreate(context, OWL_INT3,
																objectList.isotropicBoxes.indices.size(),
																objectList.isotropicBoxes.indices.data());
	OWLGeom isotropicBoxesGeom = owlGeomCreate(context, isotropicBoxesGeomType);
	owlTrianglesSetVertices(isotropicBoxesGeom, isotropicVerticesBuffer, objectList.isotropicBoxes.vertices.size(),
								sizeof(objectList.isotropicBoxes.vertices[0]), 0);
	owlTrianglesSetIndices(isotropicBoxesGeom, isotropicIndicesBuffer, objectList.isotropicBoxes.indices.size(),
								sizeof(objectList.isotropicBoxes.indices[0]), 0);
	owlGeomSetBuffer(isotropicBoxesGeom, "perBoxMaterial", isotropicMaterialsBuffer);
	owlGeomSetBuffer(isotropicBoxesGeom, "vertex", isotropicVerticesBuffer);
	owlGeomSetBuffer(isotropicBoxesGeom, "index", isotropicIndicesBuffer);

	// ##################################################################
	// set up all *ACCELS* we need to trace into those groups
	// ##################################################################

	// ----------- one group for the spheres -----------
	/* (note these are user geoms, so have to be in another group than the triangle
	   meshes) */
	OWLGeom userGeoms[] = {lambertianSpheresGeom, metalSpheresGeom, dielectricSpheresGeom, isotropicSpheresGeom};
	OWLGroup userGeomGroup = owlUserGeomGroupCreate(context, 4, userGeoms);
	owlGroupBuildAccel(userGeomGroup);

	// ----------- one group for the boxes -----------
	/* (note these are made of triangles, so have to be in another group
	   than the sphere geoms) */
	OWLGeom triangleGeoms[] = {lambertianBoxesGeom, metalBoxesGeom, dielectricBoxesGeom, isotropicBoxesGeom};
	OWLGroup triangleGeomGroup = owlTrianglesGeomGroupCreate(context, 4, triangleGeoms);
	owlGroupBuildAccel(triangleGeomGroup);

	// ----------- one final group with one instance each -----------
	/* (this is just the simplest way of creating triangular with
	non-triangular geometry: create one separate instance each, and
	combine them in an instance group) */
	OWLGroup world = owlInstanceGroupCreate(context, 2);
	owlInstanceGroupSetChild(world, 0, userGeomGroup);
	owlInstanceGroupSetChild(world, 1, triangleGeomGroup);
	owlGroupBuildAccel(world);

	// ##################################################################
	// set miss and raygen programs
	// ##################################################################

	// -------------------------------------------------------
	// set up miss prog
	// -------------------------------------------------------
	OWLVarDecl missProgVars[] = {
		{ /* sentinel to mark end of list */ }
	};
	// ........... create object  ............................
	OWLMissProg missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1);
	owlMissProgSet(context, 0, missProg);

	// ........... set variables  ............................
	/* nothing to set */

	// -------------------------------------------------------
	// set up ray gen program
	// -------------------------------------------------------
	OWLVarDecl rayGenVars[] = {
		{ "fbPtr", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr) },
		{ "fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize) },
		{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world) },
		{ "samples", OWL_INT, OWL_OFFSETOF(RayGenData, samples)},
		{ "camera.org", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.origin) },
		{ "camera.llc", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.lower_left_corner) },
		{ "camera.horiz", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.horizontal) },
		{ "camera.vert", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.vertical) },
		{ /* sentinel to mark end of list */ }
	};

	// ........... create object  ............................
	OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen", sizeof(RayGenData), rayGenVars, -1);

	// ........... compute variable values  ..................
	const float vfov = optixCamera.fovy;
	const vec3f vup = optixCamera.lookUp;
	const float aspect = optixCamera.fbSize.x / float(fbSize.y);
	const float theta = vfov * ((float)M_PI) / 180.0f;
	const float half_height = tanf(theta / 2.0f);
	const float half_width = aspect * half_height;
	const float focusDist = 10.f;
	const vec3f origin = optixCamera.lookFrom;
	const vec3f w = normalize(optixCamera.lookFrom - optixCamera.lookAt);
	const vec3f u = normalize(cross(vup, w));
	const vec3f v = cross(w, u);
	const vec3f lower_left_corner = origin - half_width * focusDist * u - half_height * focusDist * v - focusDist * w;
	const vec3f horizontal = 2.0f * half_width * focusDist * u;
	const vec3f vertical = 2.0f * half_height * focusDist * v;

	OWLBuffer frameBuffer = owlHostPinnedBufferCreate(context, OWL_INT, optixCamera.fbSize.x * optixCamera.fbSize.y);

	// ----------- set variables  ----------------------------
	owlRayGenSetBuffer(rayGen, "fbPtr", frameBuffer);
	owlRayGenSet2i(rayGen, "fbSize", (const owl2i&)optixCamera.fbSize);
	owlRayGenSetGroup(rayGen, "world", world);
	owlRayGenSet1i(rayGen, "samples", screen.samples);
	owlRayGenSet3f(rayGen, "camera.org", (const owl3f&)origin);
	owlRayGenSet3f(rayGen, "camera.llc", (const owl3f&)lower_left_corner);
	owlRayGenSet3f(rayGen, "camera.horiz", (const owl3f&)horizontal);
	owlRayGenSet3f(rayGen, "camera.vert", (const owl3f&)vertical);

	// ##################################################################
	// build *SBT* required to trace the groups
	// ##################################################################

	// programs have been built before, but have to rebuild raygen and
	// miss progs
	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);

	// ##################################################################
	// now that everything is ready: launch it ....
	// ##################################################################
	start = clock();
	owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "took " << timer_seconds << " seconds with GPU using Optix.\n";
	// for host pinned mem it doesn't matter which device we query...
	const uint32_t* fb = (const uint32_t*)owlBufferGetPointer(frameBuffer, 0);
	stbi_write_png("../optix_test.png", optixCamera.fbSize.x, optixCamera.fbSize.y, 4,
					fb, optixCamera.fbSize.x * sizeof(uint32_t));

	// ##################################################################
	// and finally, clean up
	// ##################################################################

	owlContextDestroy(context);
}