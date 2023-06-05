#include <cfloat>
#include <cstdio>
#include "optix_render.cuh"
#include "optix_util.cuh"

//constant medium sphere
template<typename ConstantMediumGeomType>
__device__ void constantMediumSphereBoundsProg(const void* geomData, box3f& primBounds, const int primID)
{
	const ConstantMediumGeomType& self = *(const ConstantMediumGeomType*)geomData;
	const constantMediumSphere& medium = self.prims[primID].constantMediumSphere;
	// Update the bounding box
	primBounds = box3f().extend(medium.sphere.center - medium.sphere.radius).extend(medium.sphere.center + medium.sphere.radius);
}

template<typename ConstantMediumGeomType>
__device__ void constantMediumSphereIntersectProg()
{
	const int primID = optixGetPrimitiveIndex();
	const auto& self = owl::getProgramData<ConstantMediumGeomType>().prims[primID];
	PerRayData& prd = owl::getPRD<PerRayData>();

	const vec3f org = optixGetObjectRayOrigin();
	const vec3f dir = optixGetObjectRayDirection();
	float hit1_t = optixGetRayTmax();
	float hit2_t = optixGetRayTmax();
	const float tmax = optixGetRayTmax();
	const float tmin = optixGetRayTmin();

	const vec3f oc = org - self.constantMediumSphere.sphere.center;

	const float a = dot(dir,dir);
	const float b = dot(oc, dir);
	const float c = dot(oc, oc) - self.constantMediumSphere.sphere.radius * self.constantMediumSphere.sphere.radius;
	const float discriminant = b * b - a * c;

	//	if (!boundary->hit(r, FLT_MIN, FLT_MAX, rec1, rand_state))
	//		return;
	if (discriminant < 0.0f)
	{
		return;
	}

	float root;
	root = (-b + sqrtf(discriminant)) / a;
	if (root < tmax && root > tmin)
		hit1_t = root;
	root = (-b - sqrtf(discriminant)) / a;
	if (root < tmax && root > tmin && root < hit1_t)
		hit1_t = root;
	else if(hit1_t == optixGetRayTmax())
		return;


//	if (!boundary->hit(r, rec1.t+0.0001f, FLT_MAX, rec2, rand_state))
//		return false;
	float temp_tmin = hit1_t+0.0001f;
	root = (-b + sqrtf(discriminant)) / a;
	if (root < tmax && root > temp_tmin)
		hit2_t = root;
	else
	{
		root = (-b - sqrtf(discriminant)) / a;
		if (root < tmax && root > temp_tmin)
			hit2_t = root;
		else
			return;
	}

	if (hit1_t < tmin)
		hit1_t = tmin;
	if (hit2_t > tmax)
		hit2_t = tmax;

	if (hit1_t >= hit2_t)
		return;

	if (hit1_t < 0)
		hit1_t = 0;

	const auto distance_inside_boundary = (hit2_t - hit1_t) * length(dir);
	const auto hit_distance = self.constantMediumSphere.neg_inv_density * log(randomFloat(prd.random));

	if (hit_distance > distance_inside_boundary)
		return;

	hit1_t = hit1_t + hit_distance / length(dir);

	optixReportIntersection(hit1_t, 0);
}

template<typename ConstantMediumGeomType>
__device__ void closestHitConstantMediumSphere()
{
	const int primID = optixGetPrimitiveIndex();
	const auto& self = owl::getProgramData<ConstantMediumGeomType>().prims[primID];

	PerRayData& prd = owl::getPRD<PerRayData>();

	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = optixGetWorldRayDirection();
	const float hit_t = optixGetRayTmax();
	const vec3f hit_P = org + hit_t * dir;
	const vec3f N = (hit_P-self.constantMediumSphere.sphere.center);

	// Update the scatter event based on the interaction with the constant medium
	prd.out.scatterEvent = scatter(self.material, hit_P, N, prd) ? rayGotBounced : rayGotCancelled;
}

//SPHERES
template<typename SphereGeomType>
__device__ void sphereBoundsProg(const void *geomData, box3f &primBounds, const int primID)
{
	const SphereGeomType &self = *(const SphereGeomType*)geomData;
	const Sphere sphere = self.prims[primID].sphere;
	primBounds = box3f().extend(sphere.center - sphere.radius).extend(sphere.center + sphere.radius);
}

template<typename SpheresGeomType>
__device__ void sphereIntersectProg()
{
	const int primID = optixGetPrimitiveIndex();
	// printf("isec %i %lx\n",primID,&owl::getProgramData<SpheresGeomType>());
	const auto &self = owl::getProgramData<SpheresGeomType>().prims[primID];

	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = optixGetWorldRayDirection();
	float hit_t = optixGetRayTmax();
	const float tmin = optixGetRayTmin();

	const vec3f oc = org - self.sphere.center;

	const float a = dot(dir,dir);
	const float b = dot(oc, dir);
	const float c = dot(oc, oc) - self.sphere.radius * self.sphere.radius;
	const float discriminant = b * b - a * c;

	if (discriminant < 0.f)
		return;

	//locally scope following values
	{
		float temp = (-b - sqrtf(discriminant)) / a;
		if (temp < hit_t && temp > tmin)
			hit_t = temp;
	}
	{
		float temp = (-b + sqrtf(discriminant)) / a;
		if (temp < hit_t && temp > tmin)
			hit_t = temp;
	}

	if (hit_t < optixGetRayTmax())
	{
		optixReportIntersection(hit_t, 0);
	}
}

template<typename SpheresGeomType>
__device__ void closestHitSpheres()
{
	const int primID = optixGetPrimitiveIndex();
	const auto &self = owl::getProgramData<SpheresGeomType>().prims[primID];

	PerRayData &prd = owl::getPRD<PerRayData>();

	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = optixGetWorldRayDirection();
	const float hit_t = optixGetRayTmax();
	const vec3f hit_P = org + hit_t * dir;
	const vec3f N = (hit_P-self.sphere.center);

	prd.out.scatterEvent = scatter(self.material, hit_P, N, prd) ? rayGotBounced : rayGotCancelled;
}

//BOXES
template<typename BoxesGeomType>
__device__ void closestHitBoxes()
{
	// printf("chbox\n");
	// return;
	const auto &self = owl::getProgramData<BoxesGeomType>();
	PerRayData &prd = owl::getPRD<PerRayData>();

	// ID of the triangle we've hit:
	const int primID = optixGetPrimitiveIndex();

	// there's 12 tris per box:
	const int materialID = primID / 12;

	const auto &material = self.perBoxMaterial[materialID];

	const vec3i index = self.index[primID];
	const vec3f &A = self.vertex[index.x];
	const vec3f &B = self.vertex[index.y];
	const vec3f &C = self.vertex[index.z];
	const vec3f N = normalize(cross(B-A,C-A));

	const vec3f org = optixGetWorldRayOrigin();
	const vec3f dir = optixGetWorldRayDirection();
	const float hit_t = optixGetRayTmax();
	const vec3f hit_P = org + hit_t * dir;

	prd.out.scatterEvent = scatter(material, hit_P,N, prd) ? rayGotBounced : rayGotCancelled;
}

__device__ vec3f missColor(const Ray &ray)
{
	const vec2i pixelID = owl::getLaunchIndex();

	const vec3f rayDir = normalize(ray.direction);
	const float t = 0.5f*(rayDir.y + 1.0f);
	const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
	return c;
}

__device__ vec3f tracePath(const RayGenData &self, owl::Ray &ray, PerRayData &prd)
{
	vec3f attenuation = 1.0f;
	printf("WTF\n");
	/* iterative version of recursion, up to depth 50 */
	for (int depth = 0; depth < 50; depth++)
	{
		prd.out.scatterEvent = rayDidntHitAnything;
		owl::traceRay(	/*accel to trace against*/self.world,
						/*the ray to trace*/ ray,
						/*prd*/prd);

		/* ray got 'lost' to the environment - 'light' it with miss
   			shader */
		if (prd.out.scatterEvent == rayDidntHitAnything)
			return attenuation * missColor(ray);
		else if (prd.out.scatterEvent == rayGotCancelled)
			return vec3f(0.0f);

		else
		{
			// ray is still alive, and got properly bounced
			attenuation *= prd.out.attenuation;
			ray = owl::Ray(prd.out.scattered_origin, //origin
							prd.out.scattered_direction, // direction:
							1e-3f, //tmin
							1e10f); //tmax
		}
	}
	// recursion did not terminate - cancel it
	return vec3f(0.0f);
}