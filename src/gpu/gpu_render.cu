#include <cstdlib>
#include <iostream>
#include "gpu_render.cuh"
#include "gpu_sphere.cuh"
#include "gpu_material.cuh"
#include "gpu_hitable_list.cuh"
#include "../util.hpp"
#include "gpu_constant_medium.cuh"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
				  file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void rand_init(curandState *rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curand_init(2023, 0, 0, rand_state);
	}
}

__global__ void gpu_render_init(int max_x, int max_y, curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	// Original: Each thread gets same seed, a different sequence number, no offset
	// curand_init(2023, pixel_index, 0, &rand_state[pixel_index]);
	// Each thread gets different seed, same sequence for
	// performance improvement of about 2x!
	curand_init(2023+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void gpu_render(gpu_vec3 *fb, int image_width, int image_height, int samples_per_pixel, gpu_camera **cam, gpu_hitable **world, curandState *rand_state, int max_depth)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if((i >= image_width) || (j >= image_height))
		return;

	int pixel_index = j*image_width + i;
	curandState local_rand_state = rand_state[pixel_index];
	gpu_colour col(0,0,0);

	for(int s = 0; s < samples_per_pixel; s++)
	{
		FLOAT u = FLOAT(i + curand_uniform(&local_rand_state)) / FLOAT(image_width);
		FLOAT v = FLOAT(j + curand_uniform(&local_rand_state)) / FLOAT(image_height);
		gpu_ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += gpu_ray_colour(r, world, &local_rand_state, max_depth);
	}

	rand_state[pixel_index] = local_rand_state;

	col /= FLOAT(samples_per_pixel);
	col[0] = sqrt(col.x());
	col[1] = sqrt(col.y());
	col[2] = sqrt(col.z());
	fb[pixel_index] = col;
}

__global__ void create_world(gpu_hitable **obj_list, gpu_hitable **world, gpu_camera **camera, int nx, int ny, curandState *rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		obj_list[0] = new gpu_sphere(gpu_point3(0,-1000.0,-1), 1000,
			new gpu_lambertian(gpu_colour(0.5, 0.5, 0.5)));
		int i = 1;
		for(int a = -11; a < 11; a++)
		{
			for(int b = -11; b < 11; b++)
			{
				FLOAT choose_mat = curand_uniform(&local_rand_state);
				gpu_point3 center(a+curand_uniform(&local_rand_state), 0.2, b+curand_uniform(&local_rand_state));
				if(choose_mat < 0.8f)
				{
					obj_list[i++] = new gpu_sphere(center, 0.2,
						new gpu_lambertian(gpu_colour(curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state), curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state), curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state))));
				}
				else if(choose_mat < 0.95f)
				{
					obj_list[i++] = new gpu_sphere(center, 0.2,
						new gpu_metal(gpu_colour(0.5f*(1.0f+curand_uniform(&local_rand_state)), 0.5f*(1.0f+curand_uniform(&local_rand_state)), 0.5f*(1.0f+curand_uniform(&local_rand_state))), 0.5f*curand_uniform(&local_rand_state)));
				}
				else
				{
					obj_list[i++] = new gpu_sphere(center, 0.2, new gpu_dielectric(1.5));
				}
			}
		}
		obj_list[i++] = new gpu_sphere(gpu_point3(0, 1,0),  1.0,
			new gpu_dielectric(1.5));
		obj_list[i++] = new gpu_sphere(gpu_point3(-4, 1, 0), 1.0,
			new gpu_lambertian(gpu_colour(0.4, 0.2, 0.1)));
		obj_list[i++] = new gpu_sphere(gpu_point3(4, 1, 0),  1.0,
			new gpu_metal(gpu_colour(0.7, 0.6, 0.5), 0.0));
		auto fog =new gpu_sphere(gpu_point3(6, 1, 0), 1.0,
			new gpu_isotropic(new gpu_colour(1, 1, 1)));
		obj_list[i++] = new gpu_constant_medium(fog, 0.7, new gpu_colour(0.9, 0.9, 0.9));

		*rand_state = local_rand_state;
		*world  = new gpu_hitable_list(obj_list, 22*22+1+4);

		gpu_vec3 lookfrom(13,2,3);
		gpu_vec3 lookat(0,0,0);
		FLOAT dist_to_focus = 10.0; (lookfrom-lookat).length();
		FLOAT aperture = 0.05;
		*camera = new gpu_camera(lookfrom,
			lookat,
			gpu_vec3(0,1,0),
			20.0,
			FLOAT(nx)/FLOAT(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(gpu_hitable **obj_list, gpu_hitable **world, gpu_camera **camera)
{
	for(int i=0; i < 22*22+1+3; i++)
	{
		delete ((gpu_sphere *)obj_list[i])->mat_ptr;
		delete obj_list[i];
	}
	delete *world;
	delete *camera;
}

__host__ void start_gpu_render(gpu_colour *fb, screenInfo screen)
{
	clock_t start, stop;
	double timer_seconds;

	cudaDeviceProp prop{};
	//just use default device (0)
	cudaGetDeviceProperties(&prop, 0);

	std::cout << "Running on " << prop.name << std::endl;

	int blockY = 8;
	int blockX = 8;

	// allocate random state
	curandState *rand_state;
	checkCudaErrors(cudaMalloc((void **)&rand_state, screen.image_width*screen.image_height*sizeof(curandState)));
	curandState *rand_state2;
	checkCudaErrors(cudaMalloc((void **)&rand_state2, 1*sizeof(curandState)));

	rand_init<<<1,1>>>(rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	gpu_hitable **obj_list;
	int num_hitables = 22*22+1+4;
	checkCudaErrors(cudaMalloc((void **)&obj_list, num_hitables*sizeof(gpu_hitable *)));
	gpu_hitable **world;
	checkCudaErrors(cudaMalloc((void **)&world, sizeof(gpu_hitable *)));
	gpu_camera **camera;
	checkCudaErrors(cudaMalloc((void **)&camera, sizeof(gpu_camera *)));
	create_world<<<1,1>>>(obj_list, world, camera, screen.image_width, screen.image_height, rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	start = clock();
	// Render our buffer
	dim3 blocks(screen.image_width/blockX+1, screen.image_height/blockY+1);
	dim3 threads(blockX, blockY);
	gpu_render_init<<<blocks, threads>>>(screen.image_width, screen.image_height, rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	gpu_render<<<blocks, threads>>>(fb, screen.image_width, screen.image_height, screen.samples, camera, world, rand_state, screen.max_depth);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "took " << timer_seconds << " seconds with GPU.\n";

	free_world<<<1,1>>>(obj_list,world,camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(obj_list));
	checkCudaErrors(cudaFree(rand_state));
	checkCudaErrors(cudaFree(rand_state2));
}