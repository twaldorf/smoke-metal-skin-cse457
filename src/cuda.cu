#ifdef USE_CUDA
#include <cstdlib>
#include <iostream>
#include "cuda.hpp"
#include "sphere.hpp"
#include "material.hpp"
#include "hitable_list.hpp"

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
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j*max_x + i;
	// Original: Each thread gets same seed, a different sequence number, no offset
	// curand_init(2023, pixel_index, 0, &rand_state[pixel_index]);
	// BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
	// performance improvement of about 2x!
	curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(gpu_vec3 *fb, int image_width, int image_height, int samples_per_pixel, gpu_camera **cam, gpu_hitable **world, curandState *rand_state, int max_depth)
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
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

__global__ void create_world(gpu_hitable **d_list, gpu_hitable **d_world, gpu_camera **d_camera, int nx, int ny, curandState *rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *rand_state;
		d_list[0] = new gpu_sphere(gpu_vec3(0,-1000.0,-1), 1000,
			new gpu_lambertian(gpu_vec3(0.5, 0.5, 0.5)));
		int i = 1;
		for(int a = -11; a < 11; a++)
		{
			for(int b = -11; b < 11; b++)
			{
				FLOAT choose_mat = curand_uniform(&local_rand_state);
				gpu_vec3 center(a+curand_uniform(&local_rand_state), 0.2, b+curand_uniform(&local_rand_state));
				if(choose_mat < 0.8f)
				{
					d_list[i++] = new gpu_sphere(center, 0.2,
						new gpu_lambertian(gpu_vec3(curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state), curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state), curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state))));
				}
				else if(choose_mat < 0.95f)
				{
					d_list[i++] = new gpu_sphere(center, 0.2,
						new gpu_metal(gpu_vec3(0.5f*(1.0f+curand_uniform(&local_rand_state)), 0.5f*(1.0f+curand_uniform(&local_rand_state)), 0.5f*(1.0f+curand_uniform(&local_rand_state))), 0.5f*curand_uniform(&local_rand_state)));
				}
				else
				{
					d_list[i++] = new gpu_sphere(center, 0.2, new gpu_dielectric(1.5));
				}
			}
		}
		d_list[i++] = new gpu_sphere(gpu_vec3(0, 1,0),  1.0,
			new gpu_dielectric(1.5));
		d_list[i++] = new gpu_sphere(gpu_vec3(-4, 1, 0), 1.0,
			new gpu_lambertian(gpu_vec3(0.4, 0.2, 0.1)));
		d_list[i++] = new gpu_sphere(gpu_vec3(4, 1, 0),  1.0,
			new gpu_metal(gpu_vec3(0.7, 0.6, 0.5), 0.0));
		*rand_state = local_rand_state;
		*d_world  = new gpu_hitable_list(d_list, 22*22+1+3);

		gpu_vec3 lookfrom(13,2,3);
		gpu_vec3 lookat(0,0,0);
		FLOAT dist_to_focus = 10.0; (lookfrom-lookat).length();
		FLOAT aperture = 0.05;
		*d_camera = new gpu_camera(lookfrom,
			lookat,
			gpu_vec3(0,1,0),
			20.0,
			FLOAT(nx)/FLOAT(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(gpu_hitable **d_list, gpu_hitable **d_world, gpu_camera **d_camera)
{
	for(int i=0; i < 22*22+1+3; i++)
	{
		delete ((gpu_sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

#endif