#include <iostream>
#include <png.h>
#include <ctime>
#include <cstring>
#include <thread>

#include "src/vec3.hpp"
#include "src/ray.hpp"
#include "src/hitable_list.hpp"
#include "src/sphere.hpp"
#include "src/png.hpp"
#include "src/util.hpp"
#include "src/camera.hpp"
#include "src/material.hpp"
#include "src/world_gen.hpp"

//CUDA only headers
#ifdef USE_CUDA
#include <curand_kernel.h>
#include "src/cuda.hpp"
#endif

void render(colour* fb, camera cam, const hitable_list& world, int blockX, int blockY, int blockSizeX, int blockSizeY, int image_height, int image_width, int samples, int max_depth);

int main(int argc, char **argv)
{
	//image
	const FLOAT aspect_ratio = 16.0/9.0;
	const int image_width = 1080; //1920 or 400
	const int image_height = static_cast<int>(image_width/aspect_ratio); //1080 or 225
	const int samples_per_pixel = 250;
	const int max_depth = 50;

	auto *row_pointers = (png_bytep*) malloc(image_height * sizeof(png_bytep));
	for (int y = 0; y < image_height; y++)
	{
		row_pointers[y] = (png_bytep) malloc(3*image_width * sizeof(png_byte));
	}
	clock_t start, stop;
	double timer_seconds;

	#ifdef USE_CUDA
	cudaDeviceProp prop{};
	//just use default device (0)
	cudaGetDeviceProperties(&prop, 0);

	if(argc >= 2)
	{
		if(strcmp(argv[1], "--gpu") == 0)
		{
			std::cout << "Running on " << prop.name << std::endl;


			int blockY = 8;
			int blockX = 8;
			int pixels = image_height*image_width;
			size_t fb_size = pixels*sizeof(gpu_vec3);

			//framebuffer allocation
			gpu_colour *fb;
			checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
			// allocate random state
			curandState *d_rand_state;
			checkCudaErrors(cudaMalloc((void **)&d_rand_state, pixels*sizeof(curandState)));
			curandState *d_rand_state2;
			checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

			rand_init<<<1,1>>>(d_rand_state2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			gpu_hitable **d_list;
			int num_hitables = 22*22+1+3;
			checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(gpu_hitable *)));
			gpu_hitable **d_world;
			checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(gpu_hitable *)));
			gpu_camera **d_camera;
			checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(gpu_camera *)));
			create_world<<<1,1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			start = clock();
			// Render our buffer
			dim3 blocks(image_width/blockX+1, image_height/blockY+1);
			dim3 threads(blockX, blockY);
			gpu_render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			gpu_render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state, max_depth);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			stop = clock();
			timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
			std::cout << "took " << timer_seconds << " seconds with GPU.\n";

			for (int i = 0; i < image_height; i++)
			{
				for (int j = 0; j < image_width; j++)
				{
					size_t pixel_index = i*image_width + j;
					row_pointers[image_height - 1 - i][3*j] = static_cast<png_byte>(255.99*fb[pixel_index].x());
					row_pointers[image_height - 1 - i][3*j+1] = static_cast<png_byte>(255.99*fb[pixel_index].y());
					row_pointers[image_height - 1 - i][3*j+2] = static_cast<png_byte>(255.99*fb[pixel_index].z());
				}
			}

			save_as_png(image_height, image_width, row_pointers, "../imageGPU.png");
			for (int y = 0; y < image_height; y++)
			{
				free(row_pointers[y]);
			}
			free(row_pointers);

			// clean up
			checkCudaErrors(cudaDeviceSynchronize());
			free_world<<<1,1>>>(d_list,d_world,d_camera);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaFree(d_camera));
			checkCudaErrors(cudaFree(d_world));
			checkCudaErrors(cudaFree(d_list));
			checkCudaErrors(cudaFree(d_rand_state));
			checkCudaErrors(cudaFree(d_rand_state2));
			checkCudaErrors(cudaFree(fb));

			cudaDeviceReset();
			return 0;
		}
	}
	#endif

	const auto cpuCount = std::thread::hardware_concurrency();
	std::cout << "Running on " << cpuCount << " threads." << std::endl;

	//setup CPU render
	auto world = random_scene();

	uint fb_size = sizeof(colour)*image_height*image_width;
	auto *fb = new colour[fb_size];

	//camera
	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	FLOAT dist_to_focus = 10.0;
	FLOAT aperture = 0.05; //bigger = more DoF

	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

	start = clock();
	for (int i = 0; i < image_height; i++)
	{
		if (i % 10 == 0)
			std::cerr << "\rScanlines remaining: " << (image_height - i) << ' ' << std::endl;
		for (int j = 0; j < image_width; j++)
		{
			render(fb, cam, world, ?,?, 8, 8, image_height, image_width, samples_per_pixel, max_depth);
		}
	}
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "took " << timer_seconds << " seconds with CPU\n";

	for (int i = 0; i < image_height; i++)
	{
		for (int j = 0; j < image_width; j++)
		{
			auto scale = 1.0 / samples_per_pixel;
			auto r = sqrt(fb[i*image_height + j].x() * scale);
			auto g = sqrt(fb[i*image_height + j].y() * scale);
			auto b = sqrt(fb[i*image_height + j].z() * scale);
			row_pointers[i][3 * j] = static_cast<png_byte>(256 * clamp(r, 0.0, 0.999));
			row_pointers[i][3 * j + 1] = static_cast<png_byte>(256 * clamp(g, 0.0, 0.999));
			row_pointers[i][3 * j + 2] = static_cast<png_byte>(256 * clamp(b, 0.0, 0.999));
		}
	}
	save_as_png(image_height, image_width, row_pointers, "../imageCPU.png");
	for (int y = 0; y < image_height; y++)
	{
		free(row_pointers[y]);
	}
	free(row_pointers);

	std::cerr << "\nDone.\n";
	return 0;
}

void render(colour* fb, camera cam, const hitable_list& world, int blockX, int blockY, int blockSizeX, int blockSizeY, int image_height, int image_width, int samples, int max_depth)
{
	for(int i = 0; i < blockSizeX; i++)
	{
		for (int j = 0; j < blockSizeY; j++)
		{
			for (int s = 0; s < samples; s++)
			{
				auto u = FLOAT(j + random_float()) / (image_width - 1);
				auto v = FLOAT(image_height - i + random_float()) / (image_height - 1);

				ray r = cam.get_ray(u, v);
				fb[i * image_height + j] += ray_colour(r, world, max_depth);
			}
		}
	}
}