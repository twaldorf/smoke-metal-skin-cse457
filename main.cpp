#include <iostream>
#include <syncstream>
#include <png.h>
#include <ctime>
#include <cstring>
#include <thread>
//#include <boost/asio/post.hpp>
//#include <boost/asio/thread_pool.hpp>
//#include <boost/bind.hpp>

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
#include "src/gpu/cuda.hpp"
#include "src/gpu/gpu_camera.hpp"
#include "src/gpu/gpu_hitable_list.hpp"
#include "src/gpu/gpu_vec3.hpp"
#endif

struct renderInfo
{
	camera cam;
	const hitable_list& world;
	int blockX;
	int blockY;
	int blockSize;
	int image_height;
	int image_width;
	int samples;
	int max_depth;
};
void render(colour* fb, renderInfo rInfo);

int main(int argc, char **argv)
{
	//image
	const FLOAT aspect_ratio = 16.0/9.0;
	const int image_width = 1280; //1920 or 400
	const int image_height = static_cast<int>(image_width/aspect_ratio); //1080 or 225
	const int samples_per_pixel = 50;
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

	int blockSize = 32;
	start = clock();
	int xBlocks = image_width/blockSize;
	int yBlocks = image_height/blockSize;
	auto *threads = new std::thread[(xBlocks+1) * (yBlocks+1)];
	//boost::asio::thread_pool thread_pool(cpuCount);

	renderInfo rInfo = {cam, world, 0, 0, blockSize, image_height, image_width, samples_per_pixel, max_depth};

	for (int i = 0; i < xBlocks+1; i++)
	{
		for (int j = 0; j < yBlocks+1; j++)
		{
			rInfo.blockX = i;
			rInfo.blockY = j;
			//boost::asio::post(thread_pool, [fb, capture0 = rInfo] { return render(fb, capture0); });
			threads[(yBlocks+1)*i + j] = std::thread(render, fb, rInfo);
			//render(fb, cam, rInfo);
		}
	}
	for(int i = 0; i < (xBlocks+1) * (yBlocks+1); i++)
	{
		threads[i].join();
	}
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC / cpuCount;
	std::cout << "took " << timer_seconds << " seconds with CPU\n";

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

	save_as_png(image_height, image_width, row_pointers, "../imageCPU.png");
	for (int y = 0; y < image_height; y++)
	{
		free(row_pointers[y]);
	}
	free(row_pointers);

	std::cerr << "\nDone.\n";
	return 0;
}

void render(colour* fb, renderInfo rInfo)
{
	for(int i = 0; i < rInfo.blockSize; i++)
	{
		int pixelX = rInfo.blockX*rInfo.blockSize + i;
		if(pixelX >= rInfo.image_width)
			continue;

		for (int j = 0; j < rInfo.blockSize; j++)
		{
			int pixelY = (rInfo.blockY * rInfo.blockSize) + j;
			if(j >= rInfo.image_height)
				continue;

			int pixelIndex = pixelX + pixelY*rInfo.image_width;
			if(fb[pixelIndex][0] != 0.0 || fb[pixelIndex][1] != 0.0 || fb[pixelIndex][2] != 0.0)
				std::cerr << "Error: " << pixelX << "x" << pixelY << " pixel already filled!" << std::endl;

			for (int s = 0; s < rInfo.samples; s++)
			{
				auto u = FLOAT(pixelX + random_float()) / (rInfo.image_width);
				auto v = FLOAT(pixelY + random_float()) / (rInfo.image_height);

				ray r = rInfo.cam.get_ray(u, v);
				fb[pixelIndex] += ray_colour(r, rInfo.world, rInfo.max_depth);
			}

			auto scale = 1.0 / rInfo.samples;
			fb[pixelIndex][0] = sqrt(fb[pixelIndex].x() * scale);
			fb[pixelIndex][1] = sqrt(fb[pixelIndex].y() * scale);
			fb[pixelIndex][2] = sqrt(fb[pixelIndex].z() * scale);
			//std::cerr << "Pixel: " << pixelX << "x" << pixelY << " done" << std::endl;
		}
	}
	std::osyncstream(std::cerr) << "Block: " << rInfo.blockX << "x" << rInfo.blockY << " done" << std::endl;
}