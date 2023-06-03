#include <iostream>
#include <syncstream>
#include <png.h>
#include <ctime>
#include <cstring>
#include <thread>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/bind.hpp>

#include "src/vec3.hpp"
#include "src/ray.hpp"
#include "src/hitable_list.hpp"
#include "src/png.hpp"
#include "src/util.hpp"
#include "src/camera.hpp"
#include "src/world_gen.hpp"

//CUDA only headers
#ifdef USE_CUDA
#include <curand_kernel.h>
#include "src/gpu/gpu_render.cuh"
#include "src/gpu/gpu_vec3.cuh"
#endif

struct renderInfo
{
	camera cam;
	const hitable_list& world;
	int blockX;
	int blockY;
	int blockSize;
	screenInfo screen;
};
void render(colour* fb, renderInfo rInfo);

int main(int argc, char **argv)
{
	//image
	const FLOAT aspect_ratio = 16.0/9.0;
	const int image_width = 3840;
	screenInfo screen
	{
		//image_width
		image_width, //1920 or 400
		//image_height
		static_cast<int>(image_width/aspect_ratio), //1080 or 225
		//samples
		500,
		//max_depth for recursion
		50
	};


	auto *row_pointers = (png_bytep*) malloc(screen.image_height * sizeof(png_bytep));
	for (int y = 0; y < screen.image_height; y++)
	{
		row_pointers[y] = (png_bytep) malloc(3*screen.image_width * sizeof(png_byte));
	}
	clock_t start, stop;
	double timer_seconds;

	#ifdef USE_CUDA
	if(argc >= 2)
	{
		if(strcmp(argv[1], "--gpu") == 0)
		{
			int pixels = screen.image_height*screen.image_width;
			size_t fb_size = pixels*sizeof(gpu_vec3);

			//framebuffer allocation
			gpu_colour *fb;
			checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

			start_gpu_render(fb, screen);

			for (int i = 0; i < screen.image_height; i++)
			{
				for (int j = 0; j < screen.image_width; j++)
				{
					size_t pixel_index = i*screen.image_width + j;
					row_pointers[screen.image_height - 1 - i][3*j] = static_cast<png_byte>(255.99*fb[pixel_index].x());
					row_pointers[screen.image_height - 1 - i][3*j+1] = static_cast<png_byte>(255.99*fb[pixel_index].y());
					row_pointers[screen.image_height - 1 - i][3*j+2] = static_cast<png_byte>(255.99*fb[pixel_index].z());
				}
			}

			save_as_png(screen.image_height, screen.image_width, row_pointers, "../imageGPU.png");
			for (int y = 0; y < screen.image_height; y++)
			{
				free(row_pointers[y]);
			}
			free(row_pointers);

			// clean up
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaFree(fb));

			cudaDeviceReset();
			return 0;
		}
	}
	#endif

	//setup CPU render
	const auto cpuCount = std::thread::hardware_concurrency();
	boost::asio::thread_pool thread_pool(cpuCount);
	std::cout << "Running on " << cpuCount << " threads." << std::endl;

	auto world = random_scene();

	uint fb_size = sizeof(colour)*screen.image_height*screen.image_width;
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
	int xBlocks = screen.image_width/blockSize;
	int yBlocks = screen.image_height/blockSize;

	renderInfo rInfo = {cam, world, 0, 0, blockSize, screen};

	for (int i = 0; i < xBlocks+1; i++)
	{
		for (int j = 0; j < yBlocks+1; j++)
		{
			rInfo.blockX = i;
			rInfo.blockY = j;
			boost::asio::post(thread_pool, [fb, rInfo] { return render(fb, rInfo); });
			//single threaded
			//render(fb, cam, rInfo);
		}
	}
	thread_pool.join();

	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC / cpuCount;
	std::cout << "took " << timer_seconds << " seconds with CPU\n";

	for (int i = 0; i < screen.image_height; i++)
	{
		for (int j = 0; j < screen.image_width; j++)
		{
			size_t pixel_index = i*screen.image_width + j;
			row_pointers[screen.image_height - 1 - i][3*j] = static_cast<png_byte>(255.99*fb[pixel_index].x());
			row_pointers[screen.image_height - 1 - i][3*j+1] = static_cast<png_byte>(255.99*fb[pixel_index].y());
			row_pointers[screen.image_height - 1 - i][3*j+2] = static_cast<png_byte>(255.99*fb[pixel_index].z());
		}
	}

	save_as_png(screen.image_height, screen.image_width, row_pointers, "../imageCPU.png");
	for (int y = 0; y < screen.image_height; y++)
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
		if(pixelX >= rInfo.screen.image_width)
			continue;

		for (int j = 0; j < rInfo.blockSize; j++)
		{
			int pixelY = (rInfo.blockY * rInfo.blockSize) + j;
			if(j >= rInfo.screen.image_height)
				continue;

			int pixelIndex = pixelX + pixelY*rInfo.screen.image_width;
			if(fb[pixelIndex][0] != 0.0 || fb[pixelIndex][1] != 0.0 || fb[pixelIndex][2] != 0.0)
				std::cerr << "Error: " << pixelX << "x" << pixelY << " pixel already filled!" << std::endl;

			for (int s = 0; s < rInfo.screen.samples; s++)
			{
				auto u = FLOAT(pixelX + random_float()) / (rInfo.screen.image_width);
				auto v = FLOAT(pixelY + random_float()) / (rInfo.screen.image_height);

				ray r = rInfo.cam.get_ray(u, v);
				fb[pixelIndex] += ray_colour(r, rInfo.world, rInfo.screen.max_depth);
			}

			auto scale = 1.0 / rInfo.screen.samples;
			fb[pixelIndex][0] = sqrt(fb[pixelIndex].x() * scale);
			fb[pixelIndex][1] = sqrt(fb[pixelIndex].y() * scale);
			fb[pixelIndex][2] = sqrt(fb[pixelIndex].z() * scale);
			//std::cerr << "Pixel: " << pixelX << "x" << pixelY << " done" << std::endl;
		}
	}
	std::osyncstream(std::cerr) << "Block: " << rInfo.blockX << "x" << rInfo.blockY << " done" << std::endl;
}