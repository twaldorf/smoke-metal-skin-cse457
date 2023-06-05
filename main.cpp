#include <iostream>
#include <syncstream>
#include <ctime>
#include <cstring>
#include <thread>

#include "src/vec3.hpp"
#include "src/ray.hpp"
#include "src/hitable_list.hpp"
#include "src/png.hpp"
#include "src/util.hpp"
#include "src/camera.hpp"
#include "src/world_gen.hpp"
#include "src/render.hpp"

//CUDA only headers
#ifdef USE_CUDA
#include <curand_kernel.h>
#include "src/gpu/gpu_render.cuh"
#include "src/gpu/gpu_vec3.cuh"
#include "src/gpu/optix/optix_host.hpp"
#endif


int main(int argc, char **argv)
{
	//image
	const FLOAT aspect_ratio = 16.0/9.0;
	const int image_width = 1920;
	screenInfo screen
	{
		//image_width
		image_width, //1920 or 400
		//image_height
		static_cast<int>(image_width/aspect_ratio), //1080 or 225
		//samples
		250,
		//max_depth for recursion
		50
	};

	auto *row_pointers = (png_bytep*) malloc(screen.image_height * sizeof(png_bytep));
	for (int y = 0; y < screen.image_height; y++)
	{
		row_pointers[y] = (png_bytep) malloc(3*screen.image_width * sizeof(png_byte));
	}

	#ifdef USE_CUDA
	if(argc >= 2)
	{
		if(strcmp(argv[1], "--gpu") == 0)
		{
			int pixels = screen.image_height*screen.image_width;
			size_t fb_size = pixels*sizeof(gpu_vec3f);

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
		if(strcmp(argv[1], "--optix") == 0)
		{
			optixRender(screen);
			return 0;
		}
	}
	#endif

	uint fb_size = sizeof(colour)*screen.image_height*screen.image_width;
	auto *fb = new colour[fb_size];
	cpuRender(fb, screen, aspect_ratio);


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
