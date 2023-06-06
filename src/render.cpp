#include <iostream>
#include <syncstream>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <thread>
#include "render.hpp"
#include "world_gen.hpp"

void cpuRender(colour* fb, screenInfo screen, FLOAT aspect_ratio)
{
	clock_t start, stop;
	double timer_seconds;

	//setup CPU render
	const auto cpuCount = std::thread::hardware_concurrency();
	boost::asio::thread_pool thread_pool(cpuCount);
	std::cout << "Running on " << cpuCount << " threads." << std::endl;

	auto world = random_scene();
	//auto world = spiral_scene();

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
			boost::asio::post(thread_pool, [fb, rInfo] { return cpuTrace(fb, rInfo); });
			//single threaded
			//cpuTrace(fb, rInfo);
		}
	}
	thread_pool.join();

	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC / cpuCount;
	std::cout << "took " << timer_seconds << " seconds with CPU\n";
}

void cpuTrace(colour* fb, renderInfo rInfo)
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