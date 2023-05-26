#include <iostream>
#include <png.h>
#include <cstring>

#include "src/hittable_list.hpp"
#include "src/sphere.hpp"
#include "src/png.hpp"
#include "src/utl.hpp"
#include "src/camera.hpp"
#include "src/material.hpp"

#ifdef USE_CUDA
#include <curand_kernel.h>
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
				  file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}
#endif

colour ray_colour(const ray& r, const hittable& world, int max_depth);

//create random scene
hittable_list random_scene() {
	//a world is just a list of objects you can hit
	hittable_list world;

	auto ground_material = make_shared<lambertian>(colour(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				shared_ptr<material> sphere_material;

				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = colour::random() * colour::random();
					sphere_material = make_shared<lambertian>(albedo);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				} else if (choose_mat < 0.95) {
					// metal
					auto albedo = colour::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = make_shared<metal>(albedo, fuzz);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				} else {
					// glass
					sphere_material = make_shared<dielectric>(1.5);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

	auto material2 = make_shared<lambertian>(colour(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = make_shared<metal>(colour(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	return world;
}

int main(int argc, char **argv)
{
	#ifdef USE_CUDA
	cudaDeviceProp prop{};
	//just use default device (0)
	cudaGetDeviceProperties(&prop, 0);

	if(argc >= 2)
	{
		if(strcmp(argv[1], "--gpu") == 0)
		{
			std::cout << "Running on " << prop.name << std::endl;
		}
	}
	#endif

	//image
	const auto aspect_ratio = 16.0/9.0;
	const int image_width = 400; //1920 or 400
	const int image_height = static_cast<int>(image_width/aspect_ratio); //1080 or 225
	const int samples_per_pixel = 50;
	const int max_depth = 50;

//	//manually specify world
//	hittable_list world;
//
//	auto material_ground = make_shared<lambertian>(colour(0.1, 0.1, 0.8));
//	auto material_center = make_shared<lambertian>(colour(0.7, 0.3, 0.9));
//	auto material_left = make_shared<dielectric>(1.5);
//	auto material_front = make_shared<dielectric>(2.4);
//	auto material_right= make_shared<metal>(colour(0.3, 0.9, 0.5), 0.3);
//
//	world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
//	world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.0),   0.5, material_center));
//	//sphere in sphere to create hollow sphere (note negative radius to flip the normal
//	world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
//	world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),  -0.4, material_left));
//	world.add(make_shared<sphere>(point3(0.5,    -0.4, -0.7),   0.1, material_front));
//	world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));
	auto world = random_scene();

	//camera
	point3 lookfrom(13,2,3);
	point3 lookat(0,0,0);
	vec3 vup(0,1,0);
	auto dist_to_focus = 10.0;
	auto aperture = 0.05; //bigger = more DoF

	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

	//render
	auto *row_pointers = (png_bytep*) malloc(image_height * sizeof(png_bytep));
	for (int y = 0; y < image_height; y++)
	{
		row_pointers[y] = (png_bytep) malloc(3*image_width * sizeof(png_byte));
	}

	for(int i = 0; i < image_height; i++)
	{
		//if(i % 10 == 0)
			std::cerr << "\rScanlines remaining: " << (image_height-i) << ' ' << std::endl;
		for(int j = 0; j < image_width; j++)
		{
			colour pixel_colour(0,0,0);
			for(int s = 0; s < samples_per_pixel; s++)
			{
				auto u = double(j + random_double()) / (image_width-1);
				auto v = double(image_height - i + random_double()) / (image_height-1);

				ray r = cam.get_ray(u, v);
				pixel_colour += ray_colour(r, world, max_depth);
			}

			//write_colour(std::cout, pixel_colour, samples_per_pixel);
			auto scale = 1.0 / samples_per_pixel;
			auto r = sqrt(pixel_colour.x() * scale);
			auto g = sqrt(pixel_colour.y() * scale);
			auto b = sqrt(pixel_colour.z() * scale);
			row_pointers[i][3*j] = static_cast<png_byte>(256 * clamp(r, 0.0, 0.999));
			row_pointers[i][3*j+1] = static_cast<png_byte>(256 * clamp(g, 0.0, 0.999));
			row_pointers[i][3*j+2] = static_cast<png_byte>(256 * clamp(b, 0.0, 0.999));
		}
	}

	save_as_png(image_height, image_width, row_pointers, "../image.png");
	for (int y = 0; y < image_height; y++)
	{
		free(row_pointers[y]);
	}
	free(row_pointers);

	std::cerr << "\nDone.\n";
	return 0;
}

//get the colour of the ray
colour ray_colour(const ray& r, const hittable& world, int depth)
{
	//hit_detail records details of the intersection
	hit_record rec;

	//prevents recursion beyond max_depth
	if(depth <= 0)
		return {0,0,0};

	if(world.hit(r, 0.001, infinity, rec))
	{
		ray scattered;
		colour attenuation;
		if(rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * ray_colour(scattered, world, depth-1);
		return {0,0,0};
	}

	//sky
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5*(unit_direction.y() + 1.0);
	return (1.0-t)*colour(1.0, 1.0, 1.0) + t*colour(0.5, 0.7, 1.0);
}