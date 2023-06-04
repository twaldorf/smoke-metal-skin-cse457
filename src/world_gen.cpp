#include "world_gen.hpp"

#include <cmath>
#include "material.hpp"
#include "sphere.hpp"
#include "constant_medium.hpp"

//create random scene
hitable_list random_scene()
{
	//a world is just a list of objects you can hit
	hitable_list world;

	auto ground_material = make_shared<lambertian>(colour(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_float();
			point3 center(a + 0.9* random_float(), 0.2, b + 0.9* random_float());

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
					auto fuzz = random_float(0, 0.5);
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

    auto material4 = make_shared<isotropic>(colour(1, 1, 1));
    auto fogball = make_shared<sphere>(point3(6, 1, 0), 1.0, material4);
    world.add(make_shared<constant_medium>(fogball, 0.7, colour(0.9, 0.9, 0.9)));

	return world;
}

hitable_list spiral_scene() {
    hitable_list world;
    FLOAT population = 200.0;

    shared_ptr<material> sphere_material;
    for (int i = 0; i < population; i++) {

        // sf = scale factor
        FLOAT sf = static_cast<FLOAT>(i) / population;
        FLOAT radius = .5 + sf * sf * pi / 4;

        vec3 center = {
                static_cast<FLOAT>(-7 + 19 * sf),
                static_cast<FLOAT>(1 + radius * cos(i)),
                static_cast<FLOAT>(2.5 + radius * sin(i))
        };

        auto albedo = colour::random() * colour::random();
//        sphere_material = make_shared<lambertian>(albedo);
        sphere_material = make_shared<metal>(colour(0.7, 0.6, 0.5), 0.0);
        world.add(make_shared<sphere>(center, 0.2, sphere_material));
    }

    return world;
}