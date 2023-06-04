#ifndef RTIOW1_SRC_RENDER_HPP_
#define RTIOW1_SRC_RENDER_HPP_

#include "vec3.hpp"
#include "camera.hpp"
#include "hitable_list.hpp"

struct renderInfo
{
	camera cam;
	const hitable_list& world;
	int blockX;
	int blockY;
	int blockSize;
	screenInfo screen;
};

void cpuRender(colour* fb, screenInfo screen, FLOAT aspect_ratio);

void cpuTrace(colour* fb, renderInfo rInfo);

#endif //RTIOW1_SRC_RENDER_HPP_
