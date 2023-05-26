#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include "png.hpp"


void save_as_png(int height, int width, png_bytep *row_pointers, std::string filename)
{
	FILE *png_fp = fopen(filename.c_str(), "wb");
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, png_fp);

	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);
	png_write_image(png_ptr, row_pointers);
	png_write_end(png_ptr, nullptr);

	fclose(png_fp);
}