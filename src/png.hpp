#ifndef RTIOW1_SRC_PNG_HPP_
#define RTIOW1_SRC_PNG_HPP_

#include <string>
#include <png.h>

void save_as_png(int height, int width, png_bytep *row_pointers, std::string filename);

#endif //RTIOW1_SRC_PNG_HPP_
