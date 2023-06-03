#ifndef RTIOW1_TEXTURE_H
#define RTIOW1_TEXTURE_H

#include "util.hpp"
#include "vec3.hpp"

class texture {
public:
    virtual colour value(double u, double v, const point3& p) const = 0;
};

class solid_colour : public texture {
public:
    solid_colour() {}
    solid_colour(colour c) : colour_value(c) {}

    solid_colour(double red, double green, double blue)
            : solid_colour(colour(red,green,blue)) {}

    virtual colour value(double u, double v, const vec3& p) const override {
        return colour_value;
    }

private:
    colour colour_value;
};

#endif //RTIOW1_TEXTURE_H
