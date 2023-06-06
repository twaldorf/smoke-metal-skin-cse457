#ifndef RTIOW1_XY_RECT_H
#define RTIOW1_XY_RECT_H

#include "hitable.hpp"
#include "vec3.hpp"

class xy_rect : public hitable {
public:
    xy_rect(double _x0, double _x1, double _y0, double _y1, double _z, shared_ptr<material> m)
    : x0(_x0), x1(_x1), y0(_y0), y1(_y1), z(_z), mat_ptr(m) {};

    bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;

    shared_ptr<material> mat_ptr;
    double x0, x1, y0, y1, z;

};

#endif //RTIOW1_XY_RECT_H
