#ifndef RTIOW1_YZ_RECT_H
#define RTIOW1_YZ_RECT_H

#include "hitable.hpp"
#include "vec3.hpp"

class yz_rect : public hitable {
public:
    yz_rect(double _y0, double _y1, double _z0, double _z1, double _x, shared_ptr<material> m)
            : y0(_y0), y1(_y1), z0(_z0), z1(_z1), x(_x), mat_ptr(m) {};

    bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;

    shared_ptr<material> mat_ptr;
    double y0, y1, z0, z1, x;

};

#endif //RTIOW1_YZ_RECT_H
