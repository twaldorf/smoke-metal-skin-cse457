#ifndef RTIOW1_XZ_RECT_H
#define RTIOW1_XZ_RECT_H

#include "hitable.hpp"
#include "vec3.hpp"

class xz_rect : public hitable {
public:
    xz_rect(double _x0, double _x1, double _z0, double _z1, double _y, shared_ptr<material> m)
            : x0(_x0), x1(_x1), z0(_z0), z1(_z1), y(_y), mat_ptr(m) {};

    bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;

    shared_ptr<material> mat_ptr;
    double x0, x1, z0, z1, y;

};

#endif //RTIOW1_XZ_RECT_H
