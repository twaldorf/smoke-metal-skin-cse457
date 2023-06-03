#ifndef RTIOW1_TRI_HPP
#define RTIOW1_TRI_HPP

#include <utility>

#include "hitable.hpp"
#include "vec3.hpp"

// triangles defined CCW
class tri : public hitable {
public:
    tri(point3 v1, point3 v2, point3 v3, shared_ptr<material> m)
        : normal(unit_vector(cross(v2 - v1, v3 - v1))),
        v1(v1), v2(v2), v3(v3),
        mat_ptr(m) {};

    bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;
    shared_ptr<material> mat_ptr;
    point3 normal;

    point3 v1;
    point3 v2;
    point3 v3;
};


#endif //RTIOW1_TRI_HPP
