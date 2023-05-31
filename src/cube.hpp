#ifndef RTIOW1_CUBE_H
#define RTIOW1_CUBE_H

#include "hitable.hpp"
#include "hitable_list.hpp"
#include "vec3.hpp"
#include "xy_rect.hpp"
#include "xz_rect.hpp"
#include "yz_rect.hpp"

class cube : public hitable {
public:
    cube(point3& p0, point3& p1, shared_ptr<material> m);

    bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;

    shared_ptr<material> mat_ptr;
    hitable_list sides;

};

#endif //RTIOW1_CUBE_H
