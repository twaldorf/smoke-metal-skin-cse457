#ifndef RTIOW1_ROTATE_H
#define RTIOW1_ROTATE_H

#include "hitable.hpp"
#include "vec3.hpp"

class rotate_y : public hitable {
public:
    rotate_y(shared_ptr<hitable> _obj, double angle_deg)
            : obj(_obj) {
        double rads = degrees_to_radians(angle_deg);
        sin_theta = sin(rads);
        cos_theta = cos(rads);
    };

    bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;

    shared_ptr<hitable> obj;
    double sin_theta;
    double cos_theta;

};

#endif //RTIOW1_ROTATE_H
