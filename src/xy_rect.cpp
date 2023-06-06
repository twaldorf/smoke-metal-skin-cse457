#include "xy_rect.hpp"

//test if ray hits rectangle
bool xy_rect::hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const
{
    // Math to detect if ray is within rectangle
    FLOAT t = (z - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max )
        return false;

    FLOAT x = r.origin().x() + t * r.direction().x();
    FLOAT y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1) {
        // Ray is outside the rectangle
        return false;
    }

    //return info about the intersection
    rec.t = t;
    rec.p = r.at(rec.t);
    vec3 outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}
