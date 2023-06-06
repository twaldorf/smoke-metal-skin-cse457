#include "yz_rect.hpp"

//test if ray hits rectangle
bool yz_rect::hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const
{
    // Math to detect if ray is within rectangle
    FLOAT t = (x - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max )
        return false;

    FLOAT y = r.origin().y() + t * r.direction().y();
    FLOAT z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1) {
        // Ray is outside the rectangle
        return false;
    }

    //return info about the intersection
    rec.t = t;
    rec.p = r.at(rec.t);
    vec3 outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}