#include "xz_rect.hpp"

//test if ray hits rectangle
bool xz_rect::hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const
{
    // Math to detect if ray is within rectangle
    FLOAT t = (y - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max )
        return false;

    FLOAT x = r.origin().x() + t * r.direction().x();
    FLOAT z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1) {
        // Ray is outside the rectangle
        return false;
    }

    //return info about the intersection
    rec.t = t;
    rec.p = r.at(rec.t);
    vec3 outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}
