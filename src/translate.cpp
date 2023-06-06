#include "translate.hpp"

//translate the ray
bool translate::hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const
{
    // move ray to position where un-translated object is
    ray moved_r(r.origin() - offset, r.direction());

    if (!obj->hit(moved_r, t_min,  t_max, rec))
        return false;

    // Offset the intersection point
    rec.p += offset;
    rec.set_face_normal(moved_r, rec.normal);

    return true;
}