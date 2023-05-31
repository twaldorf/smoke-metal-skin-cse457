#include "cube.hpp"
#include "xy_rect.hpp"
#include "xz_rect.hpp"
#include "yz_rect.hpp"

cube::cube(point3& p0, point3& p1, shared_ptr<material> m) {
    sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), m));
    sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), m));

    sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), m));
    sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), m));

    sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), m));
    sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), m));
};

bool cube::hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const {
    return sides.hit(r, t_min, t_max, rec);
}