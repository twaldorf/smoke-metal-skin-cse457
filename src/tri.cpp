#include "tri.hpp"

bool tri::hit(const ray &r, FLOAT t_min, FLOAT t_max, hit_record &rec) const {
    // if the ray is parallel to the tri, or hitting the wrong side, no intersection occurs (NIO)
    if (dot(normal, r.direction()) <= 0) {
        return false;
    }

    // get the distance to the triangle plane "along" the normal
    FLOAT d = -dot(normal, v1);

    // find the value of the parameter t in the parameterization Point = Origin + t * RayDirection
    FLOAT t = -(dot(normal, r.origin()) + d) / dot(normal, r.direction());

    // if the tri is behind the origin => NIO
    if (t < 0) {
        return false;
    }

    // check boundary
    if (t < t_min || t > t_max) {
        return false;
    }

    point3 p = r.origin() + t * r.direction();

    // inside-outside test
    // draw an edge between each vertex
    vec3 e1 = v2 - v1;
    // find the "c" vector from each vertex to the hit position (this lies within the plane)
    vec3 c1 = p - v1;
    // if any c vector is in line or to the right of any edge => we are outside the triangle hence NIO
    if (dot(normal, cross(e1, c1)) <= 0) {
        return false;
    }

    vec3 e2 = v3 - v2;
    vec3 c2 = p - v2;
    if (dot(normal, cross(e2, c2)) <= 0) {
        return false;
    }

    vec3 e3 = v1 - v3;
    vec3 c3 = p - v3;
    if (dot(normal, cross(e3, c3)) <= 0) {
        return false;
    }

    //return info about the intersection
    rec.t = t;
    rec.p = p;
    rec.set_face_normal(r, normal);
    rec.mat_ptr = mat_ptr;

    return true;
}