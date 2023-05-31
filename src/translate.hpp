#ifndef RTIOW1_TRANSLATE_H
#define RTIOW1_TRANSLATE_H

#include "hitable.hpp"
#include "vec3.hpp"

class translate : public hitable {
public:
    translate(shared_ptr<hitable> _obj, const vec3& _offset)
            : obj(_obj), offset(_offset) {};

    bool hit(const ray& r, FLOAT t_min, FLOAT t_max, hit_record& rec) const override;

    shared_ptr<hitable> obj;
    vec3 offset;

};

#endif //RTIOW1_TRANSLATE_H
