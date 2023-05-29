#ifndef RTIOW1_CONSTANT_MEDIUM_H
#define RTIOW1_CONSTANT_MEDIUM_H

#include "hitable.hpp"
#include "material.hpp"

class constant_medium : public hitable {
public:
    constant_medium(shared_ptr<hitable> b, double d, colour c)
    : boundary(b),
    neg_inv_density(-1/d),
    phase_function(make_shared<isotropic>(c)) {};

    bool hit( const ray& r, double t_min, double t_max, hit_record& rec)
    const override;

//    virtual bool bounding_box(double time0, double time1, aabb&& output_box) const override;

    shared_ptr<hitable> boundary;
    shared_ptr<material> phase_function;
    double  neg_inv_density;

};


#endif //RTIOW1_CONSTANT_MEDIUM_H
