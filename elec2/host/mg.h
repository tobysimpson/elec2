//
//  mg.h
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef mg_h
#define mg_h

#include "lvl.h"


//object
struct mg_obj
{
    cl_float    dx;     //width
    cl_float    dt;     //time
    
    cl_uint3    le;     //fine log2(ne)
    cl_uint     nl;     //depth
    
    cl_uint     nj;     //jac iter
    cl_uint     nc;     //num cycles
    
    //levels
    struct lvl_obj *lvls;
};


//methods
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg);
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg);
void mg_slv(struct ocl_obj *ocl, struct mg_obj *mg);


#endif /* mg_h */
