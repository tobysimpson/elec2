//
//  msh.h
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef msh_h
#define msh_h

#include "math.h"
#include "ocl.h"


//object
struct msh_obj
{
    cl_float    dx;
    cl_float    dt;
    
    cl_uint3    le;
    cl_ulong3   ne;
    cl_ulong3   nv;
    
    cl_ulong    ne_tot;
    cl_ulong    nv_tot;
    
    cl_float    dx2;    //dx*dx
    cl_float    rdx2;   //1/(dx*dx)
    cl_float3   ne2;    //ne/2
};


//methods
void msh_ini(struct msh_obj *msh);

#endif /* msh_h */
