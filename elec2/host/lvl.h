//
//  lvl.h
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef lvl_h
#define lvl_h

#include "msh.h"

//object
struct lvl_obj
{
    int idx;
    
    //mesh
    struct msh_obj  msh;
    
    //memory
    cl_mem          uu;
    cl_mem          bb;
    cl_mem          rr;
    cl_mem          aa;

    //kernels
    cl_kernel       vtx_ini;
    cl_kernel       vtx_zro;
    cl_kernel       vtx_prj;
    cl_kernel       vtx_itp;
    
    cl_kernel       vtx_rsd;
    cl_kernel       vtx_jac;
    
    cl_kernel       vtx_ion;
};

//methods
void lvl_ini(struct ocl_obj *ocl, struct lvl_obj *lvl);
void lvl_fin(struct ocl_obj *ocl, struct lvl_obj *lvl);

#endif /* lvl_h */
