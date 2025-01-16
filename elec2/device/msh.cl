//
//  msh.cl
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


//object
struct msh_obj
{
    float    dx;
    float    dt;
    
    uint3    le;
    ulong3   ne;
    ulong3   nv;
    
    ulong    ne_tot;
    ulong    nv_tot;
    
    float    dx2;    //dx*dx
    float    rdx2;   //1/(dx*dx)
    float3   ne2;    //ne/2
};
