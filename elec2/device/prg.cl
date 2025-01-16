//
//  prg.cl
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "msh.cl"
#include "utl.cl"
#include "sdf.cl"
#include "geo.cl"
#include "ion.cl"
#include "mg.cl"



/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_ini(const struct msh_obj    msh,
                    global float            *uu,
                    global float            *bb,
                    global float            *rr,
                    global float            *aa)
{
    ulong3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    ulong  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = fn_x1(vtx_pos, &msh);
    
//    printf("%3lu %v3lu\n", vtx_idx, vtx_pos);
    
    uu[vtx_idx] = (fn_g0(x)<=0e0f);
    bb[vtx_idx] = 0e0f;     //fn_b1(x);
    rr[vtx_idx] = 0e0f;
    aa[vtx_idx] = fn_h1(x)<=0;     //fn_u1(x);
    
    return;
}

//residual
kernel void vtx_rsd(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb,
                    global float            *rr)
{
    ulong3 vtx_pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = fn_x1(vtx_pos, &msh);
    
    float  s = 0.0f;    //L+U
    float  d = 0.0f;    //D
    
    //stencil
    for(int k=0; k<6; k++)
    {
        ulong3  adj_pos = vtx_pos + off_fac[k];
        ulong   adj_idx = fn_idx1(adj_pos, msh.nv);
        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //domain

//        d += uu[vtx_idx];                             //zero dirichlet
        
        //domain
        if(adj_bnd)
        {
            d += uu[vtx_idx];                           //zero neumann
            s += uu[adj_idx];
        }
    }
    //constants
    float alp = MD_SIG_H*msh.dt*msh.rdx2;
    
    //operator (I-alp*A)*u
    float Au = uu[vtx_idx] - alp*(s - d);
    
    //residual
    rr[vtx_idx] = (fn_h1(x)<0e0f)?bb[vtx_idx] - Au:0e0f;    //geom
//    rr[vtx_idx] = bb[vtx_idx] - Au;                       //no geom

    return;
}


//jacobi
kernel void vtx_jac(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *rr)
{
    ulong3 vtx_pos = {get_global_id(0), get_global_id(1), get_global_id(2)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float  d = 0.0f;    //degree
    
    //stencil
    for(int k=0; k<6; k++)
    {
        ulong3  adj_pos = vtx_pos + off_fac[k];
        
//        d += 1e0f;                    //zero dirichlet
        
        //domain
        if(fn_bnd1(adj_pos, msh.nv))
        {
            d += 1e0f;                    //zero neumann
        }
    }
    //constants
    float alp = MD_SIG_H*msh.dt*msh.rdx2;
    
    //du = D^-1(r)
    uu[vtx_idx] += rr[vtx_idx]/(1e0f + alp*d);

    return;
}


