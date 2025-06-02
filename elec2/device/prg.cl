//
//  prg.cl
//  mg3
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "msh.h"
#include "utl.h"
#include "geo.h"
#include "ion.h"


//monodomain
constant float MD_SIG_H     = 0.01f;          //heart conductivity (mS mm^-1) = muA mV^-1 mm^-1
constant float MD_SIG_T     = 0.50f;          //torso



/*
 ===================================
 ini
 ===================================
 */


kernel void ele_ini(const  struct msh_obj  msh,
                    global float           *uu,
                    global float           *bb,
                    global float           *rr,
                    global float           *gg)
{
    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);
    
    //write
    uu[ele_idx] = 0.5f*geo_g0(x)<=0e0f;
    bb[ele_idx] = 0e0f;
    rr[ele_idx] = 0e0f;
    gg[ele_idx] = geo_g1(x);
 
    return;
}



/*
 ============================
 poisson -Au=b
 ============================
 */


//forward
kernel void ele_fwd0(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);
    
    //torso
    if(geo_g1(x)>0e0f)
    {
        float s = 0.0f;

        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne);//zero neuman on domain
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                s += uu[adj_idx] - uu[ele_idx];
            }
        }
        
        //fwd
        bb[ele_idx] = MD_SIG_T*msh.rdx2*s;
    }
    
    return;
}



//residual
kernel void ele_res0(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb,
                    global float            *rr)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);
    
    //torso
    if(geo_g1(x)>0e0f)
    {
        float s = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne); //zero neumann on domain
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                s += uu[adj_idx] - uu[ele_idx];
            }
        }
        
        //fwd
        float Au = MD_SIG_T*msh.rdx2*s;
        
        //res
        rr[ele_idx] = bb[ele_idx] - Au;
    }
    
    return;
}


//jacobi
kernel void ele_jac0(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);
    
    //torso
    if(geo_g1(x)>0e0f)
    {
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne); //zero neumann on domain
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                d -= 1e0f;
                s += uu[adj_idx] - uu[ele_idx];
            }
        }
        
        //fwd
        float Au = MD_SIG_T*msh.rdx2*s;
        
        //res
        float r = bb[ele_idx] - Au;
        
        //du = D^-1(r)
        uu[ele_idx] += 0.9*msh.dx2*r/d;
    }
    
    return;
}


/*
 ================================================
 crank nicolson (I-alp*A)uˆ(t+1) = (I+alp*A)uˆ(t)
 ================================================
 */


//rhs crank
kernel void ele_fwd1(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);
    
    //heart
    if(geo_g1(x)<=0e0f)
    {
        float u = uu[ele_idx];
        
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            float3  adj_x   = ele_x(adj_pos, msh);
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne)*(geo_g1(adj_x)<=0e0f);
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                d -= 1e0f;
                s += uu[adj_idx];
            }
        }
        
        //constants
        float alp = MD_SIG_H*msh.dt*msh.rdx2;
        
        //rhs
        bb[ele_idx] = u + 0.5f*alp*(s + d*u);
    }
    
    return;
}



//residual crank
kernel void ele_res1(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb,
                    global float            *rr)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);
    
    //heart
    if(geo_g1(x)<=0e0f)
    {
        float u = uu[ele_idx];
        
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            float3  adj_x   = ele_x(adj_pos, msh);
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne)*(geo_g1(adj_x)<=0e0f);
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                d -= 1e0f;
                s += uu[adj_idx];
            }
        }
        
        //constants
        float alp = MD_SIG_H*msh.dt*msh.rdx2;
        
        //lhs
        float Au = u - 0.5f*alp*(s + d*u);
        
        //res
        rr[ele_idx] = bb[ele_idx] - Au;
    }
        
    return;
}


//jacobi crank
kernel void ele_jac1(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *bb)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);
    
    //heart
    if(geo_g1(x)<=0e0f)
    {
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            float3  adj_x   = ele_x(adj_pos, msh);
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne)*(geo_g1(adj_x)<=0e0f);
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                d -= 1e0f;
                s += uu[adj_idx];
            }
        }
        
        //constants
        float alp = MD_SIG_H*msh.dt*msh.rdx2;
        
        //ie
        uu[ele_idx] = (bb[ele_idx] + 0.5f*alp*s)/(1e0f - 0.5f*alp*d);
    }
    
    return;
}





/*
 ============================
 multigrid
 ============================
 */


//projection
kernel void ele_prj(const  struct msh_obj   mshc,    //coarse    (out)
                    global float            *rrf,    //fine      (in)
                    global float            *uuc,    //coarse    (out)
                    global float            *bbc)    //coarse    (out)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, mshc.ne);
    
    
    //fine
    int3 pos = 2*ele_pos;
    int3 dim = 2*mshc.ne;
    
    //sum
    float s = 0e0f;
    
    //sum fine
    for(int i=0; i<8; i++)
    {
        int3 adj_pos = pos + off_vtx[i];
        int  adj_idx = utl_idx1(adj_pos, dim);
        s += rrf[adj_idx];
    }
    
    //store/reset
    uuc[ele_idx] = 0e0f;
    bbc[ele_idx] = s;
    
    return;
}


//interp
kernel void ele_itp(const  struct msh_obj   mshf,    //fine      (out)
                    global float            *uuc,    //coarse    (in)
                    global float            *uuf)    //fine      (out)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, mshf.ne);   //fine
    
    //    printf("%2d %v3hlu\n", ele_idx, ele_pos/2);
    
    //coarse
    int3 pos = ele_pos/2;
    int3 dim = mshf.ne/2;
    
    //write - scale
    uuf[ele_idx] += 0.125f*uuc[utl_idx1(pos, dim)];
    
    return;
}

/*
 ============================
 error
 ============================
 */


//residual squared
kernel void ele_rsq(const  struct msh_obj   msh,
                    global float            *rr)
{
    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    //square/write
    rr[ele_idx] = pown(rr[ele_idx],2);
    
    return;
}


//error squared
kernel void ele_esq(const  struct msh_obj   msh,
                    global float            *uu,
                    global float            *aa,
                    global float            *rr)
{
    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    //square/write
    rr[ele_idx] = pown(aa[ele_idx] - uu[ele_idx],2);
    
    return;
}

/*
 ============================
 reduction
 ============================
 */


//fold
kernel void vec_sum(global float *uu,
                    const  int   n)
{
    int i = get_global_id(0);
    int m = get_global_size(0);

//    printf("%d %d %d %f %f\n",i, n, m, uu[i], uu[m+i]);
    
    if((m+i)<n)
    {
        uu[i] += uu[m+i];
    }
      
    return;
}
