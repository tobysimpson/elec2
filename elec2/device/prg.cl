//
//  prg.cl
//  mg3
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "msh.h"
#include "utl.h"
#include "sdf.h"
#include "geo.h"


/*
 ===================================
 const
 ===================================
 */

//stencils
constant int3 off_fac[6]  = {{-1,0,0},{+1,0,0},{0,-1,0},{0,+1,0},{0,0,-1},{0,0,+1}};
constant int3 off_vtx[8]  = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};

//monodomain
constant float MD_SIG_H     = 0.25f;        //heart conductivity (mS mm^-1) = muA mV^-1 mm^-1
constant float MD_SIG_T     = 1e+0f;        //torso

//mitchell-schaffer
constant float MS_V_GATE    = 0.13f;        //dimensionless (13 in the paper 15 for N?)
constant float MS_TAU_IN    = 0.3f;         //milliseconds
constant float MS_TAU_OUT   = 6.0f;         //should be 6.0
constant float MS_TAU_OPEN  = 120.0f;       //milliseconds
constant float MS_TAU_CLOSE = 100.0f;       //90 endocardium to 130 epi - longer


/*
 ===================================
 ini
 ===================================
 */

kernel void ele_ini(const  struct msh_obj  msh,
                    global float           *uu,
                    global float           *ww,
                    global float4          *gg)
{
    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    float3 x = msh.dx*(convert_float3(ele_pos - msh.ne/2) + 0.5f);
        
    //write
    uu[ele_idx] = (float)((x.x<=-45.0f)*(gg[ele_idx].w<=0e0f));
    ww[ele_idx] = 1e0f;
    
    return;
}


kernel void ele_geo(const  struct msh_obj  msh,
                    global float4          *gg,
                    global float4          *ss)
{
    int3  ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    float3 x = msh.dx*(convert_float3(ele_pos - msh.ne/2) + 0.5f);
    
    //spheres
    float g = 1.0f;
    for(int i=0; i<150; i++)
    {
        g = sdf_smin(g , sdf_sph(x, ss[i].xyz, ss[i].w), 3.8f);
    }
    
    //sdf
//    float g = sdf_cub(x,(float3){0.0f,0.0f,0.0f}, (float3){50.0f,50.0f,50.0f});
    
    //fibres
//    float3 f = MD_SIG_H*(float3){all(x.yz>30.0f), (x.x>30.0f)*(x.z<-30.0f), all(x.xy>30.0f)} + (float3)0.05f; //along edges
    float3 f = MD_SIG_H*(float3){1.0f, 0.1f, 1.0f};
    
    //write
    gg[ele_idx] = (float4){f, g};
    
    return;
}


/*
 ===================================
 ion
 ===================================
 */

//mitchell-schaffer
kernel void ele_ion(const  struct msh_obj  msh,
                    global float           *uu,
                    global float           *ww,
                    global float4          *gg)
{
    int3 ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  ele_idx  = utl_idx1(ele_pos, msh.ne);

    float u = uu[ele_idx];
    float w = ww[ele_idx];
    
    //mitchell-schaffer
    float du = (w*u*u*(1.0f-u)/MS_TAU_IN) - (u/MS_TAU_OUT);                   //ms dimensionless J_in, J_out, J_stim
    float dw = (u<MS_V_GATE)?((1.0f - w)/MS_TAU_OPEN):(-w)/MS_TAU_CLOSE;      //gating variable

    //geom
    int g = gg[ele_idx].w<=0e0f;
    
    //store
    uu[ele_idx] += (g)?msh.dt*du:0e0f;
    ww[ele_idx] += (g)?msh.dt*dw:0e0f;

    return;
}



/*
 ============================
 ecg poisson Au=0
 ============================
 */


//residual
kernel void ele_res0(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float            *rr,
                     global float4           *gg)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    //torso
    if(gg[ele_idx].w>0e0f)
    {
        float s = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne); //zero neuman on domain, dirichlet on heart
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                s += uu[adj_idx] - uu[ele_idx];
            }
        }
        
        //res -Au
        rr[ele_idx] = -MD_SIG_T*msh.rdx2*s;
    }
    
    return;
}


//jacobi
kernel void ele_jac0(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float4           *gg)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);

    //torso
    if(gg[ele_idx].w>0e0f)
    {
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne); //zero neuman on domain, dirichlet on heart
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            
            if(adj_bnd)
            {
                d -= 1e0f;
                s += uu[adj_idx] - uu[ele_idx];
            }
        }
        
        //res -Au
        float r = -MD_SIG_T*msh.rdx2*s;
        
        //du = D^-1(r)
        uu[ele_idx] += 0.9f*msh.dx2*r/d;  //damp
    }
    
    return;
}

/*
 ================================================
 ion implicit euler (I-alp*A)uˆ(t+1) = uˆ(t)
 ================================================
 */

/*

//residual euler
kernel void ele_res1(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float            *rr,
                     global float4           *gg)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    //heart
    if(gg[ele_idx].w<=0e0f)
    {
        float u = uu[ele_idx];
        
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne)*(gg[adj_idx].w<=0e0f);
            
            if(adj_bnd)
            {
                d -= 1e0f;
                s += uu[adj_idx];
            }
        }
        
        //constants
        float alp = MD_SIG_H*msh.dt*msh.rdx2;
        
        //lhs
        float Au = u - alp*(s + d*u);
        
        //res
        rr[ele_idx] = bb[ele_idx] - Au;
    }
        
    return;
}


//jacobi euler
kernel void ele_jac1(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float4           *gg)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    //heart
    if(gg[ele_idx].w<=0e0f)
    {
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne)*(gg[adj_idx].w<=0e0f);
            
            if(adj_bnd)
            {
                d -= 1e0f;
                s += uu[adj_idx];
            }
        }
        
        //constants
        float alp = MD_SIG_H*msh.dt*msh.rdx2;
        
        //ie
        uu[ele_idx] = (bb[ele_idx] + alp*s)/(1e0f - alp*d);
    }
    
    return;
}

*/

/*
 ================================================
 ion implicit euler with fibres
 ================================================
 */

//residual euler
kernel void ele_res1(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float            *rr,
                     global float4           *gg)
{
    int3    ele_pos = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int     ele_idx = utl_idx1(ele_pos, msh.ne);
    
    //heart
    if(gg[ele_idx].w<=0e0f)
    {
        float u = uu[ele_idx];
        
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne)*(gg[adj_idx].w<=0e0f);
            
            if(adj_bnd)
            {
                //conductivity - interp fibre and dot
                float c = fabs(dot(0.5f*(gg[ele_idx] + gg[adj_idx]).xyz, convert_float3(off_fac[i])));
                
                d -= c;
                s += c*uu[adj_idx];
            }
        }
        
        //constants
        float alp = msh.dt*msh.rdx2;
        
        //lhs
        float Au = u - alp*(s + d*u);
        
        //res
        rr[ele_idx] = bb[ele_idx] - Au;
    }
        
    return;
}


//jacobi euler
kernel void ele_jac1(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float4           *gg)
{
    int3  ele_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};
    int   ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    //heart
    if(gg[ele_idx].w<=0e0f)
    {
        float s = 0.0f;
        float d = 0.0f;
        
        //stencil
        for(int i=0; i<6; i++)
        {
            int3    adj_pos = ele_pos + off_fac[i];
            int     adj_idx = utl_idx1(adj_pos, msh.ne);
            int     adj_bnd = utl_bnd1(adj_pos, msh.ne)*(gg[adj_idx].w<=0e0f);
            
            if(adj_bnd)
            {
                //conductivity - interp fibre and dot
                float c = fabs(dot(0.5f*(gg[ele_idx] + gg[adj_idx]).xyz, convert_float3(off_fac[i])));
                
                d -= c;
                s += c*uu[adj_idx];
            }
        }
        
        //constants
        float alp = msh.dt*msh.rdx2;
        
        //ie
        uu[ele_idx] = (bb[ele_idx] + alp*s)/(1e0f - alp*d);
    }
    
    return;
}



/*
 ============================
 multigrid
 ============================
 */


//projection - fvm
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


//interp - fvm
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
