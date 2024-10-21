//
//  prg.cl
//  elec1
//
//  Created by Toby Simpson on 08.02.24.
//

//#include "geo.h"

/*
 ===================================
 constant
 ===================================
 */

//mitchell-schaffer
constant float MS_V_GATE    = 0.13f;        //dimensionless (13 in the paper 15 for N?)
constant float MS_TAU_IN    = 0.3f;         //milliseconds
constant float MS_TAU_OUT   = 6.0f;         //should be 6.0
constant float MS_TAU_OPEN  = 120.0f;       //milliseconds
constant float MS_TAU_CLOSE = 100.0f;       //90 endocardium to 130 epi - longer

//conductivity
constant float MD_SIG_H     = 0.01f;          //conductivity (mS mm^-1) = muA mV^-1 mm^-1
constant float MD_SIG_T     = 0.5f;

//stencil
constant int3 off_fac[6]    = {{-1,0,0},{+1,0,0},{0,-1,0},{0,+1,0},{0,0,-1},{0,0,+1}};

/*
 ===================================
 struct
 ===================================
 */

//object
struct msh_obj
{
    float   dx;
    float   dt;
    
    int3    ne;
    int3    nv;
    
    int     ne_tot;
    int     nv_tot;
    
    float   dx2;
};

/*
 ===================================
 prototypes
 ===================================
 */

int     fn_idx1(int3 pos, int3 dim);
int     fn_idx3(int3 pos);

/*
 ===================================
 utilities
 ===================================
 */

//flat index
int fn_idx1(int3 pos, int3 dim)
{
    return pos.x + dim.x*(pos.y + dim.y*pos.z);
}

//index 3x3x3
int fn_idx3(int3 pos)
{
    return pos.x + 3*pos.y + 9*pos.z;
}

//in-bounds
int fn_bnd1(int3 pos, int3 dim)
{
    return all(pos>=0)*all(pos<dim);
}

//on the boundary
int fn_bnd2(int3 pos, int3 dim)
{
    return (pos.x==0)||(pos.y==0)||(pos.z==0)||(pos.x==dim.x-1)||(pos.y==dim.y-1)||(pos.z==dim.z-1);
}


/*
 ==============================
 sdf
 ==============================
 */


//cuboid
float sdf_cub(float3 x, float3 c, float3 r)
{
    float3 d = fabs(x - c) - r;
    
    return max(d.x,max(d.y,d.z));
}


//sphere
float sdf_sph(float3 x, float3 c, float r)
{
    return length(x - c) - r;
}


//capsule (from,to,r)
float sdf_cap(float3 p, float3 a, float3 b, float r)
{
    float3 pa = p - a;
    float3 ba = b - a;
    
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0f, 1.0f);
    
    return length(pa - ba*h) - r;
}


//cylinder z-axis (x,c,r,h)
float sdf_cyl(float3 x, float3 c, float r, float h)
{
    return max(length(x.xy - c.xy) - r, fabs(x.z - c.z) - h);
}


/*
 ===================================
 geometry
 ===================================
 */


//stimulus
float fn_g0(float3 x)
{
    float3 c = (float3){-4e0f, 0e0f, +5e0f};
    float  r = 1.0f;
    
    return sdf_sph(x, c, r);
}


////cube
//float fn_g1(float3 x)
//{
//    float3 c = (float3){0e0f, 0e0f, 0e0f};
//    float3 r = (float3){8e0f, 8e0f, 8e0f};
//
//    return sdf_cub(x, c, r);
//}

//epicardium
float fn_e1(float3 x)
{
    return sdf_cap(x, (float3){0e0f, 0e0f, -2e0f}, (float3){0e0f, 0e0f, +2e0f}, 6.0f);
}


//heart
float fn_g1(float3 x)
{
    //epicardium
    float s1 = fn_e1(x);
    
    //subtract endocardium for void
    float cap1 = sdf_cap(x, (float3){0e0f, 0e0f, -2e0f}, (float3){0e0f, 0e0f, +2e0f}, 4.0f);    //endo
    s1 = max(s1, -cap1);
    
    //insulate a/v
    float cyl1 = sdf_cyl(x, (float3){0e0f, 0e0f, 1e0f}, 7e0f, 1e0f); //horiz
    s1 = max(s1, -cyl1);
    
    //add purk
    float cyl2 = sdf_cyl(x, (float3){0e0f, 0e0f, 0e0f}, 1e0f, 7e0f);
    s1 = min(s1,cyl2);
    
    return s1;
}


/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_ini(const  struct msh_obj  msh,
                    global float4          *xx,
                    global float4          *uu)
{
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);

    float3 x = msh.dx*convert_float3(vtx_pos - msh.nv/2);

    xx[vtx_idx] = (float4){x, fn_g1(x)};
    uu[vtx_idx] = (float4){fn_g0(x)<=0e0f, 1.0f, 0e0f, 0e0f}; //stim
    
    return;
}


//mitchell-schaffer
kernel void vtx_ion(const  struct msh_obj  msh,
                    global float4          *uu)
{
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = msh.dx*convert_float3(vtx_pos - msh.nv/2);

    float4 u = uu[vtx_idx];
    float2 du = 0.0f;

    //mitchell-schaffer
    du.x = (u.y*u.x*u.x*(1.0f-u.x)/MS_TAU_IN) - (u.x/MS_TAU_OUT);               //ms dimensionless J_in, J_out, J_stim
    du.y = (u.x<MS_V_GATE)?((1.0f - u.y)/MS_TAU_OPEN):(-u.y)/MS_TAU_CLOSE;      //gating variable

    //update
    u.xy += (fn_g1(x)<= 0e0f)*msh.dt*du; //heart
    u.z += 1e0f;
    
    //rhs for ie
    u.w = u.x;

    //store
    uu[vtx_idx] = u;

    return;
}


//heart fdm
kernel void vtx_hrt(const  struct msh_obj  msh,
                    global float4          *uu)
{
    //adjust
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = msh.dx*convert_float3(vtx_pos - msh.nv/2);

    float4 u = uu[vtx_idx];     //centre
    float  s = 0.0f;             //sum
    float  d = 0.0f;             //diag
    
    //stencil
    for(int k=0; k<6; k++)
    {
        int3    adj_pos = vtx_pos + off_fac[k];
        int     adj_idx = fn_idx1(adj_pos, msh.nv);
        float3  adj_x   = msh.dx*convert_float3(adj_pos - msh.nv/2);
        int     adj_bnd = fn_g1(adj_x)<=0e0f;   //zero neumann
        
        d -= adj_bnd;
        s += adj_bnd*(uu[adj_idx].x - u.x);
    }
    
    //params
    float alp = MD_SIG_H*msh.dt/msh.dx2;
    
    //laplace Dˆ-1(b-Au), b=0
//    uu[vtx_idx].x += -alp*s/d;
    
    //ie jacobi (I- alpD)ˆ-1 * (uˆt - (I - alpA)uˆk)), uˆk is the iterate, uˆt is rhs
    uu[vtx_idx].x += (fn_g1(x)<=0e0f)*(u.w - (u.x - alp*s))/(1.0f - alp*d);     //heart only
    
    return;
}


//torso fdm
kernel void vtx_trs(const  struct msh_obj  msh,
                    global float4          *uu)
{
    //adjust
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = msh.dx*convert_float3(vtx_pos - msh.nv/2);

    float4 u = uu[vtx_idx];     //centre
    float  s = 0.0f;             //sum
    float  d = 0.0f;             //diag
    
    //stencil
    for(int k=0; k<6; k++)
    {
        int3    adj_pos = vtx_pos + off_fac[k];
        int     adj_idx = fn_idx1(adj_pos, msh.nv);
        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //zero meumann
        
        d -= adj_bnd;
        s += adj_bnd*(uu[adj_idx].x - u.x); //zero neumann
//        s += (adj_bnd*uu[adj_idx].x - u.x); //zero dirichlet
    }
    
    //params
    float alp = MD_SIG_T*msh.dt/msh.dx2;
    
    //laplace Dˆ-1(b-Au), b=0
    uu[vtx_idx].x += (fn_e1(x)>0e0f)*-alp*s/d; //torso only, dirichlet on heart surface
    
    //ie jacobi (I- alpD)ˆ-1 * (uˆt - (I - alpA)uˆk)), uˆk is the iterate, uˆt is rhs
//    uu[vtx_idx].x += (fn_g1(x)>0e0f)*(u.w - (u.x - alp*s))/(1.0f - alp*d);
    
    return;
}

