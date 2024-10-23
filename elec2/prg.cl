//
//  prg.cl
//  elec1
//
//  Created by Toby Simpson on 08.02.24.
//


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
constant float MD_SIG_H     = 1.0f;          //conductivity (mS mm^-1) = muA mV^-1 mm^-1
constant float MD_SIG_T     = 10.0f;

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
    
    float   dx2; //dxˆ2
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


//cylinder z-axis
float sdf_cyl(float3 x, float3 r)
{
    //ellipse
    float2 p = r.xy*normalize(x.xy/r.xy);
    
    return max(length(x.xy) - length(p), fabs(x.z) - r.z);
}


/*
 ===================================
 geometry
 ===================================
 */

//torso
float fn_g0(float3 x)
{
    float3 r = (float3){256.0f, 128.0f, 128.0f};
    
    return sdf_cyl(x, r);
}


//epicardium
float fn_g1(float3 x)
{
    float3 c = (float3){0.0f, 0.0f, 0.0f};
    float3 r = (float3){40.0f, 30.0f, 60.0f};
    
    return sdf_cub(x, c, r);
    
    return sdf_cap(x, (float3){0e0f, 0e0f, -2e0f}, (float3){0e0f, 0e0f, +2e0f}, 6.0f);
}


//heart
float fn_g2(float3 x)
{
    //epicardium
    float s1 = fn_g1(x);
    
    //subtract endocardium for void
    float cap1 = sdf_cap(x, (float3){0e0f, 0e0f, -2e0f}, (float3){0e0f, 0e0f, +2e0f}, 4.0f);    //endo
    s1 = max(s1, -cap1);
    
    //insulate a/v
    float cyl1 = sdf_cyl(x, (float3){7e0f, 7e0f, 1e0f});
    s1 = max(s1, -cyl1);
    
    //add purk
    float cyl2 = sdf_cyl(x, (float3){1e0f, 1e0f, 7e0f});
    s1 = min(s1,cyl2);
    
    return s1;
}


//stimulus
float fn_g3(float3 x)
{
    float3 c = (float3){-4e0f, 0e0f, +5e0f};
    float  r = 1.0f;
    
    return sdf_sph(x, c, r);
}


/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_ini(const  struct msh_obj  msh,
                    global float4          *gg,
                    global float4          *uu)
{
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);

    float3 x = msh.dx*convert_float3(vtx_pos - msh.nv/2);

    gg[vtx_idx] = (float4){fn_g0(x), fn_g1(x), fn_g2(x), fn_g3(x)};
    uu[vtx_idx] = (float4){fn_g1(x)<=0e0f, 1.0f, 0e0f, 0e0f}; //stim
    
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

    //rhs for ie
    u.z = u.x;

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
        int     adj_bnd = fn_g2(adj_x)<=0e0f;   //zero neumann
        
        d -= adj_bnd;
        s += adj_bnd*(uu[adj_idx].x - u.x);
    }
    
    //params
    float alp = MD_SIG_H*msh.dt/msh.dx2;
    
    //laplace Dˆ-1(b-Au), b=0
//    uu[vtx_idx].x += -alp*s/d;
    
    //ie jacobi (I- alpD)ˆ-1 * (uˆt - (I - alpA)uˆk)), uˆk is the iterate, uˆt is rhs
    uu[vtx_idx].x += (fn_g2(x)<=0e0f)*(u.z - (u.x - alp*s))/(1.0f - alp*d);     //heart only
    
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
    float  s = 0.0f;            //sum
    float  d = 0.0f;            //diag
    
    //stencil
    for(int k=0; k<6; k++)
    {
        int3    adj_pos = vtx_pos + off_fac[k];
        int     adj_idx = fn_idx1(adj_pos, msh.nv);
        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //zero neumann
        
        d -= adj_bnd;
        s += adj_bnd*(uu[adj_idx].x - u.x); //zero neumann
//        s += (adj_bnd*uu[adj_idx].x - u.x); //zero dirichlet
    }
    
    //params
    float alp = MD_SIG_T*msh.dt/msh.dx2;
    
    //laplace Dˆ-1(b-Au), b=0
    uu[vtx_idx].x += (fn_g1(x)>0e0f)*-alp*s/d; //torso only, dirichlet on heart surface
    
    //ie jacobi (I- alpD)ˆ-1 * (uˆt - (I - alpA)uˆk)), uˆk is the iterate, uˆt is rhs
//    uu[vtx_idx].x += (fn_g1(x)>0e0f)*(u.w - (u.x - alp*s))/(1.0f - alp*d);
    
    return;
}


/*
 ===================================
 mg
 ===================================
 */

//reset
kernel void vtx_rst(const  struct msh_obj   msh,
                    global float4           *uu)
{
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);

    uu[vtx_idx].x = 0e0f;

    return;
}


//residual
kernel void vtx_res(const  struct msh_obj   msh,
                    global float4           *uu)
{
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = msh.dx*convert_float3(vtx_pos - msh.nv/2);

    float  s = 0.0f;            //sum
    float  d = 0.0f;            //diag
    
    //stencil
    for(int k=0; k<6; k++)
    {
        int3    adj_pos = vtx_pos + off_fac[k];
        int     adj_idx = fn_idx1(adj_pos, msh.nv);
        float3  adj_x = msh.dx*convert_float3(adj_pos - msh.nv/2);
        
        int     g0 = fn_g0(adj_x)<=0e0f;  //inside torso
        
        d -= g0;
        s += g0*(uu[adj_idx].x - uu[vtx_idx].x);
    }
    
    int g0 = fn_g0(x)<=0e0f;    //inside torso
    int g1 = fn_g1(x)>0e0f;     //outside epi
    
    //params
    float alp = MD_SIG_T/msh.dx2;
    
    //laplace Dˆ-1(b-Au), b=0
    uu[vtx_idx].w = (g0*g1)?(-alp*s):0e0f;
    
    return;
}


//jacobi
kernel void vtx_jac(const  struct msh_obj   msh,
                    global float4           *uu)
{
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = msh.dx*convert_float3(vtx_pos - msh.nv/2);

    float  s = 0.0f;            //sum
    float  d = 0.0f;            //diag
    
    //stencil
    for(int k=0; k<6; k++)
    {
        int3    adj_pos = vtx_pos + off_fac[k];
        int     adj_idx = fn_idx1(adj_pos, msh.nv);
        float3  adj_x = msh.dx*convert_float3(adj_pos - msh.nv/2);
        
        int     g0 = fn_g0(adj_x)<=0e0f;  //inside torso
       
        //zero neuman
        d -= g0;
        s += g0*(uu[adj_idx].x - uu[vtx_idx].x);
    }
    
    int g0 = fn_g0(x)<=0e0f;    //inside torso
    int g1 = fn_g1(x)>0e0f;     //outside epi
    
    //params
//    float alp = MD_SIG_T/msh.dx2;
    
    //laplace Dˆ-1(b-Au), b=0
    uu[vtx_idx].x += (g0*g1)?(-s/d):0e0f;
    
    return;
}



//projection
kernel void vtx_prj(const  struct msh_obj    msh,   //coarse    (out)
                    global float4            *u0,    //coarse    (out)
                    global float4            *u1)    //fine      (in)
{
    int3 vtx_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)}; //coarse
    int  vtx_idx0  = fn_idx1(vtx_pos, msh.nv);
    
    //injection
    int  vtx_idx1  = fn_idx1(2*vtx_pos, 2*msh.ne+1);
    
    //store r -> b
    u0[vtx_idx0].z = u1[vtx_idx1].w;

    return;
}


//interpolation
kernel void vtx_itp(const  struct msh_obj    msh,   //fine      (out)
                    global float4           *u0,    //coarse    (in)
                    global float4           *u1)    //fine      (out)
{
    int3 vtx_pos  = (int3){get_global_id(0), get_global_id(1), get_global_id(2)}; //fine
    int  vtx_idx  = fn_idx1(vtx_pos, msh.nv);   //fine
    
    //coarse
    float3 pos = convert_float3(vtx_pos)/2e0f;
    
    //round up/down
    int3 pos0 = convert_int3(floor(pos));
    int3 pos1 = convert_int3(ceil(pos));
    
    int3 dim = 1+(msh.nv-1)/2;
    
    float s = 0e0f;
    s += u0[fn_idx1((int3){pos0.x, pos0.y, pos0.z}, dim)].x;
    s += u0[fn_idx1((int3){pos1.x, pos0.y, pos0.z}, dim)].x;
    s += u0[fn_idx1((int3){pos0.x, pos1.y, pos0.z}, dim)].x;
    s += u0[fn_idx1((int3){pos1.x, pos1.y, pos0.z}, dim)].x;
    s += u0[fn_idx1((int3){pos0.x, pos0.y, pos1.z}, dim)].x;
    s += u0[fn_idx1((int3){pos1.x, pos0.y, pos1.z}, dim)].x;
    s += u0[fn_idx1((int3){pos0.x, pos1.y, pos1.z}, dim)].x;
    s += u0[fn_idx1((int3){pos1.x, pos1.y, pos1.z}, dim)].x;
    
    u1[vtx_idx].x += s/8e0f;
    
    return;
}
