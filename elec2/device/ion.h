//
//  ion.cl
//  mg2
//
//  Created by Toby Simpson on 13.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//



//mitchell-schaffer
constant float MS_V_GATE    = 0.13f;        //dimensionless (13 in the paper 15 for N?)
constant float MS_TAU_IN    = 0.3f;         //milliseconds
constant float MS_TAU_OUT   = 6.0f;         //should be 6.0
constant float MS_TAU_OPEN  = 120.0f;       //milliseconds
constant float MS_TAU_CLOSE = 100.0f;       //90 endocardium to 130 epi - longer


//conductivity
constant float MD_SIG_H     = 0.01f;          //conductivity (mS mm^-1) = muA mV^-1 mm^-1
constant float MD_SIG_T     = 0.5f;


//mitchell-schaffer
kernel void vtx_ion(const  struct msh_obj  msh,
                    global float           *uu,
                    global float           *aa)
{
    ulong3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    ulong  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float3 x = fn_x1(vtx_pos, &msh);

    float u = uu[vtx_idx];
    float a = aa[vtx_idx];
    
    //mitchell-schaffer
    float du = (a*u*u*(1.0f-u)/MS_TAU_IN) - (u/MS_TAU_OUT);                   //ms dimensionless J_in, J_out, J_stim
    float da = (u<MS_V_GATE)?((1.0f - a)/MS_TAU_OPEN):(-a)/MS_TAU_CLOSE;      //gating variable

    //store
    uu[vtx_idx] += (fn_h1(x)<= 0e0f)?msh.dt*du:0e0f;
    aa[vtx_idx] += (fn_h1(x)<= 0e0f)?msh.dt*da:0e0f;

    return;
}
