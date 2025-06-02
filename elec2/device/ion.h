//
//  ion.h
//  mg2
//
//  Created by Toby Simpson on 13.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


#ifndef ion_h
#define ion_h

#include "utl.h"
#include "geo.h"

//mitchell-schaffer
constant float MS_V_GATE    = 0.13f;        //dimensionless (13 in the paper 15 for N?)
constant float MS_TAU_IN    = 0.3f;         //milliseconds
constant float MS_TAU_OUT   = 6.0f;         //should be 6.0
constant float MS_TAU_OPEN  = 120.0f;       //milliseconds
constant float MS_TAU_CLOSE = 100.0f;       //90 endocardium to 130 epi - longer


//mitchell-schaffer
kernel void ele_ion(const  struct msh_obj  msh,
                    global float           *uu,
                    global float           *ww)
{
    int3 ele_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  ele_idx  = utl_idx1(ele_pos, msh.ne);
    
    float3 x = ele_x(ele_pos, msh);

    float u = uu[ele_idx];
    float w = ww[ele_idx];
    
    //mitchell-schaffer
    float du = (w*u*u*(1.0f-u)/MS_TAU_IN) - (u/MS_TAU_OUT);                   //ms dimensionless J_in, J_out, J_stim
    float dw = (u<MS_V_GATE)?((1.0f - w)/MS_TAU_OPEN):(-w)/MS_TAU_CLOSE;      //gating variable

    //store
    uu[ele_idx] += (geo_g1(x)<= 0e0f)?msh.dt*du:0e0f;
    ww[ele_idx] += (geo_g1(x)<= 0e0f)?msh.dt*dw:0e0f;

    return;
}


#endif
