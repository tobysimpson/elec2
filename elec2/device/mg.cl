//
//  mg.cl
//  mg2
//
//  Created by Toby Simpson on 13.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//



//reset
kernel void vtx_zro(const struct msh_obj    msh,
                    global float            *uu)
{
    ulong3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    ulong  vtx_idx  = fn_idx1(vtx_pos, msh.nv);

    uu[vtx_idx] = 0e0f;

    return;
}


//projection
kernel void vtx_prj(const  struct msh_obj   msh,    //coarse    (out)
                    global float            *bb,    //coarse    (out)
                    global float            *rr)    //fine      (in)
{
    ulong3 vtx_pos   = {get_global_id(0), get_global_id(1), get_global_id(2)}; //coarse
    ulong  vtx_idx0  = fn_idx1(vtx_pos, msh.nv);
    
    //injection
    ulong  vtx_idx1  = fn_idx1(2*vtx_pos, 2*msh.ne+1);
    
    //store
    bb[vtx_idx0] = rr[vtx_idx1];

    return;
}


//interpolation
kernel void vtx_itp(const  struct msh_obj   msh,    //fine      (out)
                    global float            *u0,    //coarse    (in)
                    global float            *u1)    //fine      (out)
{
    ulong3 vtx_pos = {get_global_id(0), get_global_id(1), get_global_id(2)}; //fine
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);   //fine
    
    //coarse
    float3 pos = convert_float3(vtx_pos)/2e0f;
    
    //round up/down
    ulong3 pos0 = convert_ulong3(floor(pos));
    ulong3 pos1 = convert_ulong3(ceil(pos));
    
    ulong3 dim = 1+(msh.nv-1)/2;
    
    float s = 0e0f;
    s += u0[fn_idx1((ulong3){pos0.x, pos0.y, pos0.z}, dim)];
    s += u0[fn_idx1((ulong3){pos1.x, pos0.y, pos0.z}, dim)];
    s += u0[fn_idx1((ulong3){pos0.x, pos1.y, pos0.z}, dim)];
    s += u0[fn_idx1((ulong3){pos1.x, pos1.y, pos0.z}, dim)];
    s += u0[fn_idx1((ulong3){pos0.x, pos0.y, pos1.z}, dim)];
    s += u0[fn_idx1((ulong3){pos1.x, pos0.y, pos1.z}, dim)];
    s += u0[fn_idx1((ulong3){pos0.x, pos1.y, pos1.z}, dim)];
    s += u0[fn_idx1((ulong3){pos1.x, pos1.y, pos1.z}, dim)];
    
    u1[vtx_idx] += s/8e0f;
    
    return;
}



