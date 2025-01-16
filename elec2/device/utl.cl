//
//  utl.cl
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


/*
 ===================================
 constant
 ===================================
 */

//stencil
constant ulong3 off_fac[6] = {{-1,0,0},{+1,0,0},{0,-1,0},{0,+1,0},{0,0,-1},{0,0,+1}};

/*
 ===================================
 index
 ===================================
 */

//coordinate
float3 fn_x1(ulong3 pos, const struct msh_obj *msh)
{
    return msh->dx*(convert_float3(pos) - msh->ne2);
}

//global index
ulong fn_idx1(ulong3 pos, ulong3 dim)
{
    return pos.x + dim.x*(pos.y + dim.y*pos.z);
}

//index 3x3x3
ulong fn_idx3(ulong3 pos)
{
    return pos.x + 3*pos.y + 9*pos.z;
}

/*
 ===================================
 bounds
 ===================================
 */

//in-bounds
int fn_bnd1(ulong3 pos, ulong3 dim)
{
    return all(pos>=0)*all(pos<dim);
}

//on the boundary
int fn_bnd2(ulong3 pos, ulong3 dim)
{
    return any(pos==0)||any(pos==dim-1);    //not tested
    
//    return (pos.x==0)||(pos.y==0)||(pos.z==0)||(pos.x==dim.x-1)||(pos.y==dim.y-1)||(pos.z==dim.z-1);
}

/*
 ===================================
 data
 ===================================
 */

//solution
float fn_u1(float3 x)
{
    return sin(x.x);
}

//rhs
float fn_b1(float3 x)
{
    return  -sin(x.x);
}

