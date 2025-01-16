//
//  msh.c
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "msh.h"


//init
void msh_ini(struct msh_obj *msh)
{
    msh->ne     = (cl_ulong3){1<<msh->le.x, 1<<msh->le.y, 1<<msh->le.z};
    msh->nv     = (cl_ulong3){msh->ne.x+1, msh->ne.y+1, msh->ne.z+1};
    
    msh->ne_tot = msh->ne.x*msh->ne.y*msh->ne.z;
    msh->nv_tot = msh->nv.x*msh->nv.y*msh->nv.z;
    
    msh->dx2    = msh->dx*msh->dx;
    msh->rdx2   = 1e0f/msh->dx2;
    msh->ne2    = (cl_float3){msh->ne.x/2e0, msh->ne.y/2e0, msh->ne.z/2e0};
    
//    printf("ne %llu,%llu,%llu \n", msh->ne.x, msh->ne.y, msh->ne.z);
//    printf("nv %llu,%llu,%llu \n", msh->nv.x, msh->nv.y, msh->nv.z);

    return;
}
