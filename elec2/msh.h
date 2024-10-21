//
//  msh.h
//  elec1
//
//  Created by Toby Simpson on 09.10.2024.
//

#ifndef msh_h
#define msh_h

//object
struct msh_obj
{
    float       dx;
    float       dt;
    
    cl_int3     ne;
    cl_int3     nv;
    
    int         ne_tot;
    int         nv_tot;
    
    float       dx2;
};


void msh_ini(struct msh_obj *msh)
{
    msh->dx2 = msh->dx*msh->dx;
    
    printf("dx %e\n", msh->dx);
    printf("dt %e\n", msh->dt);
    printf("ne %d %d %d\n", msh->ne.x, msh->ne.y, msh->ne.z);
    printf("nv %d %d %d\n", msh->nv.x, msh->nv.y, msh->nv.z);
    printf("ne_tot %d \n", msh->ne_tot);
    printf("nv_tot %d \n", msh->nv_tot);

    return;
}

#endif /* msh_h */
