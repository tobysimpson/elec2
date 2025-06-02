//
//  msh.h
//  elec2
//
//  Created by Toby Simpson on 28.05.2025.
//

#ifndef msh_h
#define msh_h


//object
struct msh_obj
{
    int3    le;
    int3    ne;
    int3    nv;
    
    int     ne_tot;
    int     nv_tot;

    float   dt;
    float   dx;
    float   dx2;
    float   rdx;
    float   rdx2;
    
    ulong   nv_sz[3];
    ulong   ne_sz[3];
    ulong   iv_sz[3];
    ulong   ie_sz[3];
};


#endif /* msh_h */
