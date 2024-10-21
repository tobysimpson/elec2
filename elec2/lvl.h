//
//  lvl.h
//  elec1
//
//  Created by Toby Simpson on 09.10.2024.
//

#ifndef lvl_h
#define lvl_h


//object
struct lvl_obj
{
    int             idx;

    struct msh_obj  msh;
    
    size_t          nv[3];
    size_t          nv_tot;
    
    //memory
    cl_mem          xx;
    cl_mem          uu;
};



#endif /* lvl_h */
