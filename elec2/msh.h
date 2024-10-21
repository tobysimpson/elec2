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
    
    float       dx2;
};



#endif /* msh_h */
