//
//  main.c
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include <stdio.h>
#include "mg.h"
#include "io.h"


//does with mg explicit euler and ms elec, no ecg yet

//doing mg as an include
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
//    printf("%lu %lu\n", sizeof(unsigned char), sizeof(char));
    
    //ocl
    struct ocl_obj ocl;
    ocl_ini(&ocl);
    
    //size
    int n  = 6;
    
    //mg
    struct mg_obj mg;
    mg.dx = 0.25f; //pow(2,-(n-1));
    mg.dt = 0.1f;
    mg.le = (cl_uint3){n,n,n};
    mg.nl = 4;
    mg.nj = 3;
    mg.nc = 1;
    
    //objects
    mg_ini(&ocl, &mg);
    
    //level
    struct lvl_obj lvl = mg.lvls[0];
    
    //ini
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl.vtx_ini, 3, NULL, (size_t*)&lvl.msh.nv, NULL, 0, NULL, NULL);
    
    //frames
    for(int f=0; f<100; f++)
    {
        printf("f %2d\n", f);
     
        //write
        wrt_xmf(&ocl, &lvl, f);
        wrt_raw(&ocl, &lvl, f);
        
        //elec iter
        for(int k=0; k<100; k++)
        {
            //elec
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl.vtx_ion, 3, NULL, (size_t*)&lvl.msh.nv, NULL, 0, NULL, NULL);
            
            //copy to rhs (for euler)
            ocl.err = clEnqueueCopyBuffer(ocl.command_queue, lvl.uu, lvl.bb, 0, 0, lvl.msh.nv_tot*sizeof(float), 0, NULL, NULL);
            
            //solve
            mg_slv(&ocl, &mg);
        }
    }
    
    //final
    ocl.err = clFlush(ocl.command_queue);
    ocl.err = clFinish(ocl.command_queue);
    

    //final
    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    printf("done\n");
    
    return 0;
}

