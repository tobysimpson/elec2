//
//  main.c
//  elec1
//
//  Created by Toby Simpson on 05.02.24.
//

#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//#include <Accelerate/Accelerate.h>

#include "ocl.h"
#include "msh.h"
#include "lvl.h"
#include "mg.h"
#include "io.h"


//monodomain/isotropic diffusion - try mg
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //ocl
    struct ocl_obj ocl;
    ocl_ini(&ocl);
    
    //mg
    struct mg_obj mg;
    mg_ini(&mg, &ocl);

    /*
     ==============================
     init
     ==============================
     */
    
    for(int i=0; i<mg.nl; i++)
    {
        //obj
        struct lvl_obj *lvl = &mg.lvls[i];
        
        //args
        ocl.err = clSetKernelArg(ocl.vtx_ini,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
        ocl.err = clSetKernelArg(ocl.vtx_ini,  1, sizeof(cl_mem),           (void*)&lvl->xx);
        ocl.err = clSetKernelArg(ocl.vtx_ini,  2, sizeof(cl_mem),           (void*)&lvl->uu);
           
        //init
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_ini, 3, NULL, (size_t*)&lvl->nv, NULL, 0, NULL, NULL);
        
        //write
        wrt_vtk(lvl, &ocl, 0);
    }
    

    /*
     ==============================
     solve
     ==============================
     */
    
    //init
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl.vtx_ini, 3, NULL, nv, NULL, 0, NULL, NULL);
    
//    //time
//    for(int t=0; t<100; t++)
//    {
//        printf("%02d\n",t);
//        
//        //write vtk
//        wrt_vtk(&lvl, &ocl, t);
//
//        //elec iter
//        for(int k=0; k<100; k++)
//        {
//            //calc
//            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl.vtx_ion, 3, NULL, nv, NULL, 0, NULL, NULL);
//
//            //heart jacobi
//            for(int l=0; l<10; l++)
//            {
//                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl.vtx_hrt, 3, NULL, nv, NULL, 0, NULL, NULL);
//            }//l
//            
//            //torso jacobi
//            for(int l=0; l<100; l++)
//            {
//                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl.vtx_trs, 3, NULL, nv, NULL, 0, NULL, NULL);
//            }//l
//            
//        }//k
//        
//    }//t
    
    //clean
    ocl_fin(&ocl);
    
    printf("done.\n");
    
    return 0;
}
