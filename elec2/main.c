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
    
    for(int l=0; l<mg.nl; l++)
    {
        //obj
        struct lvl_obj *lvl = &mg.lvls[l];
        
        //args
        ocl.err = clSetKernelArg(ocl.vtx_ini,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
        ocl.err = clSetKernelArg(ocl.vtx_ini,  1, sizeof(cl_mem),           (void*)&lvl->gg);
        ocl.err = clSetKernelArg(ocl.vtx_ini,  2, sizeof(cl_mem),           (void*)&lvl->uu);
           
        //init
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_ini, 3, NULL, (size_t*)&lvl->nv, NULL, 0, NULL, NULL);
        
        //write
//        wrt_vtk(lvl, &ocl, 0);
    }
    
    /*
     ==============================
     cycle
     ==============================
     */
    
    

    int nc = 1;    //cycles
    int nj = 1;    //jacobi iterations
    int nf = 100;   //frames
    
    //frames
    for(int f=0; f<nf; f++)
    {
        printf("%d\n", f);
        
        //write
        wrt_vtk(&mg.lvls[0], &ocl, f);
        
        //cycle
        for(int c=0; c<nc; c++)
        {
            //top
            struct lvl_obj *lvl = &mg.lvls[0];
            
            //args
            ocl.err = clSetKernelArg(ocl.vtx_jac,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
            ocl.err = clSetKernelArg(ocl.vtx_jac,  1, sizeof(cl_mem),           (void*)&lvl->uu);
            
            //args
            ocl.err = clSetKernelArg(ocl.vtx_res,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
            ocl.err = clSetKernelArg(ocl.vtx_res,  1, sizeof(cl_mem),           (void*)&lvl->uu);
            
            //jacobi
            for(int j=0; j<nj; j++)
            {
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_jac, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
            }
            
            //residual
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_res, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
            
            
            //descend
            for(int l=1; l<mg.nl; l++)
            {
                //level
                struct lvl_obj *lvl = &mg.lvls[l];
                
                //args
                ocl.err = clSetKernelArg(ocl.vtx_prj,  0, sizeof(struct msh_obj),   (void*)&mg.lvls[l+1].msh);
                ocl.err = clSetKernelArg(ocl.vtx_prj,  1, sizeof(cl_mem),           (void*)&mg.lvls[l+1].uu);
                ocl.err = clSetKernelArg(ocl.vtx_prj,  2, sizeof(cl_mem),           (void*)&lvl->uu);
                
                //args
                ocl.err = clSetKernelArg(ocl.vtx_rst,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
                ocl.err = clSetKernelArg(ocl.vtx_rst,  1, sizeof(cl_mem),           (void*)&lvl->uu);
                
                //args
                ocl.err = clSetKernelArg(ocl.vtx_jac,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
                ocl.err = clSetKernelArg(ocl.vtx_jac,  1, sizeof(cl_mem),           (void*)&lvl->uu);
                
                //args
                ocl.err = clSetKernelArg(ocl.vtx_res,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
                ocl.err = clSetKernelArg(ocl.vtx_res,  1, sizeof(cl_mem),           (void*)&lvl->uu);

                
                //project
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_prj, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_rst, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
                
                //jacobi
                for(int j=0; j<nj; j++)
                {
                    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_jac, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
                }
                
                //residual
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_res, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
                
            } //dsc
            

             //ascend
             for(int l=(mg.nl-2); l>=0; l--)
             {
                 //level
                 struct lvl_obj *lvl = &mg.lvls[l];
                 
                 //args
                 ocl.err = clSetKernelArg(ocl.vtx_itp,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
                 ocl.err = clSetKernelArg(ocl.vtx_itp,  1, sizeof(cl_mem),           (void*)&mg.lvls[l+1].uu);
                 ocl.err = clSetKernelArg(ocl.vtx_itp,  2, sizeof(cl_mem),           (void*)&lvl->uu);
                 
                 //args
                 ocl.err = clSetKernelArg(ocl.vtx_rst,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
                 ocl.err = clSetKernelArg(ocl.vtx_rst,  1, sizeof(cl_mem),           (void*)&lvl->uu);
                 
                 //args
                 ocl.err = clSetKernelArg(ocl.vtx_jac,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
                 ocl.err = clSetKernelArg(ocl.vtx_jac,  1, sizeof(cl_mem),           (void*)&lvl->uu);
                 
                 //args
                 ocl.err = clSetKernelArg(ocl.vtx_res,  0, sizeof(struct msh_obj),   (void*)&lvl->msh);
                 ocl.err = clSetKernelArg(ocl.vtx_res,  1, sizeof(cl_mem),           (void*)&lvl->uu);
                 
                 //interp
                 ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_itp, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
                 
                 //jacobi iter
                 for(int j=0; j<nj; j++)
                 {
                     ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_jac, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
                 }
                 
                 //residual
                 ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_res, 3, NULL, lvl->nv, NULL, 0, NULL, NULL);
                 
             }//l
             
        } //c
        
    }//t
    
    //clean
    ocl_fin(&ocl);
    
    printf("done.\n");
    
    return 0;
}
