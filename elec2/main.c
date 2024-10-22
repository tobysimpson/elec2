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
        ocl.err = clSetKernelArg(ocl.vtx_ini,  1, sizeof(cl_mem),           (void*)&lvl->gg);
        ocl.err = clSetKernelArg(ocl.vtx_ini,  2, sizeof(cl_mem),           (void*)&lvl->uu);
           
        //init
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_ini, 3, NULL, (size_t*)&lvl->nv, NULL, 0, NULL, NULL);
        
        //write
        wrt_vtk(lvl, &ocl, 0);
    }
    
    /*
     ==============================
     cycle
     ==============================
     */
    
    
    /*
    
    int nc = 1;     //cycles
    int nj = 10;    //jacobi iterations
    
    //cycle
    for(int cyc_idx=0; cyc_idx<nc; cyc_idx++)
    {
        //top
        struct lvl_obj *lvl = &mg.lvls[0];

        //dims
//        size_t nv[3] = {lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z};
        size_t iv[3] = {lvl->msh.nv.x-2, lvl->msh.nv.y-2, lvl->msh.nv.z-2};
        size_t ne[3] = {lvl->msh.ne.x, lvl->msh.ne.y, lvl->msh.ne.z};
        
        //jacobi iter
        for(int j=0; j<nj; j++)
        {
            //solve
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_res, 3, NULL, iv, NULL, 0, NULL, NULL);
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_jac, 3, NULL, iv, NULL, 0, NULL, NULL);
        }
        
        //residual
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_res, 3, NULL, iv, NULL, 0, NULL, NULL);
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->ele_res, 3, NULL, ne, NULL, 0, NULL, NULL);
        float r = red_sum(lvl->msh.ne_tot, &lvl->ee, &red, &ocl);
        
        //error
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->ele_err, 3, NULL, ne, NULL, 0, NULL, NULL);
        float e = red_sum(lvl->msh.ne_tot, &lvl->ee, &red, &ocl);
        

        printf("top %2d %3d %e %e\n", 0, lvl->msh.ne.x, sqrtf(r), sqrtf(e));
        
        //descend
        for(int lvl_idx=1; lvl_idx<mg.nl; lvl_idx++)
        {
            //level
            struct lvl_obj *lvl = &mg.lvls[lvl_idx];

            //dims
            size_t nv[3] = {lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z};
            size_t iv[3] = {lvl->msh.nv.x-2, lvl->msh.nv.y-2, lvl->msh.nv.z-2};
//            size_t ne[3] = {lvl->msh.ne.x, lvl->msh.ne.y, lvl->msh.ne.z};
            
            //project
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_prj, 3, NULL, nv, NULL, 0, NULL, NULL);
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_rst, 3, NULL, nv, NULL, 0, NULL, NULL);

            //jacobi iter
            for(int j=0; j<nj; j++)
            {
                //solve
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_res, 3, NULL, iv, NULL, 0, NULL, NULL);
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_jac, 3, NULL, iv, NULL, 0, NULL, NULL);
            }
            
            //residual
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_res, 3, NULL, iv, NULL, 0, NULL, NULL);
            
//            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->ele_res, 3, NULL, ne, NULL, 0, NULL, NULL);
//            float r = red_sum(lvl->msh.ne_tot, &lvl->ee, &red, &ocl);
//
//            //error
//            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->ele_err, 3, NULL, ne, NULL, 0, NULL, NULL);
//            float e = red_sum(lvl->msh.ne_tot, &lvl->ee, &red, &ocl);
            
//            printf("dsc %2d %3d %e %e\n", lvl_idx, lvl->msh.ne.x, sqrtf(r), sqrtf(e));
            
        } //dsc
        
        
        //ascend
        for(int lvl_idx=(mg.nl-2); lvl_idx>=0; lvl_idx--)
        {
            //level
            struct lvl_obj *lvl = &mg.lvls[lvl_idx];
            
            //dims
            size_t nv[3] = {lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z};
            size_t iv[3] = {lvl->msh.nv.x-2, lvl->msh.nv.y-2, lvl->msh.nv.z-2};
//            size_t ne[3] = {lvl->msh.ne.x, lvl->msh.ne.y, lvl->msh.ne.z};
            
            //interp
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_itp, 3, NULL, nv, NULL, 0, NULL, NULL);
            
            //jacobi iter
            for(int j=0; j<nj; j++)
            {
                //solve
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_res, 3, NULL, iv, NULL, 0, NULL, NULL);
                ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_jac, 3, NULL, iv, NULL, 0, NULL, NULL);
            }
            
            //residual
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->vtx_res, 3, NULL, iv, NULL, 0, NULL, NULL);
            
            
//            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->ele_res, 3, NULL, ne, NULL, 0, NULL, NULL);
//            float r = red_sum(lvl->msh.ne_tot, &lvl->ee, &red, &ocl);
//
//            //error
//            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, lvl->ele_err, 3, NULL, ne, NULL, 0, NULL, NULL);
//            float e = red_sum(lvl->msh.ne_tot, &lvl->ee, &red, &ocl);
            
//            printf("asc %2d %3d %e %e\n", lvl_idx, lvl->msh.ne.x, sqrtf(r), sqrtf(e));
            
        } //asc
        
    } //cyc_idx
    
     
     */
     
    
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
