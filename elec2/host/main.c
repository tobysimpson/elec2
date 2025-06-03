//
//  main.c
//  fsi2
//
//  Created by Toby Simpson on 06.08.2024.
//


//#ifdef __APPLE__
//#include <OpenCL/opencl.h>
//#else
//#include <CL/cl.h>
//#endif

#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

#include "ocl.h"
#include "msh.h"
#include "mg.h"
#include "io.h"


//multigrid benchmark - FVM by element
//not as good as vtx because of dirichlet conditions
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //create folders
    mkdir("/Users/toby/Downloads/raw", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir("/Users/toby/Downloads/xmf", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    /*
     ====================
     init
     ====================
     */
    
    //opencl
    struct ocl_obj ocl;
    ocl_ini(&ocl);
    
    //mesh
    struct msh_obj msh;
//    msh.x0 = (cl_float3){-1e0f,-1e0f,-1e0f};
//    msh.x1 = (cl_float3){+1e0f,+1e0f,+1e0f};
    msh.le = (cl_int3){7,7,7};
    msh.dx = 100.0f*powf(2e0f, -msh.le.x);
    msh.dt = 0.5f;
    msh_ini(&msh);
    
    //multigrid
    struct mg_obj mg;
    mg.nl = 2; //msh.le.x/2; //limit for geom
    mg.nj = 5;
    mg.nc = 5;
    mg_ini(&ocl, &mg, &msh);
    
    /*
     ====================
     init
     ====================
     */
    
    //memory
    cl_mem ww = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, msh.nv_tot*sizeof(cl_float4), NULL, &ocl.err);
    
    //kernel
    cl_kernel ele_ion = clCreateKernel(ocl.program, "ele_ion", &ocl.err);
    
    //args
    ocl.err = clSetKernelArg(ele_ion,  0, sizeof(struct msh_obj),    (void*)&msh);
    ocl.err = clSetKernelArg(ele_ion,  1, sizeof(cl_mem),            (void*)&mg.lvls[0].uu);
    ocl.err = clSetKernelArg(ele_ion,  2, sizeof(cl_mem),            (void*)&ww);
    
    //pattern for reset
    cl_float ptn = 1e0f;
    
    //reset
    ocl.err = clEnqueueFillBuffer(ocl.command_queue, ww, &ptn, sizeof(ptn), 0, msh.ne_tot*sizeof(ptn), 0, NULL, &ocl.event);
    
    //levels
    for(int l=0; l<1; l++)//mg.nl
    {
        //instance
        struct lvl_obj *lvl = &mg.lvls[l];
        
        //ini
        ocl.err = clSetKernelArg(mg.ele_ini,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
        ocl.err = clSetKernelArg(mg.ele_ini,  1, sizeof(cl_mem),            (void*)&lvl->uu);
        ocl.err = clSetKernelArg(mg.ele_ini,  2, sizeof(cl_mem),            (void*)&lvl->bb);
        ocl.err = clSetKernelArg(mg.ele_ini,  3, sizeof(cl_mem),            (void*)&lvl->rr);
        ocl.err = clSetKernelArg(mg.ele_ini,  4, sizeof(cl_mem),            (void*)&lvl->gg);
        
        //init
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, mg.ele_ini, 3, NULL, msh.ne_sz, NULL, 0, NULL, &ocl.event);
        
    }
    
    
    //frames
    for(int frm=0; frm<100; frm++)
    {
        if((frm % 10)==0)
        {
            printf("%03d\n",frm);
        }
        
        
        //write
        wrt_xmf(&ocl, &msh, frm);
        wrt_flt1(&ocl, &msh, &mg.lvls[0].uu, "uu", frm, msh.ne_tot);
        wrt_flt1(&ocl, &msh, &mg.lvls[0].bb, "bb", frm, msh.ne_tot);
        wrt_flt1(&ocl, &msh, &mg.lvls[0].rr, "rr", frm, msh.ne_tot);
        wrt_flt1(&ocl, &msh, &mg.lvls[0].gg, "gg", frm, msh.ne_tot);
    
        
        //time per frame
        for(int t=0; t<10; t++)
        {
            //cn rhs
            mg_fwd(&ocl, &mg, &mg.ops[1], &mg.lvls[0]);
            
            //cn jac
            mg_jac(&ocl, &mg, &mg.ops[1], &mg.lvls[0]);
            
            //ion
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ele_ion, 3, NULL, msh.ne_sz, NULL, 0, NULL, &ocl.event);
        }

        //ecg mg
        mg_cyc(&ocl, &mg, &mg.ops[0]);
     
    }//frm
    


    
    /*
     ====================
     final
     ====================
     */
    
    //flush
    ocl.err = clFlush(ocl.command_queue);
    ocl.err = clFinish(ocl.command_queue);
    
    //memory
    ocl.err = clReleaseMemObject(ww);
    
    //kernels
    ocl.err = clReleaseKernel(ele_ion);
    
    //clean
    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    
    printf("done\n");
    
    return 0;
}
