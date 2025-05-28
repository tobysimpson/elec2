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
    msh.le = (cl_int3){5,5,5};
    msh.dx = 10.0f*powf(2e0f, -msh.le.x); //[0,1]ˆ3
    msh.dt = 0.1f;
    msh_ini(&msh);
    
    //multigrid
    struct mg_obj mg;
    mg.nl = msh.le.x;
    mg.nj = 10;
    mg.nc = 10;
    mg_ini(&ocl, &mg, &msh);
    
    /*
     ====================
     init
     ====================
     */
    
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
        
        //cn rhs
//        mg_fwd(&ocl, &mg, &mg.ops[1], lvl);
        
        //jac
//        mg_jac(&ocl, &mg, &mg.ops[0], lvl);
        
        //norms
//        mg_nrm(&ocl, &mg, lvl);
        
    }
    
    
    //loop
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
        
        //rhs
        mg_fwd(&ocl, &mg, &mg.ops[1], &mg.lvls[0]);
        
        //jac
        mg_jac(&ocl, &mg, &mg.ops[1], &mg.lvls[0]);
        
//        //resid
//        mg_res(&ocl, &mg, &mg.ops[1], &mg.lvls[0]);
//        
//        //norms
//        mg_nrm(&ocl, &mg,  &mg.lvls[0]);
        
//        mg_cyc(&ocl, &mg, &mg.ops[1]);
     
        
    }//frm
    


    
    /*
     ====================
     final
     ====================
     */
    
    //flush
    ocl.err = clFlush(ocl.command_queue);
    ocl.err = clFinish(ocl.command_queue);
    
    //clean
    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    
    printf("done\n");
    
    return 0;
}
