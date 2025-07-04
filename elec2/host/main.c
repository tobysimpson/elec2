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
#include <sys/stat.h>
#include <time.h>

#include "ocl.h"
#include "msh.h"
#include "mg.h"
#include "io.h"


//electrophysiology FVM
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //create folders
    mkdir("/Users/toby/Downloads/raw", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir("/Users/toby/Downloads/xmf", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    //timer
    struct timespec t0;
    struct timespec t1;
    
    clock_gettime(CLOCK_REALTIME, &t0);
    
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
    msh.le = (cl_int3){8,8,8};
    msh.dx = 128.0f*powf(2.0f, -msh.le.x);
    msh.dt = 0.5f;
    msh_ini(&msh);
    
    //multigrid
    struct mg_obj mg;
    mg.nl =  msh.le.x/2;
    mg_ini(&ocl, &mg, &msh);
    

    /*
     ====================
     spheres
     ====================
     */
    
    //spheres
    int ns = 150;
    cl_float4 ss_hst[ns];
    srand((unsigned int)time(NULL));
    for(int i=0; i<ns; i++)
    {
        cl_float4 s;
        
        s.x = 0.8f*msh.dx*(rand()%msh.ne.x - msh.ne.x/2);
        s.y = 0.8f*msh.dx*(rand()%msh.ne.y - msh.ne.y/2);
        s.z = 0.8f*msh.dx*(rand()%msh.ne.z - msh.ne.z/2);
        s.w = 0.2f;
        
        ss_hst[i] = s;
    }
    
    /*
     ====================
     memory, kernels
     ====================
     */
    
    //memory
    cl_mem ww = clCreateBuffer(ocl.context, CL_MEM_HOST_NO_ACCESS , msh.nv_tot*sizeof(cl_float), NULL, &ocl.err);
    cl_mem ss = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, ns*sizeof(cl_float4), ss_hst, NULL);
    
    //kernel
    cl_kernel ele_ini = clCreateKernel(ocl.program, "ele_ini", &ocl.err);
    cl_kernel ele_ion = clCreateKernel(ocl.program, "ele_ion", &ocl.err);
    
    //fine
    struct lvl_obj lf = mg.lvls[0];
    
    //args
    ocl.err = clSetKernelArg(ele_ini,  0, sizeof(struct msh_obj),    (void*)&lf.msh);
    ocl.err = clSetKernelArg(ele_ini,  1, sizeof(cl_mem),            (void*)&lf.uu);
    ocl.err = clSetKernelArg(ele_ini,  2, sizeof(cl_mem),            (void*)&ww);
    ocl.err = clSetKernelArg(ele_ini,  3, sizeof(cl_mem),            (void*)&lf.gg);
    
    ocl.err = clSetKernelArg(ele_ion,  0, sizeof(struct msh_obj),    (void*)&lf.msh);
    ocl.err = clSetKernelArg(ele_ion,  1, sizeof(cl_mem),            (void*)&lf.uu);
    ocl.err = clSetKernelArg(ele_ion,  2, sizeof(cl_mem),            (void*)&ww);
    ocl.err = clSetKernelArg(ele_ion,  3, sizeof(cl_mem),            (void*)&lf.gg);
    
    
    /*
     ====================
     init
     ====================
     */
    
    //geom levels
    for(int l=0; l<mg.nl; l++)
    {
        struct lvl_obj lvl = mg.lvls[l];
        
        //geom
        mg_geo(&ocl, &mg, &lvl, &ss);
        
        //write
//        wrt_xmf(&ocl, &lvl.msh, 0);
//        wrt_flt4(&ocl, &lvl.msh, &lvl.gg, "gg", 0, lvl.msh.ne_tot);
//        wrt_flt1(&ocl, &lvl.msh, &lvl.uu, "uu", 0, lvl.msh.ne_tot);
    }
    
    //init
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ele_ini, 3, NULL, (size_t*)&msh.ne_sz, NULL, 0, NULL, &ocl.event);
    
    
    /*
     ====================
     calc
     ====================
     */
    

    
    //loop
    for(int frm=0; frm<100; frm++)
    {
        if((frm % 10)==0)
        {
            printf("%03d\n",frm);
        }
        
        //write
        wrt_xmf(&ocl, &lf.msh, frm);
        wrt_flt1(&ocl, &lf.msh, &lf.uu, "uu", frm, lf.msh.ne_tot);
        wrt_flt4(&ocl, &lf.msh, &lf.gg, "gg", frm, lf.msh.ne_tot);
//        wrt_flt1(&ocl, &lf.msh, &lf.uu, "uu", 0, lf.msh.ne_tot);        //overwrite files
//        wrt_flt4(&ocl, &lf.msh, &lf.gg, "gg", 0, lf.msh.ne_tot);


        //time per frame
        for(int t=0; t<10; t++)
        {
            //euler rhs
            ocl.err = clEnqueueCopyBuffer(ocl.command_queue, lf.uu, lf.bb, 0, 0, msh.ne_tot*sizeof(cl_float), 0, NULL, &ocl.event);
            
            //euler mg (nl,nj,nc)
            mg_cyc(&ocl, &mg, &mg.ops[1], mg.nl, 5, 5);
            
            //membrane
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ele_ion, 3, NULL, msh.ne_sz, NULL, 0, NULL, &ocl.event);
            
        }//t
        
        //ecg (nl,nj,nc)
        mg_cyc(&ocl, &mg, &mg.ops[0], mg.nl, 5, 5);
        
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
    
    
    clock_gettime(CLOCK_REALTIME, &t1);

    printf("%10d %f\n", msh.nv_tot, (1e9f*(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec))*1e-9);
    
    printf("done\n");
    
    return 0;
}
