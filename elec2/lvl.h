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
    int             le;     //log2(ne)
    struct  msh_obj msh;
    
    //memory
    cl_mem          xx;
    cl_mem          uu;
    
    //kernels
    cl_kernel       vtx_ini;
    cl_kernel       vtx_ion;
    cl_kernel       vtx_hrt;
    cl_kernel       vtx_trs;
};


void lvl_ini(struct lvl_obj *lvl, struct ocl_obj *ocl)
{
    printf("le %d\n", lvl->le);
    
    //mesh
    int ne = pow(2,lvl->le);
    int nv = ne+1;
    
    lvl->msh.dx     = 0.5f;
    lvl->msh.dt     = 0.05f;
    
    lvl->msh.ne     = (cl_int3){ne,ne,ne};
    lvl->msh.nv     = (cl_int3){nv,nv,nv};
    
    lvl->msh.ne_tot = ne*ne*ne;
    lvl->msh.nv_tot = nv*nv*nv;
    
    msh_ini(&lvl->msh);
    
    //memory
    lvl->xx = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    lvl->uu = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    
    //kernels
    lvl->vtx_ini = clCreateKernel(ocl->program, "vtx_ini", &ocl->err);
    lvl->vtx_ion = clCreateKernel(ocl->program, "vtx_ion", &ocl->err);
    lvl->vtx_hrt = clCreateKernel(ocl->program, "vtx_hrt", &ocl->err);
    lvl->vtx_trs = clCreateKernel(ocl->program, "vtx_trs", &ocl->err);
    
    //arguments
    ocl->err = clSetKernelArg(lvl->vtx_ini,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_ini,  1, sizeof(cl_mem),            (void*)&lvl->xx);
    ocl->err = clSetKernelArg(lvl->vtx_ini,  2, sizeof(cl_mem),            (void*)&lvl->uu);
    
    ocl->err = clSetKernelArg(lvl->vtx_ion,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_ion,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    
    ocl->err = clSetKernelArg(lvl->vtx_hrt,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_hrt,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    
    ocl->err = clSetKernelArg(lvl->vtx_trs,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_trs,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    
    return;
}


void lvl_fin(struct lvl_obj *lvl, struct ocl_obj *ocl)
{
    //kernels
    ocl->err = clReleaseKernel(lvl->vtx_ini);
    ocl->err = clReleaseKernel(lvl->vtx_ion);
    ocl->err = clReleaseKernel(lvl->vtx_hrt);
    ocl->err = clReleaseKernel(lvl->vtx_trs);
    
    //memory
    ocl->err = clReleaseMemObject(lvl->xx);
    ocl->err = clReleaseMemObject(lvl->uu);
    
    return;
}



#endif /* lvl_h */
