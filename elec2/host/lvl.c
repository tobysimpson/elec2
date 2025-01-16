//
//  lvl.c
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "lvl.h"

//init
void lvl_ini(struct ocl_obj *ocl, struct lvl_obj *lvl)
{
//    printf("lvl %d\n", );
//    printf("dx %f\n", lvl->msh.dx);
//    printf("le %u,%u,%u \n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z);
    
    //mesh
    msh_ini(&lvl->msh);
    
    printf("lvl %d %f %02d%02d%02d [%3llu %3llu %3llu] %10llu\n", lvl->idx, lvl->msh.dx, lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, lvl->msh.ne.x, lvl->msh.ne.y, lvl->msh.ne.z, lvl->msh.nv_tot);
    
    //memory
    lvl->uu = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(float), NULL, &ocl->err);
    lvl->bb = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(float), NULL, &ocl->err);
    lvl->rr = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(float), NULL, &ocl->err);
    lvl->aa = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(float), NULL, &ocl->err);

    //kernels
    lvl->vtx_ini = clCreateKernel(ocl->program, "vtx_ini", &ocl->err);
    
    lvl->vtx_zro = clCreateKernel(ocl->program, "vtx_zro", &ocl->err);
    lvl->vtx_prj = clCreateKernel(ocl->program, "vtx_prj", &ocl->err);
    lvl->vtx_itp = clCreateKernel(ocl->program, "vtx_itp", &ocl->err);
    
    lvl->vtx_rsd = clCreateKernel(ocl->program, "vtx_rsd", &ocl->err);
    lvl->vtx_jac = clCreateKernel(ocl->program, "vtx_jac", &ocl->err);
    
    lvl->vtx_ion = clCreateKernel(ocl->program, "vtx_ion", &ocl->err);

    //arguments
    ocl->err = clSetKernelArg(lvl->vtx_ini,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_ini,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(lvl->vtx_ini,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(lvl->vtx_ini,  3, sizeof(cl_mem),            (void*)&lvl->rr);
    ocl->err = clSetKernelArg(lvl->vtx_ini,  4, sizeof(cl_mem),            (void*)&lvl->aa);
    
    ocl->err = clSetKernelArg(lvl->vtx_zro,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_zro,  1, sizeof(cl_mem),            (void*)&lvl->uu);

    ocl->err = clSetKernelArg(lvl->vtx_rsd,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_rsd,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(lvl->vtx_rsd,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(lvl->vtx_rsd,  3, sizeof(cl_mem),            (void*)&lvl->rr);
    
    ocl->err = clSetKernelArg(lvl->vtx_jac,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_jac,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(lvl->vtx_jac,  2, sizeof(cl_mem),            (void*)&lvl->rr);
    
    ocl->err = clSetKernelArg(lvl->vtx_ion,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(lvl->vtx_ion,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(lvl->vtx_ion,  2, sizeof(cl_mem),            (void*)&lvl->aa);

    return;
}


//final
void lvl_fin(struct ocl_obj *ocl, struct lvl_obj *lvl)
{
    //kernels
    ocl->err = clReleaseKernel(lvl->vtx_ini);
    
    ocl->err = clReleaseKernel(lvl->vtx_zro);
    ocl->err = clReleaseKernel(lvl->vtx_prj);
    ocl->err = clReleaseKernel(lvl->vtx_itp);

    ocl->err = clReleaseKernel(lvl->vtx_rsd);
    ocl->err = clReleaseKernel(lvl->vtx_jac);
    
    ocl->err = clReleaseKernel(lvl->vtx_ion);

    //memory
    ocl->err = clReleaseMemObject(lvl->uu);
    ocl->err = clReleaseMemObject(lvl->bb);
    ocl->err = clReleaseMemObject(lvl->rr);
    ocl->err = clReleaseMemObject(lvl->aa);
    
    return;
}
