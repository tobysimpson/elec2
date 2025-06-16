//
//  mg.c
//  fsi2
//
//  Created by Toby Simpson on 14.04.2025.
//
#include <time.h>
#include <math.h>
#include "mg.h"


//init
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg, struct msh_obj *msh)
{
    printf("mg %d\n",mg->nl);
    
    //levels
    mg->lvls = malloc(mg->nl*sizeof(struct lvl_obj));
    
    //levels
    for(int l=0; l<mg->nl; l++)
    {
        //instance
        struct lvl_obj *lvl = &mg->lvls[l];
        
        //dims
        lvl->msh.le = (cl_int3){msh->le.x-l, msh->le.y-l, msh->le.z-l};
        
        //dx
        lvl->msh.dx = msh->dx*powf(2e0f,l);
        lvl->msh.dt = msh->dt;
        
        //mesh
        msh_ini(&lvl->msh);
        
        //memory
        lvl->uu = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float), NULL, &ocl->err);
        lvl->bb = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float), NULL, &ocl->err);
        lvl->rr = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float), NULL, &ocl->err);
        lvl->gg = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.ne_tot*sizeof(cl_float4), NULL, &ocl->err);
    }
    
    //geo
    mg->ele_geo = clCreateKernel(ocl->program, "ele_geo", &ocl->err);
    
    //trans
    mg->ele_prj = clCreateKernel(ocl->program, "ele_prj", &ocl->err);
    mg->ele_itp = clCreateKernel(ocl->program, "ele_itp", &ocl->err);
    

    //ecg poisson
    mg->ops[0].ele_res = clCreateKernel(ocl->program, "ele_res0", &ocl->err);
    mg->ops[0].ele_jac = clCreateKernel(ocl->program, "ele_jac0", &ocl->err);
    
    //elec euler
    mg->ops[1].ele_res = clCreateKernel(ocl->program, "ele_res1", &ocl->err);
    mg->ops[1].ele_jac = clCreateKernel(ocl->program, "ele_jac1", &ocl->err);
    
    return;
}


//geometry
void mg_geo(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lvl, cl_mem *ss)
{
    //args
    ocl->err = clSetKernelArg(mg->ele_geo,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(mg->ele_geo,  1, sizeof(cl_mem),            (void*)&lvl->gg);
    ocl->err = clSetKernelArg(mg->ele_geo,  2, sizeof(cl_mem),            (void*)ss);
    
    //geo
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_geo, 3, NULL, lvl->msh.ne_sz, NULL, 0, NULL, &ocl->event);

    return;
}


//residual
void mg_res(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl)
{
    //args
    ocl->err = clSetKernelArg(op->ele_res,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->ele_res,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->ele_res,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(op->ele_res,  3, sizeof(cl_mem),            (void*)&lvl->rr);
    ocl->err = clSetKernelArg(op->ele_res,  4, sizeof(cl_mem),            (void*)&lvl->gg);
    
    //residual
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->ele_res, 3, NULL, lvl->msh.ne_sz, NULL, 0, NULL, NULL);
    
    return;
}


//jacobi
void mg_jac(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, struct lvl_obj *lvl, int nj)
{
    ocl->err = clSetKernelArg(op->ele_jac,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);
    ocl->err = clSetKernelArg(op->ele_jac,  1, sizeof(cl_mem),            (void*)&lvl->uu);
    ocl->err = clSetKernelArg(op->ele_jac,  2, sizeof(cl_mem),            (void*)&lvl->bb);
    ocl->err = clSetKernelArg(op->ele_jac,  3, sizeof(cl_mem),            (void*)&lvl->gg);
    
    //smooth
    for(int j=0; j<nj; j++)
    {
        ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->ele_jac, 3, NULL, lvl->msh.ne_sz, NULL, 0, NULL, NULL);
    }

    return;
}


//interp
void mg_itp(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->ele_itp,  0, sizeof(struct msh_obj),    (void*)&lf->msh);     //fine
    ocl->err = clSetKernelArg(mg->ele_itp,  1, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
    ocl->err = clSetKernelArg(mg->ele_itp,  2, sizeof(cl_mem),            (void*)&lf->uu);      //fine
    
    //interp
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_itp, 3, NULL, lf->msh.ne_sz, NULL, 0, NULL, NULL);
    
    return;
}


//project
void mg_prj(struct ocl_obj *ocl, struct mg_obj *mg, struct lvl_obj *lf, struct lvl_obj *lc)
{
    //args
    ocl->err = clSetKernelArg(mg->ele_prj,  0, sizeof(struct msh_obj),    (void*)&lc->msh);     //coarse
    ocl->err = clSetKernelArg(mg->ele_prj,  1, sizeof(cl_mem),            (void*)&lf->rr);      //fine
    ocl->err = clSetKernelArg(mg->ele_prj,  2, sizeof(cl_mem),            (void*)&lc->uu);      //coarse
    ocl->err = clSetKernelArg(mg->ele_prj,  3, sizeof(cl_mem),            (void*)&lc->bb);      //coarse
    
    //project
    ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->ele_prj, 3, NULL, lc->msh.ne_sz, NULL, 0, NULL, NULL);
    
    return;
}



//v-cycles
void mg_cyc(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op, int nl, int nj, int nc)
{
    //cycle
    for(int c=0; c<nc; c++)
    {
        //descend
        for(int l=0; l<(nl-1); l++)
        {
            //levels
            struct lvl_obj *lf = &mg->lvls[l];
            struct lvl_obj *lc = &mg->lvls[l+1];
            
            //pre
            mg_jac(ocl, mg, op, lf, nj);
            
            //post
            mg_res(ocl, mg, op, lf);
            
            //prj
            mg_prj(ocl, mg, lf, lc);
            
        } //dsc
        
        //coarse
        mg_jac(ocl, mg, op, &mg->lvls[nl-1], nj);
        
        //ascend
        for(int l=(nl-2); l>=0; l--)
        {
            //levels
            struct lvl_obj *lf = &mg->lvls[l];
            struct lvl_obj *lc = &mg->lvls[l+1];
            
            //itp
            mg_itp(ocl, mg, lf, lc);
           
            //post
            mg_jac(ocl, mg, op, lf, nj);
            
        } //dsc
        
    } //cycle
    
    return;
}


//final
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg)
{
    ocl->err = clReleaseKernel(mg->ele_geo);
    ocl->err = clReleaseKernel(mg->ele_prj);
    ocl->err = clReleaseKernel(mg->ele_itp);
    
    ocl->err = clReleaseKernel(mg->ops[0].ele_res);
    ocl->err = clReleaseKernel(mg->ops[0].ele_jac);
    
    ocl->err = clReleaseKernel(mg->ops[1].ele_res);
    ocl->err = clReleaseKernel(mg->ops[1].ele_jac);

    //levels
    for(int l=0; l<mg->nl; l++)
    {
        //device
        ocl->err = clReleaseMemObject(mg->lvls[l].uu);
        ocl->err = clReleaseMemObject(mg->lvls[l].bb);
        ocl->err = clReleaseMemObject(mg->lvls[l].rr);
        ocl->err = clReleaseMemObject(mg->lvls[l].gg);
    }
    
    //mem
    free(mg->lvls);

    return;
}
