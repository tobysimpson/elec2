//
//  mg.c
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "mg.h"


void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg)
{
    //allocate
    mg->lvls = malloc(mg->nl*sizeof(struct lvl_obj));
    
    //levels
    for(int l=0; l<mg->nl; l++)
    {
        struct lvl_obj *lvl = &mg->lvls[l];
        
        //params
        lvl->idx = l;
        lvl->msh.dx = mg->dx*pow(2,l);
        lvl->msh.dt = mg->dt;
        lvl->msh.le = (cl_uint3){mg->le.x-l, mg->le.y-l, mg->le.z-l};
        
        //init
        lvl_ini(ocl, lvl);
    }
    
    //transfer ops
    for(int l=0; l<mg->nl; l++)
    {
        struct lvl_obj *lvl = &mg->lvls[l];
        
        //project (skip finest)
        if(l>0)
        {
            //arguments
            ocl->err = clSetKernelArg(lvl->vtx_prj,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);                   //coarse
            ocl->err = clSetKernelArg(lvl->vtx_prj,  1, sizeof(cl_mem),            (void*)&lvl->bb);                    //coarse
            ocl->err = clSetKernelArg(lvl->vtx_prj,  2, sizeof(cl_mem),            (void*)&mg->lvls[l-1].rr);           //fine
        }
        
        //interpolate (skip coarsest)
        if(l<(mg->nl-1))
        {
            //arguments
            ocl->err = clSetKernelArg(lvl->vtx_itp,  0, sizeof(struct msh_obj),    (void*)&lvl->msh);                   //fine
            ocl->err = clSetKernelArg(lvl->vtx_itp,  1, sizeof(cl_mem),            (void*)&mg->lvls[l+1].uu);           //coarse
            ocl->err = clSetKernelArg(lvl->vtx_itp,  2, sizeof(cl_mem),            (void*)&lvl->uu);                    //fine
        }
    }
    
    //membrane
    
    
    return;
}


void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg)
{
    //levels
    for(int l=0; l<mg->nl; l++)
    {
        struct lvl_obj *lvl = &mg->lvls[l];
        
        //init
        lvl_fin(ocl, lvl);
    }
    
    //deallocate
    free(mg->lvls);
    
    return;
}


void mg_slv(struct ocl_obj *ocl, struct mg_obj *mg)
{
    //cycle
    for(int c=0; c<mg->nc; c++)
    {
//        printf("c %d\n", c);
        
        //descend
        for(int l=0; l<mg->nl; l++)
        {
//            printf("l %d\n", l);
            
            //level
            struct lvl_obj *lvl = &mg->lvls[l];

            //skip top
            if(l>0)
            {
                //project
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_prj, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_zro, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
            }
            
            //jacobi iter
            for(int j=0; j<mg->nj; j++)
            {
                //solve
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_rsd, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_jac, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
            }
            
            //residual
            ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_rsd, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
            
        } //l
        
        
        //ascend
        for(int l=(mg->nl-2); l>=0; l--)
        {
            //level
            struct lvl_obj *lvl = &mg->lvls[l];
            
            //interp
            ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_itp, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
            
            //jacobi iter
            for(int j=0; j<mg->nj; j++)
            {
                //solve
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_rsd, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_jac, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
            }
            
            //residual
            ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, lvl->vtx_rsd, 3, NULL, (size_t*)&lvl->msh.nv, NULL, 0, NULL, NULL);
            
        } //l
    } //c
}



