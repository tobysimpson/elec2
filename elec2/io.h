//
//  io.h
//  mg1
//
//  Created by toby on 29.05.24.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef io_h
#define io_h

#define ROOT_WRITE  "/Users/toby/Downloads/vtk"


//write
void wrt_vtk(struct lvl_obj *lvl, struct ocl_obj *ocl, int frm_idx)
{
    FILE* file1;
    char file1_name[250];
    
    //file name
    sprintf(file1_name, "%s/%s.%02d.%03d.vtk", ROOT_WRITE, "grid", lvl->idx, frm_idx);
    
    //open
    file1 = fopen(file1_name,"wb");
    
    //write
    fprintf(file1,"# vtk DataFile Version 3.0\n");
    fprintf(file1,"grid1\n");
    fprintf(file1,"ASCII\n");
    fprintf(file1,"DATASET STRUCTURED_POINTS\n");
    fprintf(file1,"DIMENSIONS %d %d %d\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
    fprintf(file1,"ORIGIN %e %e %e\n", -lvl->msh.dx*lvl->msh.ne.x/2, -lvl->msh.dx*lvl->msh.ne.y/2, -lvl->msh.dx*lvl->msh.ne.z/2);
    fprintf(file1,"SPACING %e %e %e\n", lvl->msh.dx, lvl->msh.dx, lvl->msh.dx);

    
    /*
     ===================
     coords
     ===================
     */
    
//    fprintf(file1,"\nPOINTS %zu float\n", lvl->nv_tot);
//    
//    cl_int3 nv2 = {lvl->msh.nv.x/2, lvl->msh.nv.y/2, lvl->msh.nv.z/2}; //origin at centre
//    
//    for(int k=0; k<lvl->msh.nv.z; k++)
//    {
//        for(int j=0; j<lvl->msh.nv.y; j++)
//        {
//            for(int i=0; i<lvl->msh.nv.x; i++)
//            {
//                fprintf(file1, "%e %e %e\n", lvl->msh.dx*(i - nv2.x), lvl->msh.dx*(j - nv2.y), lvl->msh.dx*(k - nv2.z));
//            }
//        }
//    }

    //point data flag
    fprintf(file1,"\nPOINT_DATA %zu\n", lvl->nv_tot);
    
//    fprintf(file1,"VECTORS xx float\n");
//    //map
//    cl_float3 *vv = clEnqueueMapBuffer(ocl->command_queue, lvl->xx, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(cl_float3), 0, NULL, NULL, &ocl->err);
//    //write
//    for(int i=0; i<lvl->msh.nv_tot; i++)
//    {
//        fprintf(file1, "%e %e %e\n", xx[i].x, xx[i].y, xx[i].z);
//    }
//    //unmap
//    clEnqueueUnmapMemObject(ocl->command_queue, lvl->xx, vv, 0, NULL, NULL);
    
    //gg
    fprintf(file1,"SCALARS gg float 4\n");
    fprintf(file1,"LOOKUP_TABLE default\n");
    //map
    cl_float4 *gg = clEnqueueMapBuffer(ocl->command_queue, lvl->gg, CL_TRUE, CL_MAP_READ, 0, lvl->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    //write
    for(int i=0; i<lvl->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e %e\n", gg[i].x, gg[i].y, gg[i].z, gg[i].w);
    }
    //unmap
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->gg, gg, 0, NULL, NULL);
    
    //uu
    fprintf(file1,"SCALARS uu float 4\n");
    fprintf(file1,"LOOKUP_TABLE default\n");
    //map
    cl_float4 *uu = clEnqueueMapBuffer(ocl->command_queue, lvl->uu, CL_TRUE, CL_MAP_READ, 0, lvl->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    //write
    for(size_t i=0; i<lvl->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e %e\n", uu[i].x, uu[i].y, uu[i].z, uu[i].w);
    }
    //unmap
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->uu, uu, 0, NULL, NULL);

    //clean up
    fclose(file1);
    
    return;
}



#endif /* io_h */
