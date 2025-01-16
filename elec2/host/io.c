//
//  io.c
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "io.h"


//write xdmf
void wrt_xmf(struct ocl_obj *ocl, struct lvl_obj *lvl, int frm_idx)
{
    FILE* file1;
    char file1_name[250];
//    float* ptr1;
    
    //file name
    sprintf(file1_name, "%s/xmf/%s.%02d%02d%02d.%02d.xmf", ROOT_WRITE, "grid", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    
    //open
    file1 = fopen(file1_name,"w");
    
    fprintf(file1,"<Xdmf>\n");
    fprintf(file1,"  <Domain>\n");
    fprintf(file1,"    <Topology name=\"topo\" TopologyType=\"3DCoRectMesh\" Dimensions=\"%llu %llu %llu\"></Topology>\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
    fprintf(file1,"      <Geometry name=\"geo\" Type=\"ORIGIN_DXDYDZ\">\n");
    fprintf(file1,"        <!-- Origin -->\n");
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", -lvl->msh.dx*lvl->msh.ne2.x, -lvl->msh.dx*lvl->msh.ne2.y, -lvl->msh.dx*lvl->msh.ne2.z);
    fprintf(file1,"        <!-- DxDyDz -->\n");
    fprintf(file1,"        <DataItem Format=\"XML\" Dimensions=\"3\">%f %f %f</DataItem>\n", lvl->msh.dx, lvl->msh.dx, lvl->msh.dx);
    fprintf(file1,"      </Geometry>\n");
    fprintf(file1,"      <Grid Name=\"T1\" GridType=\"Uniform\">\n");
    fprintf(file1,"        <Topology Reference=\"/Xdmf/Domain/Topology[1]\"/>\n");
    fprintf(file1,"        <Geometry Reference=\"/Xdmf/Domain/Geometry[1]\"/>\n");
    
    fprintf(file1,"         <Attribute Name=\"uu\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%llu %llu %llu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
    fprintf(file1,"             /Users/toby/Downloads/raw/uu.%02d%02d%02d.%02d.raw\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
    fprintf(file1,"         <Attribute Name=\"bb\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%llu %llu %llu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
    fprintf(file1,"             /Users/toby/Downloads/raw/bb.%02d%02d%02d.%02d.raw\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
    fprintf(file1,"         <Attribute Name=\"rr\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%llu %llu %llu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
    fprintf(file1,"             /Users/toby/Downloads/raw/rr.%02d%02d%02d.%02d.raw\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
    fprintf(file1,"         <Attribute Name=\"aa\" Center=\"Node\" AttributeType=\"Scalar\">\n");
    fprintf(file1,"           <DataItem Format=\"Binary\" Dimensions=\"%llu %llu %llu 1\" Endian=\"Little\" Precision=\"4\" NumberType=\"Float\">\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
    fprintf(file1,"             /Users/toby/Downloads/raw/aa.%02d%02d%02d.%02d.raw\n", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    fprintf(file1,"           </DataItem>\n");
    fprintf(file1,"         </Attribute>\n");
    
    fprintf(file1,"    </Grid>\n");
    fprintf(file1," </Domain>\n");
    fprintf(file1,"</Xdmf>\n");
    
    //clean up
    fclose(file1);
}


//write raw
void wrt_raw(struct ocl_obj *ocl, struct lvl_obj *lvl, int frm_idx)
{
    FILE* file1;
    char file1_name[250];
    float* ptr1;
    
    //uu
    sprintf(file1_name, "%s/raw/%s.%02d%02d%02d.%02d.raw", ROOT_WRITE, "uu", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    file1 = fopen(file1_name,"wb");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->uu, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    fwrite(ptr1, sizeof(float), lvl->msh.nv_tot, file1);
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->uu, ptr1, 0, NULL, NULL);
    
    //bb
    sprintf(file1_name, "%s/raw/%s.%02d%02d%02d.%02d.raw", ROOT_WRITE, "bb", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    file1 = fopen(file1_name,"wb");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->bb, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    fwrite(ptr1, sizeof(float), lvl->msh.nv_tot, file1);
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->bb, ptr1, 0, NULL, NULL);
    
    //rr
    sprintf(file1_name, "%s/raw/%s.%02d%02d%02d.%02d.raw", ROOT_WRITE, "rr", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    file1 = fopen(file1_name,"wb");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->rr, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    fwrite(ptr1, sizeof(float), lvl->msh.nv_tot, file1);
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->rr, ptr1, 0, NULL, NULL);
    
    //aa
    sprintf(file1_name, "%s/raw/%s.%02d%02d%02d.%02d.raw", ROOT_WRITE, "aa", lvl->msh.le.x, lvl->msh.le.y, lvl->msh.le.z, frm_idx);
    file1 = fopen(file1_name,"wb");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->aa, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    fwrite(ptr1, sizeof(float), lvl->msh.nv_tot, file1);
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->aa, ptr1, 0, NULL, NULL);
    
    //clean up
    fclose(file1);
    
    return;
}


//vtk ascii
void vtk_asc(struct ocl_obj *ocl, struct lvl_obj *lvl, int frm_idx)
{
    FILE* file1;
    char file1_name[250];
    float* ptr1;
    
    //file name
    sprintf(file1_name, "%s/vtk/%s.%02d.%02d.vtk", ROOT_WRITE, "grid", lvl->idx, frm_idx);
    
    //open
    file1 = fopen(file1_name,"w");
    
    //write
    fprintf(file1,"# vtk DataFile Version 5.1\n");
    fprintf(file1,"grid1\n");
    fprintf(file1,"ASCII\n");
    fprintf(file1,"DATASET STRUCTURED_POINTS\n");
    fprintf(file1,"DIMENSIONS %llu %llu %llu\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
    fprintf(file1,"SPACING %f %f %f\n", lvl->msh.dx, lvl->msh.dx, lvl->msh.dx);
    fprintf(file1,"ORIGIN %f %f %f\n", -lvl->msh.dx*lvl->msh.ne2.x, -lvl->msh.dx*lvl->msh.ne2.y, -lvl->msh.dx*lvl->msh.ne2.z);
    
    //point data flag
    fprintf(file1,"POINT_DATA %llu\n", lvl->msh.nv_tot);
    
    //aa
    fprintf(file1,"SCALARS aa float\n");
    fprintf(file1,"LOOKUP_TABLE default\n");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->aa, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    for(int i=0; i<lvl->msh.nv_tot; i++)
    {
        fprintf(file1, "%e\n", ptr1[i]);
    }
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->aa, ptr1, 0, NULL, NULL);
    
    //bb
    fprintf(file1,"SCALARS bb float 1\n");
    fprintf(file1,"LOOKUP_TABLE default\n");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->bb, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    for(int i=0; i<lvl->msh.nv_tot; i++)
    {
        fprintf(file1, "%e\n", ptr1[i]);
    }
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->bb, ptr1, 0, NULL, NULL);
    
    //rr
    fprintf(file1,"SCALARS rr float 1\n");
    fprintf(file1,"LOOKUP_TABLE default\n");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->rr, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    for(int i=0; i<lvl->msh.nv_tot; i++)
    {
        fprintf(file1, "%e\n", ptr1[i]);
    }
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->rr, ptr1, 0, NULL, NULL);
    
    //uu
    fprintf(file1,"SCALARS uu float 1\n");
    fprintf(file1,"LOOKUP_TABLE default\n");
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->uu, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    for(int i=0; i<lvl->msh.nv_tot; i++)
    {
        fprintf(file1, "%e\n", ptr1[i]);
    }
    clEnqueueUnmapMemObject(ocl->command_queue, lvl->uu, ptr1, 0, NULL, NULL);
    
    //    fprintf(file1,"FIELD FieldData 1\n");
    //
    //    //cc
    //    fprintf(file1,"cc 1 %llu float\n", lvl->msh.nv_tot);
    //    ptr1 = clEnqueueMapBuffer(ocl->command_queue, lvl->aa, CL_TRUE, CL_MAP_READ, 0, lvl->msh.nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    //    for(int i=0; i<lvl->msh.nv_tot; i++)
    //    {
    //        fprintf(file1, "%e\n", ptr1[i]);
    //    }
    //    clEnqueueUnmapMemObject(ocl->command_queue, lvl->aa, ptr1, 0, NULL, NULL);
    
    //clean up
    fclose(file1);
    
    return;
}

