//
//  mg.h
//  elec2
//
//  Created by Toby Simpson on 21.10.2024.
//

#ifndef mg_h
#define mg_h


//object
struct mg_obj
{
    //levels
    int nl;     //depth
    int le[3];  //log2(ne)
    
    float dx;
    float dt;

    //array
    struct lvl_obj *lvls;
};


//init
void mg_ini(struct mg_obj *mg, struct ocl_obj *ocl)
{
    //params
    mg->nl = 4;
    
    //dims
    mg->le[0] = 4;
    mg->le[1] = 4;
    mg->le[2] = 4;
    
    //scale
    mg->dx = 200.0f/pow(2, mg->le[0]);      //mm
    mg->dt = 1.0f;                          //ms
    
    //allocate
    mg->lvls = malloc(mg->nl*sizeof(struct lvl_obj));

    //levels
    for(int i=0; i<mg->nl; i++)
    {
        struct lvl_obj *lvl = &mg->lvls[i];
        
        //assign
        lvl->idx = i;
        
        //mesh
        lvl->msh.dx = mg->dx*pow(2,i);
        lvl->msh.dt = mg->dt;
        lvl->msh.ne = (cl_int3){pow(2, mg->le[0] - i), pow(2, mg->le[1] - i), pow(2, mg->le[2] - i)};
        lvl->msh.nv = (cl_int3){lvl->msh.ne.x + 1, lvl->msh.ne.y + 1, lvl->msh.ne.z + 1};
        lvl->msh.dx2 = lvl->msh.dx*lvl->msh.dx;
        
        //size_t
        lvl->nv[0] = lvl->msh.nv.x;
        lvl->nv[1] = lvl->msh.nv.y;
        lvl->nv[2] = lvl->msh.nv.z;
        
        //size_t
        lvl->nv_tot = lvl->nv[0]*lvl->nv[1]*lvl->nv[2];
        
        //memory
        lvl->gg = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
        lvl->uu = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
        
        //debug
        printf("i  %d\n", lvl->idx);
        printf("dx %f\n", lvl->msh.dx);
        printf("dt %f\n", lvl->msh.dt);
        printf("ne %d %d %d\n", lvl->msh.ne.x, lvl->msh.ne.y, lvl->msh.ne.z);
//        printf("nv %d %d %d\n", lvl->msh.nv.x, lvl->msh.nv.y, lvl->msh.nv.z);
//        printf("nv %zu %zu %zu\n", lvl->nv[0], lvl->nv[1], lvl->nv[2]);
        printf("nv_tot  %zu\n", lvl->nv_tot);

    }
    
    return;
}


//init
void mg_fin(struct mg_obj *mg, struct ocl_obj *ocl)
{
    //levels
    for(int i=0; i<mg->nl; i++)
    {
        struct lvl_obj *lvl = &mg->lvls[i];
        
        //memory
        ocl->err = clReleaseMemObject(lvl->gg);
        ocl->err = clReleaseMemObject(lvl->uu);
    }
    
    //mem
    free(mg->lvls);

    return;
}


#endif /* mg_h */
