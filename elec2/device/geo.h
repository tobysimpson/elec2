//
//  geo.cl
//  mg2
//
//  Created by Toby Simpson on 13.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


#include "sdf.h"


//stimulus
float geo_g0(float3 x)
{
    float3 c = (float3){0.5f,0.5f,0.5f};
    float  r = 0.25f;
    
    return sdf_sph(x, c, r);
}


//cube
float geo_g1(float3 x)
{
    float3 c = (float3){1.0f,1.0f,1.0f};
    float3 r = (float3){1.0f,1.0f,1.0f};

    return sdf_cub(x, c, r);
}

//epicardium
float geo_e1(float3 x)
{
    return sdf_cap(x, (float3){0e0f, 0e0f, -2e0f}, (float3){0e0f, 0e0f, +2e0f}, 6.0f);
}


//heart
float geo_h1(float3 x)
{
    //epicardium
    float s1 = geo_e1(x);
    
    //subtract endocardium for void
    float cap1 = sdf_cap(x, (float3){0e0f, 0e0f, -2e0f}, (float3){0e0f, 0e0f, +2e0f}, 4.0f);    //endo
    s1 = max(s1, -cap1);
    
    //insulate a/v
    float cyl1 = sdf_cyl(x, (float3){0e0f, 0e0f, 1e0f}, 7e0f, 1e0f); //horiz
    s1 = max(s1, -cyl1);
    
    //add purk
    float cyl2 = sdf_cyl(x, (float3){0e0f, 0e0f, 0e0f}, 1e0f, 7e0f);
    s1 = min(s1,cyl2);
    
    return s1;
}

