//
//  sdf.cl
//  mg2
//
//  Created by Toby Simpson on 13.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


//cuboid
float sdf_cub(float3 x, float3 c, float3 r)
{
    float3 d = fabs(x - c) - r;
    
    return max(d.x,max(d.y,d.z));
}


//sphere
float sdf_sph(float3 x, float3 c, float r)
{
    return length(x - c) - r;
}


//capsule (from,to,r)
float sdf_cap(float3 p, float3 a, float3 b, float r)
{
    float3 pa = p - a;
    float3 ba = b - a;
    
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0f, 1.0f);
    
    return length(pa - ba*h) - r;
}


//cylinder z-axis (x,c,r,h)
float sdf_cyl(float3 x, float3 c, float r, float h)
{
    return max(length(x.xy - c.xy) - r, fabs(x.z - c.z) - h);
}
