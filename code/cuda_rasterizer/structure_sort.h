#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace structuresort {
struct Gaussian { float q[6], m[3], radial, confidence; int identity; };
int mode();
void setDiagnostics(float*);
void prepare(int, const float*, const float*, const float*, const int*, Gaussian*);
void selectTiles(dim3,const uint2*,const uint32_t*,const Gaussian*,const float4*,unsigned char*);
void forward(dim3,dim3,const unsigned char*,const uint2*,const uint32_t*,int,int,const float2*,const float*,const float4*,const Gaussian*,const float*,float*,uint32_t*,const float*,float*);
void backward(dim3,dim3,const unsigned char*,const uint2*,const uint32_t*,int,int,const float2*,const float*,const float4*,const Gaussian*,const float*,const float*,const uint32_t*,const float*,const float*,float3*,float4*,float*,float*);
}
