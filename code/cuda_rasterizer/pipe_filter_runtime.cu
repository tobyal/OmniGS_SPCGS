#include "pipe_filter.cuh"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <mutex>
namespace pipefilter {
static State current;
static std::once_flag initialized;
static float env(const char* key,float fallback){const char* p=std::getenv(key);return p?std::stof(p):fallback;}
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
State state(){return current;}
void initialize(){std::call_once(initialized,[]{
    current.mode=int(env("PF_MODE",0));
    current.beta=env("PF_BETA",.3f);current.beta_min=env("PF_BETA_MIN",.06f);
    current.k=env("PF_K",3);current.rho_start=env("PF_RHO_START",1);current.rho_end=env("PF_RHO_END",2);
    current.q_threshold=env("PF_Q_THRESHOLD",.25f);
    current.axis_degrees=env("PF_AXIS_DEGREES",0);current.radius_scale=env("PF_RADIUS_SCALE",1);
    if(current.mode<0||current.mode>8||current.beta<.001f||current.beta_min<.001f||current.beta_min>.3f||current.rho_end<=current.rho_start||current.radius_scale<=0)
        throw std::runtime_error("Invalid PF_* filter configuration");
    if(current.mode>=3 || std::getenv("PF_FIELD")){
        const char* path=std::getenv("PF_FIELD");if(!path)throw std::runtime_error("PF_FIELD is required for structure filters");
        std::ifstream f(path,std::ios::binary);uint32_t magic=0,capacity=0;float voxel=0;uint32_t version=0;
        f.read((char*)&magic,4);f.read((char*)&capacity,4);f.read((char*)&voxel,4);f.read((char*)&version,4);
        if(!f||magic!=0x50464631||version!=1||capacity==0||(capacity&(capacity-1))||voxel<=0)throw std::runtime_error("Invalid pipe field header");
        std::vector<Voxel> host(capacity);f.read((char*)host.data(),host.size()*sizeof(Voxel));
        if(!f||f.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Invalid pipe field payload");
        Voxel* device=nullptr;check(cudaMalloc((void**)&device,host.size()*sizeof(Voxel)));
        check(cudaMemcpy(device,host.data(),host.size()*sizeof(Voxel),cudaMemcpyHostToDevice));
        current.voxels=device;current.capacity=capacity;current.voxel_size=voxel;
    }
    std::cout<<"[PipeFilter] mode="<<current.mode<<" beta="<<current.beta<<" beta_min="<<current.beta_min<<" field_capacity="<<current.capacity<<std::endl;
});}
__global__ void diagnosticKernel(int count,const float* means,const float* cov,const float* view,int w,int h,float* out,State s){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float3 m=make_float3(means[3*i],means[3*i+1],means[3*i+2]);Voxel p=lookup(m,s);
    auto r=evaluate<float>(means+3*i,cov+6*i,view,w,h,p,s);float* o=out+i*24;
    o[0]=p.ax;o[1]=p.ay;o[2]=p.az;o[3]=p.r;o[4]=p.q;o[5]=p.used;
    for(int k=0;k<3;++k){o[6+k]=r.raw[k];o[9+k]=r.cov[k];}
    o[12]=r.tx;o[13]=r.ty;o[14]=r.nx;o[15]=r.ny;o[16]=r.width;o[17]=r.rho;o[18]=r.delta;o[19]=r.comp;o[20]=r.q;
    for(int k=0;k<3;++k)o[21+k]=r.conic[k];
}
void diagnostics(int count,const float* means,const float* cov,const float* view,int w,int h,float* out){
    initialize();diagnosticKernel<<<(count+255)/256,256>>>(count,means,cov,view,w,h,out,current);check(cudaGetLastError());
}
}
