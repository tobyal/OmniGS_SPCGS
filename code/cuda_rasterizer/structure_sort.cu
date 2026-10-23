// Industrial selective ray-conditioned ordering. The backward kernel recomputes
// exactly the forward permutation in each bounded radial candidate window.
// Sorting/prior association are discrete: no derivative of the permutation.
#include "structure_sort.h"
#include "pipe_filter.cuh"
#include "config.h"
#include "forward.h"
#include "backward.h"
#include <cstdlib>
#include <stdexcept>

namespace structuresort {
static float* diagnostic_buffer=nullptr;
void setDiagnostics(float* pointer){diagnostic_buffer=pointer;}
int mode(){ static int m=[](){const char* p=std::getenv("SGS_MODE");int v=p?std::atoi(p):0;if(v<0||v>5)throw std::runtime_error("SGS_MODE must be 0..5");return v;}();return m; }
int window(){static int w=[](){const char* p=std::getenv("SGS_WINDOW");int v=p?std::atoi(p):32;if(v!=16&&v!=32&&v!=64)throw std::runtime_error("SGS_WINDOW must be 16/32/64");return v;}();return w;}
__device__ float dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ float smooth(float x){x=fminf(1.f,fmaxf(0.f,x));return x*x*(3-2*x);}
__global__ void prepareKernel(int n,const float* xyz,const float* cov,const float* camera,const int* radii,Gaussian* out,pipefilter::State field){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||radii[i]<=0)return;
 Gaussian g={};float3 m={xyz[3*i]-camera[0],xyz[3*i+1]-camera[1],xyz[3*i+2]-camera[2]};
 g.m[0]=m.x;g.m[1]=m.y;g.m[2]=m.z;g.radial=sqrtf(dot(m,m));
 const float* c=cov+6*i;float norm=fmaxf(fmaxf(c[0],c[3]),fmaxf(c[5],1e-20f));
 float a=c[0]/norm+1e-7f,b=c[1]/norm,d=c[2]/norm,e=c[3]/norm+1e-7f,f=c[4]/norm,h=c[5]/norm+1e-7f;
 float cof[6]={e*h-f*f,d*f-b*h,b*f-d*e,a*h-d*d,b*d-a*f,a*e-b*b};
 float det=a*cof[0]+b*cof[1]+d*cof[2];
 if(det>1e-20f){for(int j=0;j<6;++j)g.q[j]=cof[j]/(det*norm);}
 else{g.q[0]=g.q[3]=g.q[5]=1.f/norm;}
 auto prior=pipefilter::lookup({xyz[3*i],xyz[3*i+1],xyz[3*i+2]},field);
 if(prior.used){g.identity=int(prior.p0);g.confidence=prior.p1;
  if(g.identity==0 && prior.q>=.25f){g.identity=1;g.confidence=prior.q;}}
 out[i]=g;
}
void prepare(int n,const float* xyz,const float* cov,const float* camera,const int* radii,Gaussian* out){prepareKernel<<<(n+255)/256,256>>>(n,xyz,cov,camera,radii,out,pipefilter::state());}
__global__ void selectKernel(const uint2* ranges,const uint32_t* list,const Gaussian* g,const float4* conic,unsigned char* selected,int mode){
 int tile=blockIdx.x;uint2 range=ranges[tile];bool active=mode==2||mode==3;
 for(int i=range.x+threadIdx.x;i<range.y&&!active;i+=blockDim.x){int id=list[i];active=g[id].identity>0&&g[id].confidence>0&&conic[id].w>=1.f/255.f;}
 int any=__syncthreads_or(active);if(threadIdx.x==0)selected[tile]=any;
}
void selectTiles(dim3 grid,const uint2* ranges,const uint32_t* list,const Gaussian* g,const float4* conic,unsigned char* selected){selectKernel<<<grid.x*grid.y,256>>>(ranges,list,g,conic,selected,mode());}

__device__ float3 ray(int x,int y,int w,int h,const float* v){
 float lon=((x+.5f)/w-.5f)*6.283185307179586f,lat=((y+.5f)/h-.5f)*3.141592653589793f;
 float3 d={cosf(lat)*sinf(lon),sinf(lat),cosf(lat)*cosf(lon)};
 return {v[0]*d.x+v[1]*d.y+v[2]*d.z,v[4]*d.x+v[5]*d.y+v[6]*d.z,v[8]*d.x+v[9]*d.y+v[10]*d.z};
}
__device__ void rayDepth(const Gaussian& g,float3 d,float& t,float& sigma){
 float3 qd={g.q[0]*d.x+g.q[1]*d.y+g.q[2]*d.z,g.q[1]*d.x+g.q[3]*d.y+g.q[4]*d.z,g.q[2]*d.x+g.q[4]*d.y+g.q[5]*d.z};
 float den=fmaxf(dot(d,qd),1e-20f);t=(qd.x*g.m[0]+qd.y*g.m[1]+qd.z*g.m[2])/den;sigma=rsqrtf(den);
 if(!isfinite(t)||t<=0)t=g.radial;
}
__device__ float alphaAt(int id,float2 pixel,const float2* xy,const float4* conic){
 float2 d={xy[id].x-pixel.x,xy[id].y-pixel.y};float4 c=conic[id];
 float p=-.5f*(c.x*d.x*d.x+c.z*d.y*d.y)-c.y*d.x*d.y;
 if(p>0)return 0;float alpha=fminf(.99f,c.w*expf(p));return alpha<1.f/255?0:alpha;
}
template<int K> __device__ int permutation(uint2 range,int start,const uint32_t* list,float2 pixel,float3 d,const float2* xy,const float4* conic,const Gaussian* gaussians,int mode,int (&ids)[K],float (&alpha)[K],float& risk_out){
 risk_out=0;
 int n=min(K,int(range.y-range.x)-start);float key[K],t[K],sigma[K];bool structure=false;
 for(int j=0;j<n;++j){ids[j]=list[range.x+start+j];alpha[j]=alphaAt(ids[j],pixel,xy,conic);const auto& g=gaussians[ids[j]];key[j]=g.radial;structure|=alpha[j]>0&&g.identity>0&&g.confidence>0;}
 if((mode==1||mode==4||mode==5)&&!structure)return n;
 for(int j=0;j<n;++j)if(alpha[j]>0)rayDepth(gaussians[ids[j]],d,t[j],sigma[j]);
 float risk=mode==3?1.f:0.f;
 if(mode!=3)for(int i=0;i<n;++i)if(alpha[i]>0)for(int j=i+1;j<n;++j)if(alpha[j]>0){
  const auto& a=gaussians[ids[i]];const auto& b=gaussians[ids[j]];
  float identity=1.f;
  if(mode!=2){
   if(a.identity==b.identity)continue;
   identity=(a.identity&&b.identity)?sqrtf(a.confidence*b.confidence):.5f*fmaxf(a.confidence,b.confidence);
  }
  float overlap=smooth(alpha[i]*alpha[j]/.0004f);
  float depth_scale=fmaxf(sqrtf(sigma[i]*sigma[i]+sigma[j]*sigma[j]),1e-5f);
  float near=smooth(1.f-fabsf(a.radial-b.radial)/(3.f*depth_scale));
  float reversal=(a.radial-b.radial)*(t[i]-t[j])<0?smooth(fabsf(t[i]-t[j])/depth_scale):0.f;
  if(mode==4){near=1.f;reversal=1.f;}
  if(mode==5)overlap=1.f;
  risk=fmaxf(risk,identity*overlap*near*reversal);
 }
 if(risk<=0)return n;
 risk_out=risk;
 for(int j=0;j<n;++j)if(alpha[j]>0)key[j]+=risk*(t[j]-key[j]);
 // Stable insertion sort. Zero-contribution candidates remain in the window,
 // so all significant contributors, including unlabelled background, participate.
 for(int j=1;j<n;++j){int id=ids[j];float a=alpha[j],k=key[j];int q=j-1;while(q>=0&&key[q]>k){ids[q+1]=ids[q];alpha[q+1]=alpha[q];key[q+1]=key[q];--q;}ids[q+1]=id;alpha[q+1]=a;key[q+1]=k;}
 return n;
}
template<int K> __global__ void forwardKernel(const unsigned char* tiles,const uint2* ranges,const uint32_t* list,int w,int h,const float2* xy,const float* colors,const float4* conic,const Gaussian* gaussian,const float* view,float* finalT,uint32_t* count,const float* bg,float* out,int mode,float* diagnostic){
 if(!tiles[blockIdx.y*((w+BLOCK_X-1)/BLOCK_X)+blockIdx.x])return;
 int x=blockIdx.x*BLOCK_X+threadIdx.x,y=blockIdx.y*BLOCK_Y+threadIdx.y;if(x>=w||y>=h)return;int p=y*w+x;
 uint2 range=ranges[blockIdx.y*((w+BLOCK_X-1)/BLOCK_X)+blockIdx.x];float3 d=ray(x,y,w,h,view);float T=1,C[3]={};uint32_t last=0;bool done=false;
 float windows=0,valid=0,moved=0,maxrisk=0;
 for(int start=0;start<int(range.y-range.x)&&!done;start+=K){int ids[K];float alpha[K],risk;int n=permutation<K>(range,start,list,{float(x),float(y)},d,xy,conic,gaussian,mode,ids,alpha,risk);
  if(diagnostic){windows+=risk>0;maxrisk=fmaxf(maxrisk,risk);for(int j=0;j<n;++j)if(alpha[j]>0){++valid;moved+=ids[j]!=list[range.x+start+j];}}
  for(int j=0;j<n;++j){float a=alpha[j];if(a==0)continue;float test=T*(1-a);if(test<.0001f){done=true;break;}for(int ch=0;ch<3;++ch)C[ch]+=colors[3*ids[j]+ch]*a*T;T=test;last=start+j+1;}}
 finalT[p]=T;count[p]=last;for(int ch=0;ch<3;++ch)out[ch*w*h+p]=C[ch]+T*bg[ch];
 if(diagnostic){diagnostic[p]=windows;diagnostic[w*h+p]=valid;diagnostic[2*w*h+p]=moved;diagnostic[3*w*h+p]=maxrisk;}
}
template<int K> __global__ void backwardKernel(const unsigned char* tiles,const uint2* ranges,const uint32_t* list,int w,int h,const float2* xy,const float* colors,const float4* conic,const Gaussian* gaussian,const float* view,const float* finalT,const uint32_t* count,const float* bg,const float* dPixel,float3* dXY,float4* dConic,float* dOpacity,float* dColor,int mode){
 if(!tiles[blockIdx.y*((w+BLOCK_X-1)/BLOCK_X)+blockIdx.x])return;
 int x=blockIdx.x*BLOCK_X+threadIdx.x,y=blockIdx.y*BLOCK_Y+threadIdx.y;if(x>=w||y>=h)return;int p=y*w+x,last=count[p];if(!last)return;
 uint2 range=ranges[blockIdx.y*((w+BLOCK_X-1)/BLOCK_X)+blockIdx.x];float3 direction=ray(x,y,w,h,view);float T=finalT[p],suffix[3]={},lastcolor[3]={},lastalpha=0,dl[3];for(int c=0;c<3;++c)dl[c]=dPixel[c*w*h+p];
 for(int start=((last-1)/K)*K;start>=0;start-=K){int ids[K];float alpha[K],risk;int n=permutation<K>(range,start,list,{float(x),float(y)},direction,xy,conic,gaussian,mode,ids,alpha,risk);
  for(int j=n-1;j>=0;--j){if(start+j>=last||alpha[j]==0)continue;int id=ids[j];float a=alpha[j];float4 c=conic[id];float2 delta={xy[id].x-x,xy[id].y-y};float G=expf(-.5f*(c.x*delta.x*delta.x+c.z*delta.y*delta.y)-c.y*delta.x*delta.y);T/=1-a;
   float da=0,bgd=0;for(int ch=0;ch<3;++ch){suffix[ch]=lastalpha*lastcolor[ch]+(1-lastalpha)*suffix[ch];lastcolor[ch]=colors[3*id+ch];da+=(lastcolor[ch]-suffix[ch])*dl[ch];atomicAdd(dColor+3*id+ch,a*T*dl[ch]);bgd+=bg[ch]*dl[ch];}
   da=da*T-finalT[p]/(1-a)*bgd;lastalpha=a;
   // Match the baseline derivative convention, including the opacity cap.
   float dg=c.w*da,gx=G*delta.x,gy=G*delta.y;
   atomicAdd(&dXY[id].x,dg*(-gx*c.x-gy*c.y)*(.5f*w));atomicAdd(&dXY[id].y,dg*(-gy*c.z-gx*c.y)*(.5f*h));
   atomicAdd(&dConic[id].x,-.5f*gx*delta.x*dg);atomicAdd(&dConic[id].y,-.5f*gx*delta.y*dg);atomicAdd(&dConic[id].w,-.5f*gy*delta.y*dg);atomicAdd(dOpacity+id,G*da);
  }}
}
void forward(dim3 grid,dim3 block,const unsigned char* tiles,const uint2* ranges,const uint32_t* list,int w,int h,const float2* xy,const float* colors,const float4* conic,const Gaussian* gaussian,const float* view,float* T,uint32_t* n,const float* bg,float* out){
 FORWARD::render(grid,block,ranges,list,w,h,xy,colors,conic,T,n,bg,out,tiles);
#define RUN(K) forwardKernel<K><<<grid,block>>>(tiles,ranges,list,w,h,xy,colors,conic,gaussian,view,T,n,bg,out,mode(),diagnostic_buffer)
 if(window()==16){RUN(16);}else if(window()==64){RUN(64);}else{RUN(32);}
#undef RUN
}
void backward(dim3 grid,dim3 block,const unsigned char* tiles,const uint2* ranges,const uint32_t* list,int w,int h,const float2* xy,const float* colors,const float4* conic,const Gaussian* gaussian,const float* view,const float* T,const uint32_t* n,const float* bg,const float* dp,float3* dx,float4* dc,float* dop,float* dcolor){
 BACKWARD::render(grid,block,ranges,list,w,h,bg,xy,conic,colors,T,n,dp,dx,dc,dop,dcolor,tiles);
#define RUN(K) backwardKernel<K><<<grid,block>>>(tiles,ranges,list,w,h,xy,colors,conic,gaussian,view,T,n,bg,dp,dx,dc,dop,dcolor,mode())
 if(window()==16){RUN(16);}else if(window()==64){RUN(64);}else{RUN(32);}
#undef RUN
}
}
