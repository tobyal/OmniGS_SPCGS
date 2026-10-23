#pragma once
// Local screen-space filtering; all prior associations are fixed, non-learned data.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace pipefilter {
struct Voxel { int x,y,z,used; float ax,ay,az,r,q,p0,p1,p2; };
struct State {
    int mode=0,capacity=0;
    float beta=.3f,beta_min=.06f,k=3.f,rho_start=1.f,rho_end=2.f,q_threshold=.25f;
    float voxel_size=.03f,axis_degrees=0.f,radius_scale=1.f;
    const Voxel* voxels=nullptr;
};
State state();
void initialize();
void diagnostics(int count,const float* means,const float* cov,const float* view,int w,int h,float* out);

template<int N> struct Dual {
    float v,d[N];
    __host__ __device__ Dual(float value=0):v(value){for(int i=0;i<N;++i)d[i]=0;}
    __host__ __device__ static Dual variable(float value,int j){Dual a(value);a.d[j]=1;return a;}
};
template<int N> __host__ __device__ Dual<N> operator+(const Dual<N>& a,const Dual<N>& b){Dual<N> c(a.v+b.v);for(int i=0;i<N;++i)c.d[i]=a.d[i]+b.d[i];return c;}
template<int N> __host__ __device__ Dual<N> operator-(const Dual<N>& a,const Dual<N>& b){Dual<N> c(a.v-b.v);for(int i=0;i<N;++i)c.d[i]=a.d[i]-b.d[i];return c;}
template<int N> __host__ __device__ Dual<N> operator-(const Dual<N>& a){Dual<N> c(-a.v);for(int i=0;i<N;++i)c.d[i]=-a.d[i];return c;}
template<int N> __host__ __device__ Dual<N> operator*(const Dual<N>& a,const Dual<N>& b){Dual<N> c(a.v*b.v);for(int i=0;i<N;++i)c.d[i]=a.d[i]*b.v+a.v*b.d[i];return c;}
template<int N> __host__ __device__ Dual<N> operator/(const Dual<N>& a,const Dual<N>& b){Dual<N> c(a.v/b.v);for(int i=0;i<N;++i)c.d[i]=(a.d[i]-c.v*b.d[i])/b.v;return c;}
__host__ __device__ inline float value(float a){return a;}
template<int N> __host__ __device__ float value(const Dual<N>& a){return a.v;}
__host__ __device__ inline float root(float a){return sqrtf(a);}
template<int N> __host__ __device__ Dual<N> root(const Dual<N>& a){Dual<N> c(sqrtf(a.v));for(int i=0;i<N;++i)c.d[i]=.5f*a.d[i]/c.v;return c;}
template<class T> __host__ __device__ T lower(const T& a,float b){return value(a)>b?a:T(b);}
template<class T> __host__ __device__ T clamp01(const T& a){return value(a)<0?T(0):(value(a)>1?T(1):a);}
__host__ __device__ inline uint32_t hash(int x,int y,int z){return uint32_t(x)*73856093u^uint32_t(y)*19349663u^uint32_t(z)*83492791u;}
__host__ __device__ inline Voxel lookup(float3 mean,const State& s){
    Voxel empty={};if(!s.voxels||!s.capacity)return empty;
    int x=int(floorf(mean.x/s.voxel_size)),y=int(floorf(mean.y/s.voxel_size)),z=int(floorf(mean.z/s.voxel_size));
    uint32_t slot=hash(x,y,z)&(s.capacity-1);
    for(int i=0;i<s.capacity;++i){const Voxel v=s.voxels[slot];if(!v.used)return empty;if(v.x==x&&v.y==y&&v.z==z)return v;slot=(slot+1)&(s.capacity-1);}
    return empty;
}
template<class T> struct Result {
    T raw[3],cov[3],conic[3],comp,tx,ty,nx,ny,width,rho,delta,q;
};

// The projection below is the existing OmniGS lon/lat Jacobian, including its
// 1e-7 safeguards. Shared by the new forward, backward and diagnostic paths.
template<class T> __host__ __device__ Result<T> evaluate(const T m[3],const T c[6],const float* v,int width,int height,const Voxel& prior,const State& s){
    Result<T> o={};o.comp=T(1);
    T t[3];for(int j=0;j<3;++j)t[j]=m[0]*T(v[j])+m[1]*T(v[4+j])+m[2]*T(v[8+j])+T(v[12+j]);
    T xz2=t[0]*t[0]+t[2]*t[2],xz=root(lower(xz2,1e-20f));
    T invxz2=T(1)/(xz2+T(1e-7f)),invxz=T(1)/(xz+T(1e-7f));
    T invr2=T(1)/(xz2+t[1]*t[1]+T(1e-7f));
    const float fw=width*.5f*0.3183098861837907f,fh=height*0.3183098861837907f;
    T J[2][3]={{T(fw)*t[2]*invxz2,T(0),-T(fw)*t[0]*invxz2},
      {-T(fh)*t[0]*t[1]*invxz*invr2,T(fh)*xz*invr2,-T(fh)*t[2]*t[1]*invxz*invr2}};
    T A[2][3];for(int j=0;j<2;++j)for(int k=0;k<3;++k)A[j][k]=J[j][0]*T(v[4*k])+J[j][1]*T(v[4*k+1])+J[j][2]*T(v[4*k+2]);
    T C[3][3]={{c[0],c[1],c[2]},{c[1],c[3],c[4]},{c[2],c[4],c[5]}};
    for(int ij=0;ij<3;++ij){int a=ij==2?1:0,b=ij==0?0:1;T sum(0);for(int k=0;k<3;++k){T inner(0);for(int l=0;l<3;++l)inner=inner+C[k][l]*A[b][l];sum=sum+A[a][k]*inner;}o.raw[ij]=sum;}
    float beta=(s.mode==1||s.mode==2)?s.beta:.3f;
    o.cov[0]=o.raw[0]+T(beta);o.cov[1]=o.raw[1];o.cov[2]=o.raw[2]+T(beta);
    if(s.mode==7 && prior.used){o.q=T(1);o.delta=T(.3f-s.beta_min);o.cov[0]=o.cov[0]-o.delta;o.cov[2]=o.cov[2]-o.delta;}
    // The categorical field lookup and reliability membership are detached.
    if(s.mode>=3 && s.mode!=7 && prior.used && prior.r>0 && value(xz2)>1e-10f){
        float ax=prior.ax,ay=prior.ay,az=prior.az;
        if(s.axis_degrees!=0){float angle=s.axis_degrees*.0174532925199433f;float xx=cosf(angle)*ax-sinf(angle)*az;az=sinf(angle)*ax+cosf(angle)*az;ax=xx;}
        T du=A[0][0]*T(ax)+A[0][1]*T(ay)+A[0][2]*T(az);
        T dv=A[1][0]*T(ax)+A[1][1]*T(ay)+A[1][2]*T(az);
        T tangent2=du*du+dv*dv;
        T camera_axis[3];for(int k=0;k<3;++k)camera_axis[k]=T(ax*v[4*0+k]+ay*v[4*1+k]+az*v[4*2+k]);
        T radius2=xz2+t[1]*t[1];T dot=camera_axis[0]*t[0]+camera_axis[1]*t[1]+camera_axis[2]*t[2];
        T side=T(1)-dot*dot/lower(radius2,1e-12f);
        if(value(tangent2)>1e-10f && value(side)>.04f){
            T length=root(tangent2);o.tx=du/length;o.ty=dv/length;o.nx=-o.ty;o.ny=o.tx;
            // A circular cross-section perpendicular to axis: transverse support
            // r*sqrt(n^T J R (I-aa^T) R^T J^T n). Full diameter is twice this.
            T g[3];for(int k=0;k<3;++k)g[k]=o.nx*A[0][k]+o.ny*A[1][k];
            T ga=g[0]*T(ax)+g[1]*T(ay)+g[2]*T(az);
            T widthvar=g[0]*g[0]+g[1]*g[1]+g[2]*g[2]-ga*ga;
            o.width=T(2*prior.r*s.radius_scale)*root(lower(widthvar,1e-12f));
            T transverse=o.nx*o.nx*o.raw[0]+T(2)*o.nx*o.ny*o.raw[1]+o.ny*o.ny*o.raw[2];
            o.rho=T(2*s.k)*root(lower(transverse+T(.3f),1e-12f))/(o.width+T(1e-6f));
            T u=clamp01((o.rho-T(s.rho_start))/T(s.rho_end-s.rho_start));
            T response=u*u*(T(3)-T(2)*u);
            // Smooth band-pass in projected diameter: preserve anti-aliasing
            // below 1 px; fade in over 1--2 px and out over 8--16 px.
            T lo=clamp01(o.width-T(1));
            T hi=clamp01((T(16)-o.width)/T(8));
            T band=lo*lo*(T(3)-T(2)*lo)*hi*hi*(T(3)-T(2)*hi);
            response=response*band;
            if(s.mode==3 || s.mode==7)response=T(1);
            float confidence=(s.mode==6||s.mode==8)?(prior.q>=s.q_threshold?prior.q:0.f):1.f;
            o.q=T(confidence);o.delta=T(.3f-s.beta_min)*response*o.q;
            if(s.mode==4||s.mode==7){o.cov[0]=o.cov[0]-o.delta;o.cov[2]=o.cov[2]-o.delta;}
            else{o.cov[0]=o.cov[0]-o.delta*o.nx*o.nx;o.cov[1]=o.cov[1]-o.delta*o.nx*o.ny;o.cov[2]=o.cov[2]-o.delta*o.ny*o.ny;}
        }
    }
    T determinant=lower(o.cov[0]*o.cov[2]-o.cov[1]*o.cov[1],1e-12f);
    o.conic[0]=o.cov[2]/determinant;o.conic[1]=-o.cov[1]/determinant;o.conic[2]=o.cov[0]/determinant;
    if(s.mode==2||s.mode==8){T detraw=lower(o.raw[0]*o.raw[2]-o.raw[1]*o.raw[1],1e-12f);o.comp=root(clamp01(detraw/determinant));}
    return o;
}
}
