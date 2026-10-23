#include "include/gaussian_rasterizer.h"
#include "cuda_rasterizer/pipe_filter.cuh"
#include <iostream>
#include <cmath>
std::pair<float,std::vector<float>> run(const std::vector<float>& p,bool grad){
 auto opt=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
 auto make=[&](int start,int count){return torch::from_blob((void*)(p.data()+start),{1,count},torch::kFloat32).clone().to(torch::kCUDA).set_requires_grad(grad);};
 auto xyz=make(0,3),cov=make(3,6),opacity=make(9,1);
 auto bg=torch::zeros({3},opt),view=torch::eye(4,opt),proj=torch::eye(4,opt),rot=torch::eye(3,opt),cam=torch::zeros({3},opt);
 GaussianRasterizationSettings settings(32,64,0,0,bg,1.f,view,proj,rot,0,cam,false,Camera::CameraModelType::LONLAT);
 GaussianRasterizer raster(settings);auto empty=torch::empty({0},opt),screen=torch::zeros({1,3},opt);
 auto color=torch::tensor({{.8f,.5f,.2f}},opt);
 auto image=std::get<0>(raster.forward(xyz,screen,opacity,false,true,false,false,true,empty,color,empty,empty,cov));
 auto weight=(torch::arange(3*32*64,opt).reshape({3,32,64})/(3*32*64)+.2).sin();auto loss=(image*weight).sum();
 std::vector<float> gradients;
 if(grad){loss.backward();for(auto x:{xyz,cov,opacity}){auto g=x.grad().cpu().contiguous().flatten();for(int i=0;i<g.numel();++i)gradients.push_back(g.data_ptr<float>()[i]);}}
 return {loss.item<float>(),gradients};
}
int main(){
 pipefilter::initialize();std::vector<float> p={.2f,.1f,2.f,.003f,.0001f,-.0002f,.004f,.00015f,.005f,.4f};
 auto analytic=run(p,true);float worst=0;
 for(int j=0;j<10;++j){float old=p[j],eps=(j<3||j==9)?1e-3f:2e-5f;p[j]=old+eps;float plus=run(p,false).first;p[j]=old-eps;float minus=run(p,false).first;p[j]=old;
  float numerical=(plus-minus)/(2*eps),g=analytic.second[j];float scaled=std::abs(numerical-g)/(.005f+.025f*std::max(std::abs(numerical),std::abs(g)));worst=std::max(worst,scaled);
  if(!(scaled<1)){std::cerr<<"Raster gradient failure "<<j<<" analytic="<<g<<" numerical="<<numerical<<" scaled="<<scaled<<std::endl;return 2;}
 }
 std::cout<<"{\"status\":\"passed\",\"mode\":"<<pipefilter::state().mode<<",\"components\":10,\"max_scaled_error\":"<<worst<<"}"<<std::endl;
}
