#include "include/gaussian_rasterizer.h"
#include "cuda_rasterizer/pipe_filter.cuh"
#include "cuda_rasterizer/structure_sort.h"
#include <iostream>
#include <cmath>
std::pair<float,std::vector<float>> run(const std::vector<float>& p,bool grad){
 auto opt=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
 auto make=[&](int start,int width){return torch::from_blob((void*)(p.data()+start),{2,width},torch::kFloat32).clone().cuda().set_requires_grad(grad);};
 auto xyz=make(0,3),cov=make(6,6),opacity=make(18,1);
 auto bg=torch::tensor({.1f,.2f,.05f},opt),view=torch::eye(4,opt),proj=torch::eye(4,opt),rot=torch::eye(3,opt),cam=torch::zeros({3},opt);
 GaussianRasterizationSettings settings(32,64,0,0,bg,1.f,view,proj,rot,0,cam,false,Camera::CameraModelType::LONLAT);
 GaussianRasterizer raster(settings);auto empty=torch::empty({0},opt),screen=torch::zeros({2,3},opt);
 auto color=torch::tensor({{.8f,.2f,.1f},{.1f,.4f,.9f}},opt);
 auto image=std::get<0>(raster.forward(xyz,screen,opacity,false,true,false,false,true,empty,color,empty,empty,cov));
 auto weight=(torch::arange(3*32*64,opt).reshape({3,32,64})/(3*32*64)+.2).sin();auto loss=(image*weight).sum();
 std::vector<float> gradients;
 if(grad){loss.backward();for(auto x:{xyz,cov,opacity}){auto g=x.grad().cpu().contiguous().flatten();for(int i=0;i<g.numel();++i)gradients.push_back(g.data_ptr<float>()[i]);}}
 return {loss.item<float>(),gradients};
}
int main(){
 pipefilter::initialize();std::vector<float> p={.12f,.08f,2.11f,-.12f,.07f,2.09f,.03f,0.f,.02f,.02f,0.f,.08f,.03f,0.f,.02f,.02f,0.f,.08f,.6f,.55f};
 auto analytic=run(p,true);float worst=0;
 for(int j=0;j<20;++j){float old=p[j],eps=(j<6||j>=18)?.001f:.0001f;p[j]=old+eps;float plus=run(p,false).first;p[j]=old-eps;float minus=run(p,false).first;p[j]=old;
  float numerical=(plus-minus)/(2*eps),g=analytic.second[j];float scaled=std::abs(numerical-g)/(.3f+.035f*std::max(std::abs(numerical),std::abs(g)));worst=std::max(worst,scaled);
  if(!(scaled<1)){std::cerr<<"Sort gradient failure "<<j<<" analytic="<<g<<" numerical="<<numerical<<" scaled="<<scaled<<std::endl;return 2;}
 }
 std::cout<<"{\"status\":\"passed\",\"filter_mode\":"<<pipefilter::state().mode<<",\"sorting_mode\":"<<structuresort::mode()<<",\"components\":20,\"loss\":"<<analytic.first<<",\"max_scaled_error\":"<<worst<<"}"<<std::endl;
}
