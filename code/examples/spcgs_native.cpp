// SPC-GS research adapter. OmniGS retains projection, compositing, backward,
// PCD initialization, Gaussian optimization and densification.
#include <torch/torch.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <json/json.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <iostream>
#include <cstdlib>
#include "include/gaussian_model.h"
#include "include/gaussian_renderer.h"
#include "include/loss_utils.h"
#include "cuda_rasterizer/forward.h"

namespace fs=std::filesystem;
using torch::indexing::Slice;
using T=torch::Tensor;
const auto GPU=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
Json::Value readj(const fs::path& p) { std::ifstream f(p); TORCH_CHECK(f.good(),"Missing ",p.string()); Json::Value j; f>>j; return j; }
void writej(const fs::path& p,const Json::Value& j) { std::ofstream f(p); f<<j; TORCH_CHECK(f.good(),"Write failed ",p.string()); }
T raw(const fs::path& p,std::vector<int64_t> shape,torch::ScalarType dtype=torch::kFloat32) {
    auto x=torch::empty(shape,torch::TensorOptions().dtype(dtype));
    std::ifstream f(p,std::ios::binary|std::ios::ate); TORCH_CHECK(f.good() && f.tellg()==x.nbytes(),"Bad binary size ",p.string());
    f.seekg(0);f.read((char*)x.data_ptr(),x.nbytes()); return x.cuda();
}
void put(const fs::path& p,T x) {x=x.detach().cpu().contiguous();std::ofstream f(p,std::ios::binary);f.write((char*)x.data_ptr(),x.nbytes());TORCH_CHECK(f.good(),"Write failed");}
void png(const fs::path& p,T x) {x=x.detach().clamp(0,1);auto im=tensor_utils::torchTensor2CvMat_Float32(x);cv::cvtColor(im,im,cv::COLOR_RGB2BGR);im.convertTo(im,CV_8UC3,255);cv::imwrite(p.string(),im);}
T image(const std::string& p,int H,int W) {auto im=cv::imread(p);TORCH_CHECK(!im.empty(),"Missing image ",p);cv::resize(im,im,{W,H},0,0,cv::INTER_AREA);cv::cvtColor(im,im,cv::COLOR_BGR2RGB);im.convertTo(im,CV_32FC3,1./255.);return tensor_utils::cvMat2TorchTensor_Float32(im,torch::kCUDA);}
T norm(T x,int d) {return x/torch::norm(x,2,d,true).clamp_min(1e-6);}
T resize(T x,int H,int W) {return torch::nn::functional::interpolate(x.unsqueeze(0),torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{H,W}).mode(torch::kBilinear).align_corners(false)).squeeze(0);}
std::shared_ptr<GaussianKeyframe> camera(const Json::Value& j,int H,int W) {
    auto f=std::make_shared<GaussianKeyframe>(0); Camera c;c.camera_id_=0;c.setModelId(Camera::CameraModelType::LONLAT);c.width_=W;c.height_=H;c.num_gaus_pyramid_sub_levels_=0;
    f->setCameraParams(c);f->znear_=0.01;f->zfar_=1000;
    // Match the original readViewsOpenMVG float conversion before quaternion construction.
    Eigen::Matrix3f R;Eigen::Vector3f C;for(int a=0;a<3;++a){C[a]=j["center"][a].asFloat();for(int b=0;b<3;++b)R(a,b)=j["rotation"][a][b].asFloat();}
    Eigen::Vector3f t=-R*C;f->setPose(Eigen::Quaterniond(R.cast<double>()),t.cast<double>());f->computeTransformTensors();return f;
}
struct Render {T rgb,sem,ids,weights,screen,visible,radii;};
Render render(std::shared_ptr<GaussianModel> m,const Json::Value& cam,int H,int W,int HS,int WS,bool semantics=true) {
    auto f=camera(cam,H,W);GaussianPipelineParams pipe;T bg=torch::zeros({3},GPU),empty=torch::empty({0},GPU);
    auto rr=GaussianRenderer::renderLonlat(f,H,W,m,pipe,bg,empty,1.f,false);
    Render r;std::tie(r.rgb,r.screen,r.visible,r.radii)=rr;
    if(semantics && m->semantic_.defined()) {
        r.ids=torch::full({HS,WS},-1,torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));r.weights=torch::zeros({HS,WS},GPU);
        std::vector<T> chunks;for(int d=0;d<32;d+=3) {
            auto color=m->semantic_.slice(1,d,std::min(32,d+3));
            if(color.size(1)<3)color=torch::cat({color,torch::zeros({color.size(0),3-color.size(1)},GPU)},1);
            color=color.contiguous(); if(d==0)FORWARD::setSpcDiagnostics(r.ids.data_ptr<int>(),r.weights.data_ptr<float>());
            auto result=GaussianRenderer::renderLonlat(f,HS,WS,m,pipe,bg,color,1.f,true); FORWARD::setSpcDiagnostics(nullptr,nullptr);
            chunks.push_back(std::get<0>(result).slice(0,0,std::min(3,32-d)));
        } r.sem=torch::cat(chunks,0);
    }
    return r;
}

struct Heads {
    T decoder,bias,classifier,cbias,projection,text;
    std::unique_ptr<torch::optim::Adam> optim;
    void init(const Json::Value& c,int K) {
        decoder=raw(c["decoder"].asString(),{512,32}).detach().requires_grad_();bias=torch::zeros({512},GPU.requires_grad(true));
        classifier=torch::eye(K,GPU).requires_grad_();cbias=torch::zeros({K},GPU.requires_grad(true));projection=torch::eye(32,GPU).requires_grad_();
        text=raw(c["text"].asString(),{K,512});setup();
    }
    void setup(){optim=std::make_unique<torch::optim::Adam>(std::vector<T>{decoder,bias,classifier,cbias,projection},torch::optim::AdamOptions(.0005));}
    T features(T x){return decoder.matmul(x.flatten(1))+bias.unsqueeze(1);}
    T logits(T x){return classifier.matmul(text.matmul(norm(features(x),0))*10.f)+cbias.unsqueeze(1);}
};
T semantic_loss(Heads& h,T features,const Json::Value& v,int H,int W,T* cosine_out=nullptr,T* ce_out=nullptr) {
    int ch=v["feature_height"].asInt(),cw=v["feature_width"].asInt();auto gt=resize(raw(v["feature"].asString(),{512,ch,cw},torch::kFloat16).to(torch::kFloat32),H,W).flatten(1);
    TORCH_CHECK(v.isMember("relevancy"),"Missing cached S relevancy input; regenerate the feature manifest");
    TORCH_CHECK(v["relevancy_classes"].asInt()==h.text.size(0),"S/T class row count mismatch");
    int sh=v["relevancy_height"].asInt(),sw=v["relevancy_width"].asInt();
    auto scores=resize(raw(v["relevancy"].asString(),{h.text.size(0),sh,sw},torch::kFloat16).to(torch::kFloat32),H,W).flatten(1);
    auto pred=h.features(features);auto cosine=(1-(norm(pred,0)*norm(gt,0)).sum(0)).mean();T ce;
    if(v.get("semantic_target","softmax_relevancy").asString()=="hard_argmax") {
        // Compatibility for the explicitly selected legacy extractor: still read
        // S from disk, never silently reconstruct labels from F and T here.
        ce=torch::nn::functional::cross_entropy(h.logits(features).transpose(0,1),scores.argmax(0));
    } else {
        // SPC-GS does not publish the target conversion. Retain continuous S
        // using the class softmax used by 3D-OVS, with SPC-GS cross entropy.
        auto target=torch::softmax(scores,0);
        ce=-(target*torch::log_softmax(h.logits(features),0)).sum(0).mean();
    }
    if(cosine_out)*cosine_out=cosine;if(ce_out)*ce_out=ce;return cosine+ce;
}
T erode(T m) {
    // Circular horizontal padding keeps ERP seam continuous; vertical outside is invalid.
    auto x=m.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);
    x=torch::cat({x.slice(3,x.size(3)-2,x.size(3)),x,x.slice(3,0,2)},3);
    x=torch::nn::functional::pad(x,torch::nn::functional::PadFuncOptions({0,0,2,2}).value(0));
    return (torch::nn::functional::avg_pool2d(x,torch::nn::functional::AvgPool2dFuncOptions(5).stride(1))>0.999).squeeze();
}
T zero_loss(T x){return x.sum()*0;}
T inter(Heads& h,const Render& a,const Render& b,T ma,T mb) {
    if(!ma.any().item<bool>()||!mb.any().item<bool>())return zero_loss(b.sem);
    auto fa=a.sem.flatten(1).index({Slice(),ma.flatten()}),fb=b.sem.flatten(1).index({Slice(),mb.flatten()});
    auto labs=h.logits(a.sem).detach().index({Slice(),ma.flatten()}).argmax(0);auto cls=torch::bincount(labs,{},h.text.size(0)).argmax().item<int64_t>();
    auto logb=h.logits(b.sem).index({Slice(),mb.flatten()}).transpose(0,1);
    return torch::norm(fb.mean(1)-fa.mean(1).detach())+torch::nn::functional::cross_entropy(logb,torch::full({logb.size(0)},cls,logb.options().dtype(torch::kLong)));
}
T contrastive(Heads& h,const Render& r,T region,int sample_limit) {
    auto labels=h.logits(r.sem).detach().argmax(0);auto reg=region.flatten();auto valid=reg>=0;
    for(int id=0;id<=reg.max().item<int>();++id){auto mask=reg==id;if(mask.any().item<bool>()){auto lab=labels.index({mask});labels.index_put_({mask},torch::bincount(lab,{},h.text.size(0)).argmax());}}
    auto idx=torch::nonzero(valid).flatten();if(idx.numel()<3)return zero_loss(r.sem);
    idx=idx.index_select(0,torch::randperm(idx.numel(),idx.options()).slice(0,0,std::min<int64_t>(sample_limit,idx.numel())));
    auto lab=labels.index_select(0,idx);auto z=norm(h.projection.matmul(r.sem.flatten(1).index_select(1,idx)),0).transpose(0,1);
    auto scores=z.matmul(z.t())/.2;auto diag=torch::eye(idx.numel(),scores.options().dtype(torch::kBool));
    auto positive=(lab.unsqueeze(0)==lab.unsqueeze(1)) & ~diag;auto negative=(lab.unsqueeze(0)!=lab.unsqueeze(1));auto anchors=positive.any(1)&negative.any(1);
    if(!anchors.any().item<bool>())return zero_loss(r.sem);
    scores=scores.masked_fill(diag,-1e4);auto logp=scores-torch::logsumexp(scores,1,true);
    return (-(logp*positive).sum(1)/positive.sum(1).clamp_min(1)).index({anchors}).mean();
}
T kl3d(std::shared_ptr<GaussianModel> m,const Render& a,const Render& b,T ma,T mb) {
    ma=erode(ma)&(a.ids>=0);mb=erode(mb)&(b.ids>=0);
    if(!ma.any().item<bool>()||!mb.any().item<bool>())return zero_loss(m->semantic_);
    auto ia=a.ids.index({ma}).to(torch::kLong),ib=b.ids.index({mb}).to(torch::kLong);
    // Pixel-weighted selection matches the paper; softmax ensures valid KL probabilities.
    auto q=torch::softmax(m->semantic_.index_select(0,ia).detach(),1).mean(0).clamp_min(1e-8);
    auto logp=torch::log_softmax(m->semantic_.index_select(0,ib),1);
    return (q.unsqueeze(0)*(q.log().unsqueeze(0)-logp)).sum(1).mean();
}
T local3d(std::shared_ptr<GaussianModel> m,int samples) {
    int64_t n=m->xyz_.size(0),s=std::min<int64_t>(samples,n),k=std::min<int64_t>(5,n-1);if(k<1||s<1)return zero_loss(m->semantic_);
    auto ids=torch::randint(n,{s},m->xyz_.options().dtype(torch::kLong));T near,dist;
    {torch::NoGradGuard g;auto d=torch::cdist(m->xyz_.index_select(0,ids).unsqueeze(0),m->xyz_.unsqueeze(0)).squeeze(0);std::tie(dist,near)=d.topk(k+1,1,false);dist=dist.slice(1,1,k+1);near=near.slice(1,1,k+1);}
    auto delta=m->semantic_.index_select(0,ids).unsqueeze(1)-m->semantic_.index_select(0,near.flatten()).view({s,k,32});
    return (torch::exp(-dist)*torch::norm(delta,2,2)).mean();
}
std::vector<std::pair<std::string,T*>> model_tensors(GaussianModel& m){return {{"xyz",&m.xyz_},{"dc",&m.features_dc_},{"rest",&m.features_rest_},{"scale",&m.scaling_},{"rotation",&m.rotation_},{"opacity",&m.opacity_},{"semantic",&m.semantic_},{"radii",&m.max_radii2D_},{"gradient",&m.xyz_gradient_accum_},{"denom",&m.denom_},{"exist",&m.exist_since_iter_}};}
std::vector<std::pair<std::string,T*>> head_tensors(Heads& h){return {{"decoder",&h.decoder},{"bias",&h.bias},{"classifier",&h.classifier},{"cbias",&h.cbias},{"projection",&h.projection},{"text",&h.text}};}
void vectors(GaussianModel& m){m.Tensor_vec_xyz_={m.xyz_};m.Tensor_vec_feature_dc_={m.features_dc_};m.Tensor_vec_feature_rest_={m.features_rest_};m.Tensor_vec_opacity_={m.opacity_};m.Tensor_vec_scaling_={m.scaling_};m.Tensor_vec_rotation_={m.rotation_};}
void save(const fs::path& dir,std::shared_ptr<GaussianModel> m,Heads& h,int step,const std::string& rng,const std::string& signature) {
    torch::NoGradGuard g;fs::create_directories(dir);m->savePly(dir/"point_cloud.ply");
    torch::serialize::OutputArchive a;for(auto& kv:model_tensors(*m))a.write(kv.first,kv.second->detach());for(auto& kv:head_tensors(h))a.write(kv.first,kv.second->detach());
    a.write("step",torch::tensor(step));a.write("sh",torch::tensor(m->active_sh_degree_));a.write("extent",torch::tensor(m->spatial_lr_scale_));
    a.write("added",torch::tensor(m->spc_added_));a.write("pruned",torch::tensor(m->spc_pruned_));
    a.write("cpu_rng",at::detail::getDefaultCPUGenerator().get_state());a.write("cuda_rng",at::cuda::detail::getDefaultCUDAGenerator().get_state());
    a.write("camera_rng",torch::from_blob((void*)rng.data(),{(int64_t)rng.size()},torch::kUInt8).clone());
    a.write("signature",torch::from_blob((void*)signature.data(),{(int64_t)signature.size()},torch::kUInt8).clone());
    torch::serialize::OutputArchive mo,ho;m->optimizer_->save(mo);h.optim->save(ho);a.write("optimizer",mo);a.write("head_optimizer",ho);
    a.save_to((dir/"model.pt.tmp").string());fs::rename(dir/"model.pt.tmp",dir/"model.pt");
    put(dir/"semantic.f32",m->semantic_);put(dir/"decoder.f32",h.decoder);put(dir/"decoder_bias.f32",h.bias);put(dir/"text.f32",h.text);
    Json::Value j;j["step"]=step;j["gaussians"]=Json::Int64(m->xyz_.size(0));j["semantic_dim"]=32;j["format"]="native LibTorch archive with Gaussian, semantic, heads, Adam, RNG; OmniGS-compatible PLY + semantic sidecar";writej(dir/"model.json",j);
}
int load(const fs::path& p,std::shared_ptr<GaussianModel> m,Heads& h,GaussianOptimizationParams opt,std::mt19937& rng,bool resume,const std::string& signature) {
    torch::serialize::InputArchive a;a.load_from(p.string(),torch::Device(torch::kCUDA));
    for(auto& kv:model_tensors(*m)){a.read(kv.first,*kv.second);*kv.second=kv.second->detach();if(kv.first!="radii"&&kv.first!="gradient"&&kv.first!="denom"&&kv.first!="exist")kv.second->requires_grad_();}
    for(auto& kv:head_tensors(h)){a.read(kv.first,*kv.second);*kv.second=kv.second->detach();if(kv.first!="text")kv.second->requires_grad_();}
    // LibTorch 2.0 InputArchive::read may reuse existing TensorImpl dtype.
    // A fresh tensor per archive entry is essential when scalar types differ.
    auto entry=[&](const std::string& key){T x;a.read(key,x);return x;};
    m->active_sh_degree_=entry("sh").item<int>();m->spatial_lr_scale_=entry("extent").item<float>();
    TORCH_CHECK(std::isfinite(m->spatial_lr_scale_) && m->spatial_lr_scale_>0 && m->spatial_lr_scale_<1e5,"Invalid checkpoint scene extent");
    auto radii=m->max_radii2D_,grad=m->xyz_gradient_accum_,denom=m->denom_;vectors(*m);m->trainingSetup(opt);m->max_radii2D_=radii;m->xyz_gradient_accum_=grad;m->denom_=denom;h.setup();
    if(resume){auto x=entry("signature").cpu().contiguous();TORCH_CHECK(std::string((char*)x.data_ptr(),x.numel())==signature,"Resume config/input signature mismatch");torch::serialize::InputArchive mo,ho;a.read("optimizer",mo);a.read("head_optimizer",ho);m->optimizer_->load(mo);h.optim->load(ho);
        T added,pruned;if(a.try_read("added",added))m->spc_added_=added.item<int64_t>();if(a.try_read("pruned",pruned))m->spc_pruned_=pruned.item<int64_t>();
        auto cr=at::detail::getDefaultCPUGenerator();cr.set_state(entry("cpu_rng").cpu());auto gr=at::cuda::detail::getDefaultCUDAGenerator();gr.set_state(entry("cuda_rng").cpu());
        x=entry("camera_rng").cpu().contiguous();std::istringstream ss(std::string((char*)x.data_ptr(),x.numel()));ss>>rng;}
    return entry("step").item<int>();
}
void export_view(fs::path out,std::string name,Render& r,Heads& h) {
    fs::create_directories(out);png(out/(name+"_rgb.png"),r.rgb);put(out/(name+"_rgb.f32"),r.rgb.detach().clamp(0,1));
    if(r.sem.defined()){put(out/(name+"_features.f16"),r.sem.to(torch::kFloat16));put(out/(name+"_logits.f16"),h.logits(r.sem).to(torch::kFloat16));put(out/(name+"_ids.i32"),r.ids);put(out/(name+"_weights.f32"),r.weights);}
}
std::string quote(const std::string& s){std::string o="'";for(char c:s)o+=c=='\''?"'\\''":std::string(1,c);return o+"'";}

int main(int argc,char** argv) try {
    TORCH_CHECK(argc==2,"Usage: spcgs_native request.json");auto c=readj(argv[1]);setenv("PF_MODE","0",1);setenv("SGS_MODE","0",1);unsetenv("PF_FIELD");
    torch::set_num_threads(2);int seed=c.get("seed",0).asInt();torch::manual_seed(seed);std::mt19937 rng(seed);
    int H=c["height"].asInt(),W=c["width"].asInt(),HS=c["semantic_height"].asInt(),WS=c["semantic_width"].asInt(),K=c["classes"].size();
    fs::path out=c["output"].asString();fs::create_directories(out);auto m=std::make_shared<GaussianModel>(3);Heads heads;GaussianOptimizationParams opt;
    int steps=c.get("iterations",0).asInt();opt.position_lr_max_steps_=std::max(steps,1);m->spatial_lr_scale_=c.get("extent",1.).asFloat();
    std::string action=c.get("action","train").asString(),signature=c.get("signature","").asString();int begin=0;
    if(c.isMember("archive"))begin=load(c["archive"].asString(),m,heads,opt,rng,c.get("resume",false).asBool(),signature);
    else {
        if(c.isMember("ply")) {m->loadPly(c["ply"].asString());m->max_radii2D_=torch::zeros({m->xyz_.size(0)},GPU);m->exist_since_iter_=torch::zeros({m->xyz_.size(0)},GPU.dtype(torch::kInt32));}
        else {auto p=raw(c["points"].asString(),{c["point_count"].asInt64(),6}).cpu().contiguous();std::map<point3D_id_t,Point3D> pts;for(int64_t i=0;i<p.size(0);++i){auto d=p.data_ptr<float>()+i*6;Point3D pt;pt.xyz_=Eigen::Vector3d(d[0],d[1],d[2]);pt.color_=Eigen::Vector3f(d[3],d[4],d[5]);pts[i]=pt;}m->createFromPcd(pts,m->spatial_lr_scale_);}
        if(c.isMember("text")){heads.init(c,K);m->semantic_=raw(c["initial_semantic"].asString(),{m->xyz_.size(0),32}).detach().requires_grad_();}
        for(auto p:{&m->xyz_,&m->features_dc_,&m->features_rest_,&m->opacity_,&m->scaling_,&m->rotation_})p->requires_grad_();
        vectors(*m);m->trainingSetup(opt);
    }
    if(action=="render") {torch::NoGradGuard g;
        if(c.isMember("query_text")){heads.text=raw(c["query_text"].asString(),{K,512});heads.classifier=torch::eye(K,GPU);heads.cbias=torch::zeros({K},GPU);}
        for(auto& v:c["views"]){auto r=render(m,v,H,W,HS,WS);export_view(out,v["name"].asString(),r,heads);}return 0;}
    TORCH_CHECK(m->semantic_.defined(),"Real CLIP supervision required for training");
    if(c.get("stage_reset",false).asBool()) {
        torch::NoGradGuard g;auto xyz=m->xyz_.clone();auto d=distCUDA2(xyz).clamp_min(1e-7);m->scaling_=torch::log(torch::sqrt(d)).unsqueeze(1).repeat({1,3}).requires_grad_();m->rotation_=torch::zeros({xyz.size(0),4},GPU);m->rotation_.index_put_({Slice(),0},1);m->rotation_.requires_grad_();m->opacity_=torch::full({xyz.size(0),1},std::log(.1/.9),GPU.requires_grad(true));m->features_rest_=torch::zeros_like(m->features_rest_).requires_grad_();m->active_sh_degree_=0;vectors(*m);m->trainingSetup(opt);heads.setup();begin=0;
    } else if(!c.get("resume",false).asBool())begin=0;
    auto save_now=[&](fs::path p,int i){std::ostringstream ss;ss<<rng;save(p,m,heads,i,ss.str(),signature);};
    if(begin==0)save_now(out/"initial",0);
    std::ofstream log(out/"losses.jsonl",begin?std::ios::app:std::ios::out);auto views=c["views"];TORCH_CHECK(!views.empty(),"No training views");
    int isp_every=c.get("isp_every",100).asInt();
    for(int step=begin+1;step<=steps;++step) {
        auto v=views[static_cast<int>(rng()%views.size())];m->updateLearningRate(step);if(step%1000==0)m->oneUpShDegree();
        auto r=render(m,v,H,W,HS,WS);auto gt=image(v["image"].asString(),H,W);
        auto lc=.8*torch::abs(r.rgb-gt).mean()+.2*(1-loss_utils::ssim(r.rgb,gt,torch::kCUDA));
        T lcos,lce;auto ls=semantic_loss(heads,r.sem,v,HS,WS,&lcos,&lce),lg=zero_loss(ls),ll=local3d(m,c.get("local_samples",64).asInt());
        if(c["generated_views"].size()){auto gv=c["generated_views"][step%c["generated_views"].size()];auto gr=render(m,gv,H,W,HS,WS);lg=semantic_loss(heads,gr.sem,gv,HS,WS);}
        auto li=zero_loss(ls),ln=zero_loss(ls),lk=zero_loss(ls);Json::Value row;row["isp_valid_pairs"]=0;
        bool use2=c.get("spc2d",false).asBool(),use3=c.get("spc3d",false).asBool();
        if((use2||use3) && step%isp_every==0) {
            std::mt19937 isp_rng(seed+step);
            fs::path snap=out/"isp"/std::to_string(step);fs::create_directories(snap);Json::Value request;request["height"]=HS;request["width"]=WS;request["seed"]=seed+step;request["output"]=snap.string();request["root"]=c["root"];request["sam_config"]=c["sam_config"];request["sam_checkpoint"]=c["sam_checkpoint"];
            png(snap/"0.png",gt);request["images"].append((snap/"0.png").string());request["cameras"].append(v);
            std::vector<Render> pseudos;std::vector<std::pair<double,int>> near;
            for(int i=0;i<(int)views.size();++i){double d=0;for(int k=0;k<3;++k)d+=std::pow(views[i]["center"][k].asDouble()-v["center"][k].asDouble(),2);if(d>1e-10)near.emplace_back(d,i);}std::sort(near.begin(),near.end());TORCH_CHECK(!near.empty(),"Need >=2 distinct training centers for ISP");
            for(int p=0;p<2;++p){auto pv=v;auto nv=views[near[std::min<int>(p,near.size()-1)].second];std::normal_distribution<double> noise(0,c.get("pose_noise_fraction",.02).asDouble()*std::sqrt(near[0].first));std::uniform_real_distribution<double> lerp(.1,.3);double t=lerp(isp_rng);
                for(int k=0;k<3;++k)pv["center"][k]=(1-t)*v["center"][k].asDouble()+t*nv["center"][k].asDouble()+noise(isp_rng);
                auto pr=render(m,pv,H,W,HS,WS);png(snap/(std::to_string(p+1)+".png"),pr.rgb);request["images"].append((snap/(std::to_string(p+1)+".png")).string());request["cameras"].append(pv);export_view(snap,"pseudo_"+std::to_string(p),pr,heads);pseudos.push_back(pr);}
            writej(snap/"request.json",request);std::string cmd=quote(c["python"].asString())+" "+quote(c["sam_worker"].asString())+" "+quote((snap/"request.json").string())+" > "+quote((snap/"sam.log").string())+" 2>&1";
            TORCH_CHECK(std::system(cmd.c_str())==0,"SAM2 failed; see ",(snap/"sam.log").string());auto masks=raw(snap/"masks.u8",{3,HS,WS},torch::kUInt8).to(torch::kBool);auto regions=raw(snap/"regions.i32",{HS,WS},torch::kInt32);
            auto status=readj(snap/"outputs.json");
            for(int p=0;p<2;++p)if(status["valid"][p].asBool()){row["isp_valid_pairs"]=row["isp_valid_pairs"].asInt()+1;if(use2)li=li+inter(heads,r,pseudos[p],masks[0],masks[p+1]);if(use3)lk=lk+kl3d(m,r,pseudos[p],masks[0],masks[p+1]);}
            if(use2)ln=contrastive(heads,r,regions,c.get("contrast_samples",256).asInt());export_view(snap,"real",r,heads);
        }
        float ramp=std::min(1.f,float(step)/std::max(1,c.get("spc_ramp",4000).asInt()));
        auto loss=lc+ls+lg+c.get("local_weight",1.).asFloat()*ll+ramp*(li+ln+lk);
        TORCH_CHECK(torch::isfinite(loss).item<bool>(),"Nonfinite loss at ",step);loss.backward();
        for(auto& kv:model_tensors(*m))if(kv.second->grad().defined())TORCH_CHECK(torch::isfinite(kv.second->grad()).all().item<bool>(),"Nonfinite gradient ",kv.first);
        row["semantic_grad_norm"]=m->semantic_.grad().norm().item<double>();row["xyz_grad_norm"]=m->xyz_.grad().norm().item<double>();
        row["semantic_cosine"]=lcos.item<double>();row["semantic_ce"]=lce.item<double>();row["decoder_grad_norm"]=heads.decoder.grad().norm().item<double>();
        m->optimizer_->step();heads.optim->step();
        {torch::NoGradGuard g;auto visible=r.visible;auto screen=r.screen;m->addDensificationStats(screen,visible);m->max_radii2D_.index_put_({visible},torch::maximum(m->max_radii2D_.index({visible}),r.radii.index({visible}).to(torch::kFloat32)));
            int from=c.get("densify_from",500).asInt(),until=c.get("densify_until",15000).asInt(),interval=c.get("densify_every",100).asInt();
            if(step>=from&&step<=until&&step%interval==0&&m->xyz_.size(0)<c.get("max_gaussians",1000000).asInt())m->densifyAndPrune(c.get("densify_threshold",.0002).asFloat(),.005,m->spatial_lr_scale_,0,false);
            int every=c.get("ogr_every",3000).asInt();if(c.get("ogr",false).asBool()&&step%every==0){put(out/"ogr_xyz.f32",m->xyz_);Json::Value oj;oj["xyz"]=(out/"ogr_xyz.f32").string();oj["output"]=(out/"ogr_mask.u8").string();oj["radius"]=c.get("ogr_radius",1.);oj["min_neighbors"]=c.get("ogr_min_neighbors",5);writej(out/"ogr_request.json",oj);std::string cmd=quote(c["python"].asString())+" "+quote(c["geometry_worker"].asString())+" ogr "+quote((out/"ogr_request.json").string());TORCH_CHECK(std::system(cmd.c_str())==0,"OGR worker failed");auto mask=raw(out/"ogr_mask.u8",{m->xyz_.size(0)},torch::kUInt8).to(torch::kBool);row["ogr_removed"]=mask.sum().item<int>();TORCH_CHECK((~mask).sum().item<int>()>=4,"OGR would remove entire cloud: adjust radius for coordinate scale");m->prunePoints(mask);}
            int reset=c.get("opacity_reset_every",3000).asInt();if(reset>0&&step%reset==0&&step<until)m->resetOpacity();
        }
        m->optimizer_->zero_grad(true);heads.optim->zero_grad(true);
        row["step"]=step;row["rgb"]=lc.item<double>();row["clip"]=ls.item<double>();row["generated_semantic"]=lg.item<double>();row["local3d"]=ll.item<double>();row["inter"]=li.item<double>();row["intra"]=ln.item<double>();row["spc3d"]=lk.item<double>();row["lambda"]=ramp;row["total"]=loss.item<double>();row["gaussians"]=Json::Int64(m->xyz_.size(0));row["added_cumulative"]=Json::Int64(m->spc_added_);row["pruned_cumulative"]=Json::Int64(m->spc_pruned_);
        Json::StreamWriterBuilder compact;compact["indentation"]="";log<<Json::writeString(compact,row)<<"\n";log.flush();
        if(step%10==0||step==1)std::cout<<"step="<<step<<" loss="<<loss.item<float>()<<" gaussians="<<m->xyz_.size(0)<<" semantic_grad="<<row["semantic_grad_norm"]<<std::endl;
        if(step%c.get("save_every",500).asInt()==0||step==steps)save_now(out/("checkpoint_"+std::to_string(step)),step);
    }
    save_now(out/"final",steps);return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
