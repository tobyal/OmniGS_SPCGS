// Isolated numerical tests only: these fixtures are not experiment data.
#define main spcgs_training_main
#include "../code/examples/spcgs_native.cpp"
#undef main
int main(int argc,char**argv) try {
    TORCH_CHECK(argc==2,"output directory");fs::path out=argv[1];fs::create_directories(out);
    setenv("PF_MODE","0",1);setenv("SGS_MODE","0",1);unsetenv("PF_FIELD");torch::manual_seed(4);torch::set_num_threads(2);
    auto m=std::make_shared<GaussianModel>(3);std::map<point3D_id_t,Point3D> points;
    for(int i=0;i<4;++i){Point3D p;p.xyz_=Eigen::Vector3d(0,0,2+i);p.color_=Eigen::Vector3f(.2,.3,.4);points[i]=p;}
    m->createFromPcd(points,1);{
        torch::NoGradGuard g;m->scaling_=torch::log(torch::tensor({.3,.45,.6,.75},GPU)).unsqueeze(1).repeat({1,3}).requires_grad_();
        auto op=torch::tensor({.05,.8,.1,.1},GPU).unsqueeze(1);m->opacity_=torch::log(op/(1-op)).requires_grad_();
        m->semantic_=torch::arange(128,GPU).view({4,32}).div(128).requires_grad_();
    } vectors(*m);GaussianOptimizationParams opt;m->trainingSetup(opt);
    Json::Value cam;for(int i=0;i<3;++i){cam["center"].append(0);for(int j=0;j<3;++j)cam["rotation"][i].append(i==j?1:0);}
    auto r=render(m,cam,32,64,32,64);auto objective=r.sem[0].sum();objective.backward();float analytical=m->semantic_.grad()[1][0].item<float>();
    float eps=1e-3,plus,minus;{torch::NoGradGuard g;m->semantic_.index_put_({1,0},m->semantic_[1][0]+eps);plus=render(m,cam,32,64,32,64).sem[0].sum().item<float>();m->semantic_.index_put_({1,0},m->semantic_[1][0]-2*eps);minus=render(m,cam,32,64,32,64).sem[0].sum().item<float>();m->semantic_.index_put_({1,0},m->semantic_[1][0]+eps);}
    float numerical=(plus-minus)/(2*eps),relative=std::abs(analytical-numerical)/std::max(1.f,std::abs(analytical));TORCH_CHECK(relative<.003,"Semantic finite difference mismatch");
    // The second point is farther away but contributes more alpha*T than the first.
    int dominant=r.ids[16][32].item<int>();TORCH_CHECK(dominant==1,"Expected actual maximum-weight point, not nearest point");
    auto f=camera(cam,32,64);GaussianPipelineParams pipe;T bg=torch::zeros({3},GPU),color=m->semantic_.slice(1,0,3).contiguous();
    float chunk_error;
    {torch::NoGradGuard g;auto direct=std::get<0>(GaussianRenderer::renderLonlat(f,32,64,m,pipe,bg,color,1.f,true));chunk_error=(direct-r.sem.slice(0,0,3)).abs().max().item<float>();TORCH_CHECK(chunk_error<1e-6,"Semantic compositor differs from RGB override");}
    float gg=m->xyz_.grad().abs().sum().item<float>();TORCH_CHECK(std::isfinite(gg)&&gg>0,"No geometry gradient from semantic rendering");
    // Exercise the native clone/split and prune operations with semantic optimizer state.
    m->optimizer_->step();m->optimizer_->zero_grad(true);
    {torch::NoGradGuard g;auto before=m->semantic_.clone();T grads=torch::ones({4,1},GPU);m->setPercentDense(10);m->densifyAndClone(grads,0,1);TORCH_CHECK(m->semantic_.size(0)==8 && (m->semantic_.slice(0,4,8)-before).abs().max().item<float>()==0,"Clone ancestry mismatch");
    T mask=torch::tensor({false,true,false,false,false,false,false,false},torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));m->prunePoints(mask);TORCH_CHECK(m->semantic_.size(0)==7,"Prune semantic count mismatch");
    m->setPercentDense(.001);grads=torch::ones({7,1},GPU);m->densifyAndSplit(grads,0,1);TORCH_CHECK(m->semantic_.size(0)==14,"Split semantic count mismatch");
    auto& param=m->optimizer_->param_groups()[6].params()[0];auto key=c10::guts::to_string(param.unsafeGetTensorImpl());auto& state=static_cast<torch::optim::AdamParamState&>(*m->optimizer_->state().at(key));TORCH_CHECK(state.exp_avg().sizes()==m->semantic_.sizes(),"Adam state does not follow densification");}
    Heads h;h.decoder=torch::ones({512,32},GPU.requires_grad(true));h.bias=torch::zeros({512},GPU.requires_grad(true));h.classifier=torch::eye(3,GPU).requires_grad_();h.cbias=torch::zeros({3},GPU.requires_grad(true));h.projection=torch::eye(32,GPU).requires_grad_();h.text=torch::ones({3,512},GPU);h.setup();std::mt19937 rng(17);std::ostringstream ss;ss<<rng;
    m->spatial_lr_scale_=15.6794f;save(out/"checkpoint",m,h,23,ss.str(),"selftest");
    auto restored=std::make_shared<GaussianModel>(3);Heads rh;std::mt19937 rrng;int step=load(out/"checkpoint/model.pt",restored,rh,opt,rrng,true,"selftest");TORCH_CHECK(step==23 && restored->spatial_lr_scale_==m->spatial_lr_scale_ && rrng()==rng(),"Checkpoint metadata or camera RNG mismatch");
    for(auto& kv:model_tensors(*m)){T other;for(auto& rv:model_tensors(*restored))if(kv.first==rv.first)other=*rv.second;TORCH_CHECK(torch::equal(*kv.second,other),"Reload tensor differs ",kv.first);}
    float reload_error;{torch::NoGradGuard g;reload_error=(render(m,cam,32,64,32,64).rgb-render(restored,cam,32,64,32,64).rgb).abs().max().item<float>();TORCH_CHECK(reload_error==0,"Reload render mismatch");}
    Json::Value result;result["status"]="passed";result["semantic_analytic"]=analytical;result["semantic_finite_difference"]=numerical;result["relative_error"]=relative;result["max_contributor_id"]=dominant;result["nearest_gaussian_id"]=0;result["geometry_grad_from_semantics"]=gg;result["rgb_semantic_first_chunk_max_error"]=chunk_error;result["semantic_adam_clone_split_prune"]="passed";result["reload_rgb_max_error"]=reload_error;result["metadata_and_rng_reload"]="passed";writej(out/"native_gradient_checks.json",result);std::cout<<result<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
