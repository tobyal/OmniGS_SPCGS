#include "include/gaussian_mapper.h"
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <sstream>
#include <cstdlib>

void GaussianMapper::saveTrainingCheckpoint(){
 torch::NoGradGuard guard;torch::serialize::OutputArchive archive;auto& m=*gaussians_;
 auto write=[&](const char* name,torch::Tensor t){archive.write(name,t.detach().cpu());};
 write("xyz",m.xyz_);write("dc",m.features_dc_);write("rest",m.features_rest_);write("scale",m.scaling_);write("rotation",m.rotation_);write("opacity",m.opacity_);
 write("radii",m.max_radii2D_);write("gradient",m.xyz_gradient_accum_);write("denom",m.denom_);write("exist",m.exist_since_iter_);
 torch::serialize::OutputArchive optimizer;m.optimizer_->save(optimizer);archive.write("optimizer",optimizer);
 write("cpu_rng",at::detail::getDefaultCPUGenerator().get_state());write("cuda_rng",at::cuda::detail::getDefaultCUDAGenerator().get_state());
 Json::Value meta;meta["iteration"]=iteration_;meta["sh"]=m.active_sh_degree_;meta["spatial_lr_scale"]=m.spatial_lr_scale_;meta["ema"]=ema_loss_for_log_;
 meta["shuffle_index"]=Json::UInt64(kfid_shuffle_idx_);meta["shuffled"]=kfid_shuffled_;
 for(auto id:kfid_shuffle_)meta["shuffle"].append(Json::UInt64(id));
 for(auto& item:scene_->keyframes()){Json::Value row;row["id"]=Json::UInt64(item.first);row["remaining"]=item.second->remaining_times_of_use_;row["used"]=kfs_used_times_[item.first];meta["cameras"].append(row);}
 std::ostringstream rng;rng<<camera_rng_;meta["camera_rng"]=rng.str();
 for(auto key:{"PF_MODE","SGS_MODE","SGS_WINDOW","SGS_SEED"})meta[key]=std::getenv(key)?std::getenv(key):"";
 std::string json=Json::writeString(Json::StreamWriterBuilder(),meta);auto bytes=torch::from_blob((void*)json.data(),{(int64_t)json.size()},torch::kUInt8).clone();archive.write("metadata",bytes);
 auto temporary=result_dir_/"resume.pt.tmp",final=result_dir_/"resume.pt";archive.save_to(temporary.string());std::filesystem::rename(temporary,final);
 std::cout<<"CHECKPOINT step="<<iteration_<<" file="<<final.string()<<std::endl;
}

void GaussianMapper::loadTrainingCheckpoint(){
 auto path=result_dir_/"resume.pt";if(!std::filesystem::exists(path))return;
 torch::NoGradGuard guard;torch::serialize::InputArchive archive;archive.load_from(path.string(),torch::Device(torch::kCUDA));
 torch::Tensor bytes;archive.read("metadata",bytes);bytes=bytes.cpu().contiguous();std::string json((char*)bytes.data_ptr<uint8_t>(),bytes.numel());std::istringstream stream(json);Json::Value meta;stream>>meta;
 for(auto key:{"PF_MODE","SGS_MODE","SGS_WINDOW","SGS_SEED"})TORCH_CHECK(meta[key].asString()==(std::getenv(key)?std::getenv(key):""),"Checkpoint configuration mismatch ",key);
 auto& m=*gaussians_;
 auto read=[&](const char* name,bool train){torch::Tensor t;archive.read(name,t);return t.to(torch::kCUDA).detach().set_requires_grad(train);};
 m.xyz_=read("xyz",true);m.features_dc_=read("dc",true);m.features_rest_=read("rest",true);m.scaling_=read("scale",true);m.rotation_=read("rotation",true);m.opacity_=read("opacity",true);
 m.Tensor_vec_xyz_={m.xyz_};m.Tensor_vec_feature_dc_={m.features_dc_};m.Tensor_vec_feature_rest_={m.features_rest_};m.Tensor_vec_scaling_={m.scaling_};m.Tensor_vec_rotation_={m.rotation_};m.Tensor_vec_opacity_={m.opacity_};
 m.spatial_lr_scale_=meta["spatial_lr_scale"].asFloat();m.active_sh_degree_=meta["sh"].asInt();m.trainingSetup(opt_params_);
 m.max_radii2D_=read("radii",false);m.xyz_gradient_accum_=read("gradient",false);m.denom_=read("denom",false);m.exist_since_iter_=read("exist",false);
 torch::serialize::InputArchive optimizer;archive.read("optimizer",optimizer);m.optimizer_->load(optimizer);
 auto cpu=at::detail::getDefaultCPUGenerator();auto cuda=at::cuda::detail::getDefaultCUDAGenerator();cpu.set_state(read("cpu_rng",false).cpu());cuda.set_state(read("cuda_rng",false).cpu());
 iteration_=meta["iteration"].asInt();ema_loss_for_log_=meta["ema"].asFloat();kfid_shuffle_idx_=meta["shuffle_index"].asUInt64();kfid_shuffled_=meta["shuffled"].asBool();kfid_shuffle_.clear();
 for(auto& id:meta["shuffle"])kfid_shuffle_.push_back(id.asUInt64());
 for(auto& row:meta["cameras"]){auto id=row["id"].asUInt64();scene_->keyframes().at(id)->remaining_times_of_use_=row["remaining"].asInt();kfs_used_times_[id]=row["used"].asInt();}
 std::istringstream rng(meta["camera_rng"].asString());rng>>camera_rng_;
 std::cout<<"RESUMED step="<<iteration_<<" optimizer_and_rng=restored"<<std::endl;
}
