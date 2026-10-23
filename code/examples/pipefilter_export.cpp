/**
 * This file is part of OmniGS
 *
 * Copyright (C) 2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * OmniGS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OmniGS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with OmniGS.
 * If not, see <http://www.gnu.org/licenses/>.
 * 
 * OmniGS is Derivative Works of Gaussian Splatting.
 * Its usage should not break the terms in LICENSE.md.
 */

#include <unordered_map>
#include <filesystem>
#include <fstream>

#include <torch/torch.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "third_party/colmap/utils/endian.h"
#include "third_party/tinyply/tinyply.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

void readViewsOpenMVG(
    std::shared_ptr<GaussianMapper> pMapper,
    const std::filesystem::path& path)
{
    std::ifstream in_stream;
    in_stream.open(path);
    if (!in_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + path.string());

    Json::Value json_root;
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!Json::parseFromStream(builder, in_stream, &json_root, &errs))
        throw std::runtime_error(errs);

    // Cameras
    const std::size_t num_cameras = json_root["intrinsics"].size();
    int i = 1;
    for (const auto& intr : json_root["intrinsics"]) {
        class Camera camera;
        camera.camera_id_ = intr["key"].asUInt();

        std::cout << "                                                                             \r";
        std::cout.flush();
        std::cout << "Loading camera " << camera.camera_id_ << ", " << i << "/" << num_cameras << "\r";
        std::cout.flush();
        i++;

        camera.setModelId(Camera::CameraModelType::LONLAT);
        camera.width_ = intr["value"]["ptr_wrapper"]["data"]["value0"]["width"].asUInt64();
        camera.height_ = intr["value"]["ptr_wrapper"]["data"]["value0"]["height"].asUInt64();

        camera.num_gaus_pyramid_sub_levels_ = pMapper->num_gaus_pyramid_sub_levels_;
        camera.gaus_pyramid_width_.resize(pMapper->num_gaus_pyramid_sub_levels_);
        camera.gaus_pyramid_height_.resize(pMapper->num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < pMapper->num_gaus_pyramid_sub_levels_; ++l) {
            camera.gaus_pyramid_width_[l] = camera.width_ * pMapper->kf_gaus_pyramid_factors_[l];
            camera.gaus_pyramid_height_[l] = camera.height_ * pMapper->kf_gaus_pyramid_factors_[l];
        }

        cv::Mat K = (
            cv::Mat_<float>(3, 3)
                << camera.width_, 0.f, camera.width_ / 2.0f,
                    0.f, camera.height_, camera.height_ / 2.0f,
                    0.f, 0.f, 1.f
        );
        camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, pMapper->isdoingGausPyramidTraining());

        pMapper->undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(
                camera.undistort_mask, pMapper->device_type_);

        cv::Mat viewer_main_undistort_mask;
        int viewer_image_height_main_ = camera.height_ * pMapper->rendered_image_viewer_scale_main_;
        int viewer_image_width_main_ = camera.width_ * pMapper->rendered_image_viewer_scale_main_;
        cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                    cv::Size(viewer_image_width_main_, viewer_image_height_main_));
        pMapper->viewer_main_undistort_mask_[camera.camera_id_] =
            tensor_utils::cvMat2TorchTensor_Float32(
                viewer_main_undistort_mask, pMapper->device_type_);

        if (!pMapper->viewer_camera_id_set_) {
            pMapper->viewer_camera_id_ = camera.camera_id_;
            pMapper->viewer_camera_id_set_ = true;
        }

        pMapper->scene_->addCamera(camera);
    }
    std::cout << std::endl;

    // Frames
    std::unordered_map<std::size_t, Json::Value> views;
    std::unordered_map<std::size_t, Json::Value> extrinsics;
    // Read json data
    for (const auto& json_view : json_root["views"])
        views.emplace(json_view["key"].asUInt64(), json_view["value"]);
    for (const auto& json_extrinsic : json_root["extrinsics"])
        extrinsics.emplace(json_extrinsic["key"].asUInt64(), json_extrinsic["value"]);
    std::filesystem::path image_dir(json_root["root_path"].asString());
    std::size_t num_frames = views.size();

    i = 1;
    for (const auto& view : views) {
        std::size_t fid = view.first;
        auto json_view = view.second;
        auto json_extrinsic = extrinsics[json_view["ptr_wrapper"]["data"]["id_pose"].asUInt64()];

        std::cout << "                                                                             \r";
        std::cout.flush();
        std::cout << "Loading frame " << fid << ", " << i << "/" << num_frames << "\r";
        std::cout.flush();
        i++;

        std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(fid);
        pkf->img_filename_ = json_view["ptr_wrapper"]["data"]["filename"].asString();
        pkf->zfar_ = pMapper->z_far_;
        pkf->znear_ = pMapper->z_near_;

        // Pose
        Eigen::Vector3f ccw;
        ccw.x() = json_extrinsic["center"][0].asFloat();
        ccw.y() = json_extrinsic["center"][1].asFloat();
        ccw.z() = json_extrinsic["center"][2].asFloat();
        Eigen::Matrix3f Rcw;
        Rcw(0, 0) = json_extrinsic["rotation"][0][0].asFloat();
        Rcw(0, 1) = json_extrinsic["rotation"][0][1].asFloat();
        Rcw(0, 2) = json_extrinsic["rotation"][0][2].asFloat();
        Rcw(1, 0) = json_extrinsic["rotation"][1][0].asFloat();
        Rcw(1, 1) = json_extrinsic["rotation"][1][1].asFloat();
        Rcw(1, 2) = json_extrinsic["rotation"][1][2].asFloat();
        Rcw(2, 0) = json_extrinsic["rotation"][2][0].asFloat();
        Rcw(2, 1) = json_extrinsic["rotation"][2][1].asFloat();
        Rcw(2, 2) = json_extrinsic["rotation"][2][2].asFloat();
        Eigen::Vector3f tcw = -Rcw * ccw;
        pkf->setPose(
            Eigen::Quaterniond(Rcw.cast<double>()),
            tcw.cast<double>());

        try {
            // Camera
            unsigned int camera_id = json_view["ptr_wrapper"]["data"]["id_intrinsic"].asUInt();
            Camera& camera = pMapper->scene_->cameras_.at(camera_id);
            std::size_t width = json_view["ptr_wrapper"]["data"]["width"].asUInt64();
            std::size_t height = json_view["ptr_wrapper"]["data"]["height"].asUInt64();
            assert(width == camera.width_ && height == camera.height_);
            pkf->setCameraParams(camera);
            // Transformations
            pkf->computeTransformTensors();
            // Image
            std::filesystem::path image_path = image_dir / pkf->img_filename_;
            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            cv::cvtColor(image, image, CV_BGR2RGB);
            image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
            cv::Mat imgRGB_undistorted;
            camera.undistortImage(image, imgRGB_undistorted);
            pkf->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image, pMapper->device_type_);
            // Prepare for gaussian-pyramid-based training
            pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
            pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
            pkf->gaus_pyramid_times_of_use_ = pMapper->kf_gaus_pyramid_times_of_use_;

            pMapper->scene_->addKeyframe(pkf, &pMapper->kfid_shuffled_);
        }
        catch (std::out_of_range) {
            throw std::runtime_error("[GaussianMapper::keyframesFromJson]KeyFrame Camera not found!");
        }
    }
    std::cout << std::endl;
}

void readScene(std::shared_ptr<GaussianMapper> pMapper)
{
    auto& model_params = pMapper->getGaussianModelParams();
    auto scene = pMapper->scene_;

    std::filesystem::path frames_file = model_params.source_path_ / "openMVG/data_openmvg_test.json";

    readViewsOpenMVG(pMapper, frames_file);
}

#include "include/gaussian_renderer.h"
#include "cuda_rasterizer/pipe_filter.cuh"
#include "cuda_rasterizer/structure_sort.h"
#include <cuda_runtime.h>
#include <set>
#include <chrono>
#include "include/loss_utils.h"
using torch::indexing::Slice;
void writeTensor(const std::filesystem::path& path,torch::Tensor x){
 x=x.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();std::ofstream f(path,std::ios::binary);f.write((char*)x.data_ptr<float>(),x.numel()*4);TORCH_CHECK(f.good(),"Failed output ",path.string());
}
Json::Value asJson(torch::Tensor x){x=x.detach().cpu().to(torch::kFloat32).flatten().contiguous();Json::Value a(Json::arrayValue);for(int64_t i=0;i<x.numel();++i)a.append(x.data_ptr<float>()[i]);return a;}
int main(int argc,char** argv){
 TORCH_CHECK(argc==6,"config dataset checkpoint request.json output");
 std::filesystem::path out(argv[5]);std::filesystem::create_directories(out);
 Json::Value req;{std::ifstream f(argv[4]);f>>req;TORCH_CHECK(f.good()||f.eof(),"Bad request");}
 int W=req.get("width",2048).asInt(),H=req.get("height",1024).asInt();
 auto mapper=std::make_shared<GaussianMapper>(std::filesystem::path(argv[1]),out,0,torch::kCUDA);
 mapper->setSensorType(MONOCULAR);mapper->setDatasetSourcePath(argv[2]);
 auto camera_json=req.isMember("camera_json")?std::filesystem::path(req["camera_json"].asString()):std::filesystem::path(argv[2])/"openMVG"/("data_openmvg_"+req.get("split","test").asString()+".json");
 readViewsOpenMVG(mapper,camera_json);mapper->loadPly(std::filesystem::path(argv[3]));
 torch::NoGradGuard guard;auto model=mapper->gaussians_;auto opt=model->xyz_.options();
 GaussianPipelineParams pipeline;pipeline.convert_SHs_=false;pipeline.compute_cov3D_=false;
 std::set<std::string> selected;for(auto& n:req["views"])selected.insert(n.asString());
 bool moments=req.get("moments",false).asBool(),diag=req.get("diagnostics",false).asBool(),bench=req.get("benchmark",false).asBool();
 for(auto name:{"rgb","moments","diagnostics"})std::filesystem::create_directories(out/name);
 auto render=[&](std::shared_ptr<GaussianKeyframe> frame,torch::Tensor color,bool over){auto bg=torch::zeros({3},opt);return std::get<0>(GaussianRenderer::renderLonlat(frame,H,W,model,pipeline,bg,color,1.f,over));};
 auto cov=diag?model->getCovarianceActivation():torch::Tensor();
 int64_t n=model->xyz_.size(0),stride=std::max<int64_t>(1,(n+8191)/8192);
 auto ids=torch::arange(0,n,stride,torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
 if(diag&&req.isMember("diagnostic_ids")){
  std::ifstream f(req["diagnostic_ids"].asString(),std::ios::binary|std::ios::ate);std::streamsize bytes=static_cast<std::streamsize>(f.tellg());TORCH_CHECK(bytes>0&&bytes%8==0,"Invalid diagnostic IDs");f.seekg(0);std::vector<int64_t> selected_ids(size_t(bytes)/8);f.read((char*)selected_ids.data(),bytes);ids=torch::from_blob(selected_ids.data(),{(int64_t)selected_ids.size()},torch::kLong).clone().cuda();TORCH_CHECK(ids.min().item<int64_t>()>=0&&ids.max().item<int64_t>()<n,"Diagnostic ID out of range");
 }
 Json::Value meta;meta["checkpoint"]=argv[3];meta["width"]=W;meta["height"]=H;meta["gaussians"]=Json::Int64(n);meta["request"]=req;
 for(auto& entry:mapper->scene_->keyframes()){
  auto frame=entry.second;auto name=std::filesystem::path(frame->img_filename_).stem().string();
  if(!selected.empty()&&!selected.count(name))continue;
  Json::Value row;row["name"]=name;row["camera_id"]=Json::UInt64(entry.first);row["world_view_transform"]=asJson(frame->world_view_transform_);row["reference_filename"]=frame->img_filename_;
  auto sorting=diag?torch::zeros({4,H,W},opt):torch::Tensor();
  if(diag)structuresort::setDiagnostics(sorting.data_ptr<float>());
  auto rgb=render(frame,torch::empty({0},opt),false);
  structuresort::setDiagnostics(nullptr);
  if(diag){
    writeTensor(out/"diagnostics"/(name+"_sorting.f32"),sorting);
    row["sorting_active_pixel_fraction"]=(sorting[0]>0).to(torch::kFloat32).mean().item<float>();
    row["sorting_reordered_pixel_fraction"]=(sorting[2]>0).to(torch::kFloat32).mean().item<float>();
    row["sorting_mean_valid_candidates"]=sorting[1].mean().item<float>();
  }
  TORCH_CHECK(torch::isfinite(rgb).all().item<bool>(),"Nonfinite render");
  writeTensor(out/"rgb"/(name+".f32"),rgb.clamp(0,1));
  auto clamped_rgb=rgb.clamp(0,1);auto im=tensor_utils::torchTensor2CvMat_Float32(clamped_rgb);cv::cvtColor(im,im,CV_RGB2BGR);im.convertTo(im,CV_8UC3,255);cv::imwrite((out/"rgb"/(name+".png")).string(),im);
  if(moments||diag){auto tr=frame->world_view_transform_;auto xyz=model->xyz_.matmul(tr.index({Slice(0,3),Slice(0,3)}))+tr.index({3,Slice(0,3)});auto r=torch::norm(xyz,2,1);auto image=render(frame,torch::stack({r,r*r,torch::ones_like(r)},1),true);writeTensor(out/"moments"/(name+".f32"),image);}
  if(diag){auto xyz=model->xyz_.index_select(0,ids).contiguous(),co=cov.index_select(0,ids).contiguous();auto d=torch::empty({ids.numel(),24},opt);pipefilter::diagnostics(ids.numel(),xyz.data_ptr<float>(),co.data_ptr<float>(),frame->world_view_transform_.contiguous().data_ptr<float>(),W,H,d.data_ptr<float>());TORCH_CHECK(torch::isfinite(d).all().item<bool>(),"Nonfinite diagnostics");writeTensor(out/"diagnostics"/(name+".f32"),d);row["diagnostic_count"]=Json::Int64(ids.numel());}
  if(bench){
   int warmup=req.get("warmup",10).asInt(),repeats=req.get("repeats",100).asInt();
   for(int k=0;k<warmup;++k)render(frame,torch::empty({0},opt),false);
   cudaDeviceSynchronize();c10::cuda::CUDACachingAllocator::resetPeakStats(0);
   cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);cudaEventRecord(begin);
   for(int k=0;k<repeats;++k)render(frame,torch::empty({0},opt),false);
   cudaEventRecord(end);cudaEventSynchronize(end);float ms=0;cudaEventElapsedTime(&ms,begin,end);cudaEventDestroy(begin);cudaEventDestroy(end);
   auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(0);int agg=static_cast<int>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE);
   row["render_ms"]=ms/repeats;row["fps"]=1000.f*repeats/ms;row["peak_allocated_bytes"]=Json::Int64(stats.allocated_bytes[agg].peak);row["peak_reserved_bytes"]=Json::Int64(stats.reserved_bytes[agg].peak);
   if(req.get("training_step",false).asBool()){
    // An in-memory fixed-count microbenchmark, never saved over the checkpoint.
    torch::AutoGradMode enable_grad(true);GaussianOptimizationParams train_params;
    model->trainingSetup(train_params);
    auto gt=frame->original_image_;
    TORCH_CHECK(gt.size(1)==H&&gt.size(2)==W,"Training benchmark must use native resolution");
    auto step=[&](){auto image=render(frame,torch::empty({0},opt),false);auto loss=.8*loss_utils::l1_loss(image,gt)+.2*(1.-loss_utils::ssim(image,gt,torch::kCUDA));loss.backward();model->optimizer_->step();model->optimizer_->zero_grad(true);};
    for(int k=0;k<5;++k)step();cudaDeviceSynchronize();c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    cudaEventCreate(&begin);cudaEventCreate(&end);cudaEventRecord(begin);
    int updates=req.get("training_repeats",20).asInt();for(int k=0;k<updates;++k)step();
    cudaEventRecord(end);cudaEventSynchronize(end);cudaEventElapsedTime(&ms,begin,end);cudaEventDestroy(begin);cudaEventDestroy(end);
    row["optimization_step_ms"]=ms/updates;row["optimization_scope"]="fixed Gaussian count, native 0.8 L1 + 0.2 DSSIM + Adam; no densification; original checkpoint untouched";
    auto train_stats=c10::cuda::CUDACachingAllocator::getDeviceStats(0);row["training_peak_allocated_bytes"]=Json::Int64(train_stats.allocated_bytes[agg].peak);
   }
  }
  meta["frames"].append(row);std::cout<<"EXPORTED "<<name<<std::endl;
 }
 if(diag){auto selected_ids=ids.cpu().contiguous();std::ofstream f(out/"diagnostics/ply_row_ids.i64",std::ios::binary);f.write((char*)selected_ids.data_ptr<int64_t>(),ids.numel()*8);writeTensor(out/"diagnostics/xyz.f32",model->xyz_.index_select(0,ids));}
 meta["diagnostic_columns"]="axis_x,axis_y,axis_z,radius,confidence,has_prior,raw_xx,raw_xy,raw_yy,filtered_xx,filtered_xy,filtered_yy,t_x,t_y,n_x,n_y,width,rho,delta,amplitude,effective_q,conic_xx,conic_xy,conic_yy";
 meta["diagnostic_visibility_note"]="Geometric projections are not guaranteed visible contributors. Diagnostics also include unnormalized radial-depth, squared-depth and alpha moments for approximate surface-consistency filtering.";
 TORCH_CHECK(meta["frames"].size()>0,"No frames selected");meta["status"]="complete";std::ofstream f(out/"render_manifest.json");f<<meta;
}
