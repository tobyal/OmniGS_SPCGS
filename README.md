# OmniGS-SPCGS 工业全景研究工程

独立服务器工作目录：`/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023/`。
上游为 `../OmniGS_Structure/`；原版 `baseline_s0` 只读复制到 `baseline/checkpoint.ply`，不修改上游任何训练结果。数据引用固定 56/6/16 划分及 3,979 个训练集 SfM 点。

## Git 仓库与数据接入

源码仓库：[tobyal/OmniGS_SPCGS](https://github.com/tobyal/OmniGS_SPCGS)。Git 仅保存代码、配置、测试程序和说明；数据集、基线模型、CLIP/SAM2 权重、构建产物和实验输出保存在服务器工作目录中，由 `.gitignore` 排除。

在同一服务器建立新的源码副本，可接入已有的只读输入：

```bash
cd /mnt/sda/ubuntu/lkx/od
git clone https://github.com/tobyal/OmniGS_SPCGS.git
cd OmniGS_SPCGS
mkdir -p data baseline
ln -s /mnt/sda/ubuntu/lkx/od/OmniGS_PipeFilter/data/dataset data/dataset
cp ../OmniGS_SPCGS_20261023/data/protocol.json data/protocol.json
cp ../OmniGS_SPCGS_20261023/baseline/provenance.json baseline/provenance.json
ln -s /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023/baseline/checkpoint.ply baseline/checkpoint.ply
bash tools/build.sh
bash tools/setup_models.sh
```

其他机器需自行准备相同数据协议和基线，并按环境调整 `tools/build.sh`、`tools/setup_models.sh` 中的 CUDA、LibTorch、OpenCV 和基础 Python 路径；Wan360 模式的 OpenMVG 路径通过 `m1_geometry.openmvg_bin` 配置，旧 M1 模式的路径由 `spcgs/sfm.py` 定义。下面的运行命令以本次服务器工作目录为例，在新副本中执行时使用新副本路径。上游许可证保留在 `code/LICENSE`、`code/LICENSE.md` 及各第三方目录中。

## 运行

```bash
cd /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023
bash tools/build.sh
# 首次安装模型依赖；写入本工程 .venv、vendor、weights，不修改其他 conda 环境
bash tools/setup_models.sh

# 快速完整流程：8 训练视角，512×256 RGB，128×64 语义，每阶段 100 次
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m spcgs.pipeline --run quick_new --stage all

# 较长试验：56 训练 / 16 测试；配置中的 SGI 与正式阶段各 1000 次
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m spcgs.pipeline --run pilot_new --config configs/pilot.json --stage all

# 论文预算参考：56/6/16，全景 RGB 2048×1024，SGI/正式阶段各 10k
# 预计明显比原 RGB 训练慢：语义采用 11 次原始三通道光栅化，SAM2 周期在线运行。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m spcgs.pipeline --run full_new --mode full --config configs/full.json --stage all

# 单模块：baseline, m1, features, s0, sgi, c0, c1, c2, compare
.venv/bin/python -m spcgs.pipeline --run pilot_new --config configs/pilot.json --stage c2

# 同一配置内从检查点恢复；保留 Adam、随机状态和高斯增密统计
.venv/bin/python -m spcgs.pipeline --run pilot_new --config configs/pilot.json --stage c2 \
  --resume runs/industrial_spcgs/pilot_new/05_spc/c2/checkpoint_250/model.pt

# 读取已有模型，独立执行 SAM2 随机提示与两邻近视角传播
.venv/bin/python -m spcgs.isp --model runs/industrial_spcgs/pilot_new/05_spc/c2/final/model.pt \
  --output runs/isp_example --seed 123

# 任意新文本查询；未见过的词表直接使用 CLIP 相似度，不套旧类别分类头
.venv/bin/python -m spcgs.query --model runs/industrial_spcgs/pilot_new/05_spc/c2/final/model.pt \
  --output runs/query_example --classes 'pipe,metal cabinet,concrete floor,ceiling,pump' \
  --view 2023-05-11_10.20.43 --width 1024

# 新相机用 --cameras cameras.json，内容为 [{name,center,rotation}]，R 为 world-to-camera
# 对比任意两组同分辨率、同视角渲染
.venv/bin/python -m spcgs.comparison RUN_A/renders RUN_B/renders runs/comparison_example

# 训练后补齐感知指标、初始化并排图，以及固定 SAM2 区域上的一致性比较
.venv/bin/python -m spcgs.extra_evaluation runs/industrial_spcgs/pilot_new
.venv/bin/python -m spcgs.extra_evaluation runs/industrial_spcgs/pilot_new --initialization
.venv/bin/python -m spcgs.region_evaluation runs/industrial_spcgs/pilot_new
.venv/bin/python -m spcgs.pipeline --run pilot_new --config configs/pilot.json --stage compare
```

同一 run ID 的配置必须完全相同。更改分辨率、词表、迭代或模块参数时使用新 run ID；这样保留失败和负面结果。`--resume` 用于同一配置中断后的继续执行，不用于改变训练预算。原生请求也可直接用 `code/bin/spcgs_native request.json` 运行。

## 代码与算法对应

| 模块 | 实现 | 原有核心复用 |
|---|---|---|
| 基线 | `spcgs/pipeline.py`、原生 render action | `GaussianModel::loadPly`、`GaussianRenderer::renderLonlat` |
| M1 | `spcgs/wan360.py`、`m1_wan360.py`、`m1_geometry.py`；旧模式 `sfm.py` | 官方 Wan360、原始 OpenMVG SIFT/RootSIFT 观测、球面注册与多视图三角化 |
| M2 | `spcgs_native.cpp` SGI stage、`geometry_worker.py` | `createFromPcd`、最近 3 邻点球形尺度、原生增密裁剪；增加 OGR |
| M3 | `spcgs/features.py`、`spcgs_native.cpp` | 原有透明度累积和完整几何反传，32 维语义分 11 批渲染 |
| M4 | `sam2_worker.py`、`spcgs/isp.py` | 当前模型渲染邻近相机；真实 SAM2 传播与自动掩码 |
| M5 | `inter` / `contrastive` / `kl3d` / `local3d` | 训练视角自渲染语义提示、区域多数类别、正负对比、腐蚀及最大权重高斯 KL |
| 评估 | `spcgs/comparison.py` | 相同相机、词表、分辨率的渲染、差异图和指标表 |

实际修改的原有文件：

- `code/include/gaussian_model.h`、`code/src/gaussian_model.cpp`：可选语义参数，第七个 Adam 参数组；clone/split 继承父点语义，prune 同步裁剪语义和 Adam 状态。修复复制代码中 opacity reset 的 `ones_like(opacity*0.01)` 实际返回 1 的问题。
- `code/cuda_rasterizer/forward.h`、`forward.cu`：在实际前向混合循环中记录最大 `alpha*T` 及对应高斯行号；无贡献为 `-1`。不使用最近空间点代替。
- `code/CMakeLists.txt`：加入独立训练/渲染程序及数值测试目标。

其他改动通过新增模块接入。RGB 渲染保持 `PF_MODE=0`、`SGS_MODE=0`，不使用旧工业结构先验、CAD 标签或历史语义场。

## 文件协议

每阶段保存 `inputs.json`、`outputs.json`、`config.json`、`metrics.json`、`preview/`、日志与失败原因。目录按 `00_baseline` 至 `07_reports` 编排。

- `model.pt`：原生 LibTorch archive，包含高斯位置、SH、log-scale、rotation、opacity logits、32 维语义、512 维映射头、分类/对比头、Adam、随机状态和迭代；同一训练可恢复。
- `point_cloud.ply`：OmniGS 原生全部高斯字段；单独读取它只得到 RGB 模型。
- `semantic.f32`：按 PLY 行号对齐的 N×32 float32；`decoder.f32` 为 512×32，`decoder_bias.f32` 为 512；同时保存文本向量。
- RGB `.f32`：CHW float32；`.png` 为预览。语义 `.f16`：32×H×W；logits 为 K×H×W；ID `.i32`：H×W；weight `.f32`：H×W。
- CLIP 缓存 `.f16`：512×cache_H×cache_W，与真实全景像素中心对应；相关性图压缩 NPZ 保存。
- SAM2 `masks.u8`：3×H×W，顺序为真实、伪 1、伪 2；`regions.i32` 为真实视角自动区域编号，背景 -1。原始/滚动相机、随机提示、接缝平移和有效性均保存。
- OGR 半径单位与固定 RMX/OpenMVG 点云一致；计邻居排除自身，默认半径 1、邻居数 5。

SGI → 正式阶段只传递位置、DC 颜色和语义及其映射头。尺度按原来的最近邻公式重新计算，旋转为单位四元数、透明度为 0.1、高阶 SH 清零，优化器重新创建。不要把这个阶段初始化与同阶段 `--resume` 混用。

## M1 接入协议与限制

新增 **`m1_mode=wan360`**：直接执行官方 Wan360 推理、早期选帧、基于原始训练集 2D–3D 观测的球面位姿注册、真实—生成多视图三角化及点云增强。无需预先提供 `registered=true`。完整环境、坐标约定、配置和输出说明见 [M1_WAN360.md](docs/M1_WAN360.md)。

2026-10-23 实测：一张真实训练全景生成四方向视频，早期帧配置选中三个注册视角，新增 **233 点（3979 → 4212）**；100 步 SGI 的初始化数量与位置逐点核对通过。推荐 `configs/m1_wan360_early.json`；晚帧形变、left 拒绝原因、完整计数和可视化路径见 [M1 实测报告](docs/M1_RESULTS_20261023.md)。8/56 张批量实验尚未执行。

```bash
bash tools/setup_wan360.sh
.venv/bin/python -m spcgs.pipeline --run m1_wan360_test \
  --config configs/m1_wan360_test.json --stage m1
```

配置分别控制生成源视角数和真实几何支持集，初始只对 1 张训练图生成四方向视频。`configs/m1_wan360_8.json`、`configs/m1_wan360_56.json` 提供后续扩展设置，复用相同参数的已有视频和特征缓存。Wan360 使用独立 `.venv_wan360`；原 M2—M5 核心算法不变。以下为仍保留的旧 `skip/import/online` 模式。

默认 `m1_mode=skip`，明确输出 `status=skipped`、`enhanced_ply=null`，SGI 读取原始 SfM。未声称运行 MotionCtrl。

已有生成结果：配置 `m1_mode=import`、`generated_manifest=/path/manifest.json`。在线模型：设置 `m1_mode=online`、`generation_command` 为 argv 数组，由生成程序先产出同一 manifest。格式：

```json
{
  "coordinate_system": "rmx_openmvg_world",
  "views": [{
    "name": "generated_0001", "image": "/absolute/generated.png",
    "projection": "equirectangular", "registered": true,
    "registration_evidence": "/absolute/geometric_registration.json",
    "source_training_view": "2023-05-11_10.18.46",
    "center": [0, 0, 0], "rotation": [[1,0,0],[0,1,0],[0,0,1]]
  }]
}
```

其中示例位姿是格式说明，运行必须替换为实际注册位姿。会拒绝只有旋转没有新相机中心的视图、非训练图像来源，以及未注册输入。当前 OpenMVG 接口支持 ERP；透视生成图需要生成/注册提供者完成兼容转换，不能直接冒充全景。真实/生成匹配需通过重投影和视差检查，原始点云保留；生成视角只增加语义监督，不进入 RGB 损失。

## 验证与研究限制

```bash
/mnt/sda/ubuntu/anaconda3/envs/lkx_omnigs/bin/cmake --build code/build -j 2 --target spcgs_check
CUDA_VISIBLE_DEVICES=0 code/bin/spcgs_check tests/results
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/check_regression.py
```

数值测试检查语义梯度、几何反传、最大贡献点并非最近点、增密/裁剪语义继承、Adam 对齐、完整保存重载。实际基线测试与原工作目录二进制渲染同一视角；中断恢复测试使用真实点云、CLIP 和训练数据。

快速试验已发现并保留：第一版重载场景尺度类型错误导致第二阶段几何发散；修复后独立重载测试通过。100 步 SGI 的伪视角仍很模糊，SAM2 可能无法传播有效区域，相关失败不计为 SPC 成功。

CLIP 两种后端：`patch_tokens` 为快速原始 patch 投影；`local_crops` 使用重叠透视局部图的真实 CLIP image embeddings，定位粗但避免把未训练的 patch 投影当成可靠分类器。两者都不等于论文的所有预处理细节。SAM2 Hiera-tiny 替代论文的 Hiera-L；语义分辨率和在线执行频率可配置。全景接缝用环形平移处理，极区畸变仍需检查。

为了控制资源，intra 对比项在自动掩码内采样 256/512 个像素，沿用相同的监督对比损失，未使用原文的完整 128×128 patch。没有有效正负样本时该项跳过，不伪造非零损失。

语义伪标签不是独立真值，不报告正式 mIoU/mAcc。SAM2 有效率只是非空/面积比检查；人工可视化仍是必要依据。PSNR/SSIM 在固定测试集上算，不混入训练视角。较短试验不能与 32,010 步 RGB 基线直接比较并宣称改进。

参考：[SPC-GS](https://arxiv.org/html/2503.12535)、[SAM2 官方实现](https://github.com/facebookresearch/sam2)、[CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16)。
