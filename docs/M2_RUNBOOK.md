# 工业语义 M2 独立运行

在服务器工程根目录执行，使用现有 `.venv`、本地 CLIP ViT-B/16 和 `code/bin/spcgs_native`。`configs/m2_industrial_multiscale.json` 中 GPU、缓存根目录、已完成 M1 的路径可按部署调整。改变训练预算或目标使用新的 run ID。

## A：场景理解与类别

```bash
.venv/bin/python tools/prepare_industrial_review.py
# 原生图像查看必须由人/Codex实际完成；脚本不执行识别。
.venv/bin/python tools/finalize_industrial_review.py
```

`finalize` 仅整理 `docs/industrial_scene_observations/authored*.json` 中已经完成的观察和 `zoom_review.json` 的复核结论，校验 56 张图片名称及已查看 RGB 的哈希。它不能替代新图片的实际视觉理解。最终逐图 JSON 和图像材料位于 `runs/industrial_semantics_20261023/00_scene_review/`；最终词表为 `configs/industrial_classes.json`。

## B：提取、比较、核查

```bash
# 代表图：入口立柱、泵管区、近景管线仪表、柜式设备区
.venv/bin/python tools/m2_semantics.py legacy --indices 1 23 47 52
.venv/bin/python tools/m2_semantics.py features --indices 1 23 47 52
.venv/bin/python tools/compare_m2_teachers.py
.venv/bin/python tools/inspect_m2_scales.py \
  runs/industrial_semantics_20261023/01_clip/features_pilot/manifest.json \
  --output runs/industrial_semantics_20261023/01_clip/scales_1024

# 全部 56 张真实训练图，及 M1 已注册的 4 张生成图
.venv/bin/python tools/m2_semantics.py features
.venv/bin/python tools/m2_semantics.py features --generated

# 数值及实际 C++ S 数据流检查
.venv/bin/python -m unittest tests.test_clip_dense
.venv/bin/python tools/check_m2_cache.py \
  runs/industrial_semantics_20261023/01_clip/features_all/manifest.json \
  --output runs/industrial_semantics_20261023/01_clip/full_cache_validation.json
.venv/bin/python tests/check_m2_s_contract.py \
  --cache runs/industrial_semantics_20261023/01_clip/features_pilot/manifest.json \
  --output runs/industrial_semantics_20261023/02_gaussians/S_contract_new
```

可用 `--output` 设置独立清单目录；相同输入会复用 `01_clip/cache/{F,T,S}` 中按内容签名的缓存。换词表需设置新 `class_config` 和输出目录；视觉 F 独立复用。缓存目录不放入 Git，服务器保存完整数组、覆盖次数、映射网格、prompt 向量和响应图。运行状态由各清单目录的 `outputs.json` 记录，不能用存在目录代替完成状态。

诊断配置可独立修改 `clip_face_size`（默认 1024）和 `clip_fov`（默认 100°），并用 `--config path/to/diagnostic.json --output path/to/new_manifest` 运行。二者均影响 F 的缓存签名。已保存的 2048 像素面、150° 视野及基础模板文本诊断均未替换正式配置；这些诊断不需要重新执行 M1。

教师对比中，上右图使用**旧 12 类**；其单独图例在最下方。其余语义图使用新 20 类。旧 F+新 T 的面板用于分离词表与视觉特征变化；raw cosine 与空间 min-max 的面板用于显示 relevancy 构造影响。

## C：短训及完整 SGI

```bash
bash tools/build.sh
.venv/bin/python tools/train_m2_semantics.py \
  --cache runs/industrial_semantics_20261023/01_clip/features_pilot/manifest.json \
  --run runs/industrial_semantics_20261023/02_gaussians/short_s0_new \
  --kind s0 --iterations 400 --rgb-width 1024 --no-test

# 检查短训 validation/ 的教师/学生图及 metrics.json 后执行完整 SGI。
.venv/bin/python tools/train_m2_semantics.py \
  --cache runs/industrial_semantics_20261023/01_clip/features_all/manifest.json \
  --generated-cache runs/industrial_semantics_20261023/01_clip/features_generated/manifest.json \
  --run runs/industrial_semantics_20261023/02_gaussians/sgi_full_new \
  --kind sgi
```

短训从原 OmniGS 的 647,929 个高斯初始化，冻结增密操作；RGB/语义仍按原接口联合优化。完整 SGI 从已完成 M1 的 4,267 个点初始化，沿用原增密、裁剪、OGR 和生成图语义监督，默认 10,000 步、RGB 2048×1024、语义 256×128。不重新生成 Wan360 视频、不重新执行 M1、不运行后续 SPC。`--resume path/to/checkpoint_N/model.pt` 仅用于相同配置中断续跑。

训练 run 保存引用和配置、初始/最终高斯及映射头、检查点、逐步损失、原生程序日志和 RGB/语义渲染。`validation/` 保存 4 个代表视角的初始/最终教师学生并排图、F 余弦、S/学生概率、KL、有效贡献比例及损失曲线。完整 SGI 同时按既有接口渲染 16 个测试视角用于 RGB 评估；测试图没有参与词表设计和 CLIP 监督生成。

KL/标签一致率仅衡量学生拟合教师，不能当成真实语义准确率；S 做 temperature=1 softmax 后接近均匀，argmax 对很小的分数变化也敏感。二维教师存在的设备混淆必须在结果报告中保留。

已有旧版 SGI 模型的四视角重渲染保存在 `02_gaussians/archived_sgi_1000_reference/`。完整新训练结束后，可生成带独立类别图例的历史对照图：

```bash
.venv/bin/python tools/compare_m2_history.py \
  runs/industrial_semantics_20261023/02_gaussians/archived_sgi_1000_reference \
  runs/industrial_semantics_20261023/02_gaussians/sgi_full_new/02_sgi_gaussians \
  runs/industrial_semantics_20261023/02_gaussians/history_comparison_new
```

旧模型训练为 1,000 步、512×256、12 类；新模型为 10,000 步、2048×1024、20 类，初始化点云和部分训练设置也不同。它们虽用相同相机与显示分辨率重渲染，仍不构成控制变量实验。脚本保留两个训练配置，不计算不可比较的类别一致率。

文本查询继续使用原入口，例如：

```bash
.venv/bin/python -m spcgs.query --gpu 1 --split train \
  --model runs/industrial_semantics_20261023/02_gaussians/short_s0_new/03_semantic/s0/final/model.pt \
  --view 2023-05-11_10.18.46 --classes 'pipe,structural column,equipment cabinet,floor' \
  --output runs/industrial_semantics_20261023/query_new
```

严格断点恢复回归可在独立测试目录重跑：

```bash
SPCGS_TEST_GPU=1 SPCGS_TEST_OUTPUT=tests/regression_m2_new \
  .venv/bin/python tests/check_regression.py
```

本次该检查的原版 RGB 对照通过，但语义参数恢复等价断言未通过；详情见 [结果报告](M2_RESULTS.md)。不要将其作为已通过的检查。已保存的首次失败产物位于 `runs/industrial_semantics_20261023/02_gaussians/regression_first/`，使用新的输出目录可保留既有测试文件。完整 SGI 采用连续训练，没有依赖该恢复检查。
