# M1 增强点云：同配置 OmniGS RGB 训练与训练集评估

完成时间（北京时间）：2026-10-24T02:44:22.791442+08:00。Run：`m1_rgb_baseline_11901_20261024`。

仅替换稀疏初始化点云：3,979 → 11,901。使用原基线训练可执行文件、完全相同的 YAML 配置、56 张真实训练图、2048×1024、32,010 步、seed=0、PF_MODE=0；原有学习率、增密、裁剪、透明度重置、SH 和损失保持一致。
初始化沿用原生 `createFromPcd()`：颜色转 SH，最近邻距离决定球形尺度，单位四元数旋转，初始透明度 0.1。原始 3,979 点的位置和颜色完全保留。训练监督仅使用同一组真实 RGB，相机与训练集划分不变。

## 全部 56 张训练图的最终指标

| 指标 | 原始点初始化基线 | M1 增强点初始化 | 新 − 旧 |
|---|---:|---:|---:|
| PSNR ↑ | 28.729582 dB | 28.922526 dB | +0.192944 dB |
| SSIM ↑ | 0.880480 | 0.883963 | +0.003483 |
| LPIPS ↓ | 0.254133 | 0.248132 | -0.006002 |
| 最终高斯数 | 647,929 | 660,735 | +12,806 |

逐视角改善数量：PSNR 38/56、SSIM 48/56、LPIPS 45/56。完整逐视角数值保存在 run 下的 `per_view_comparison.csv`。

## 评估口径

两模型使用同一原生渲染器重新渲染全部 56 张真实训练图。指标从 2048×1024 浮点 RGB 计算，渲染值裁剪到 [0,1]，不使用有损 JPEG，不缩小图像，不挑选训练子集。各视角指标取等权算术平均。
PSNR 使用全 RGB 像素的均方误差，与原生 OmniGS `psnr` 定义一致。SSIM 使用原有 ODGS 的 11 像素 Gaussian 窗、sigma=1.5。LPIPS 使用 AlexNet v0.1，并将 [0,1] RGB 映射至 [-1,1]。
历史辅助评估脚本另有逐通道平均 PSNR 与直接输入 [0,1] 的 LPIPS 口径。为防止混用，JSON 同时保存 `legacy_channel_mean_PSNR` 和 `legacy_input01_LPIPS`，主表采用上述统一定义。
这些是训练集重建指标；本次没有计算验证集或测试集指标，也没有进行语义训练。

## 验证与产物

- 原训练二进制 SHA256：`602e0ddd6b57948bd1cb439d4e2b507d7552bbcd48ac10aff56791f38c823244`。
- 配置 SHA256：`1d799fe05ecfec289de8e797712837ad2da18dfa1aaa85241f668806257a346e`，与基线训练记录一致。
- 原生训练日志确认载入 11,901 个初始点；56 个相机均被使用，总更新次数 32,010。
- 原基线模型、M1 增强输入、所有训练 RGB 和原生库的哈希保持不变；两次评估使用同一相机、渲染器与指标实现。
- 新训练耗时：约 27.77 分钟；计时来源：Approximate wall time from training command file to latest final checkpoint/camera-usage file mtime. 旧基线记录来自并发运行，不能据此作训练速度比较。
- 原生 PLY 透明度字段保存 logit：本模型有 22,426 个 +∞，经原生 sigmoid 激活为有限透明度 1；其余字段有限，透明度无 NaN。
- 本次自动收尾最初误将无限透明度 logit 判为异常；修正校验后直接评估已完成的模型，没有修改模型或重新训练。原始检查日志与恢复脚本保留在 run 目录。

完整服务器目录：

```text
/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023/runs/industrial_spcgs/m1_rgb_baseline_11901_20261024
```

- `model/32010_shutdown/ply/point_cloud/iteration_32010/point_cloud.ply`：最终模型。
- `evaluation/baseline_train/`、`evaluation/enhanced_train/`：全量训练视角 float RGB、PNG、逐视角 CSV 与指标 JSON。
- `experiment.json`、`training_result.json`：配置、数据、二进制、模型校验与相机使用次数。
- `preview/train_rgb_comparison.png`：固定均匀抽取的四个训练视角，真实图 / 基线 / 增强初始化模型。
- `preview/training_curves.png`：新训练的损失与高斯数量曲线。

机器可读结果：[results/m1_rgb_baseline_20261024.json](results/m1_rgb_baseline_20261024.json)。
