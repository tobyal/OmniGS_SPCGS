# 27 视角官方语义 SGI 复现说明

本流程使用 IRIS 的 27 张官方标注训练全景，重新三角化基础点云，复用 Wan360 缓存增强几何，再以七类 CLIP 文本原型训练 SGI。输出用于后续 SPC 初始化。本入口不会生成视频，也不会启动 M4 或 M5。

## 数据与训练协议

服务器工程位于 `/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023`。本轮独立结果目录为 `runs/industrial_spcgs/sparse27_official_sgi_20261024`。

`00_protocol/manifest.json` 根据语义审核数据中的六个官方面记录选出实际图像 ID，再核对原训练序号：

`1, 2, 10, 12, 13, 14, 15, 18, 19, 20, 21, 25, 26, 27, 30, 34, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 56`

其余 29 张训练图像仅用于未见视角检查；原有 6 张验证和 16 张测试图像保持独立。相机按原始三维位置分组，保留高度差异。沿用提供的 RMX 位姿；历史轨迹仅作为关联提案，P0 坐标全部由所选真实观测重新求解。

标签快照保存当前 revision、来源图、官方映射哈希和实际监督标签。source 1 必须与官方映射一致；source 3 仅在对应 revision 获得人工批准时使用。未批准的 SAM 补全不进入监督；已检查并删除的三脚架等区域保持 ignore。既有审核数据不会被改写。

## 执行步骤

在工程根目录使用 `.venv/bin/python`。改变实验配置时使用新的 `--run` 名称。

```bash
.venv/bin/python tools/run_sparse27.py --stage protocol
.venv/bin/python tools/run_sparse27.py --stage p0
.venv/bin/python tools/run_sparse27.py --stage m1
.venv/bin/python tools/run_sparse27.py --stage teacher --gpu 0
.venv/bin/python tools/check_official_teacher.py
.venv/bin/python tools/run_sparse27.py --stage short --gpu 0
.venv/bin/python tools/run_sparse27.py --stage render --model short --representatives --gpu 0
.venv/bin/python tools/run_sparse27.py --stage sgi --gpu 0
```

相同 27 张 RGB、10,000 步的 P0 普通初始化参考可在空闲 GPU 独立运行：

```bash
.venv/bin/python tools/run_sparse27.py --stage rgb --gpu 2
```

`check_official_teacher.py` 默认检查上述本轮目录。检查输出位于 `03_semantic/mask_validation`，覆盖无效像素不影响损失或梯度、全零 W、无 W 的旧接口兼容，以及原生实现与独立 PyTorch 公式的一致性。

训练后保存初始及最终渲染，再生成报告：

```bash
.venv/bin/python tools/run_sparse27.py --stage render --model rgb --initial --gpu 2
.venv/bin/python tools/run_sparse27.py --stage render --model rgb --gpu 2
.venv/bin/python tools/run_sparse27.py --stage render --model sgi --initial --gpu 0
.venv/bin/python tools/run_sparse27.py --stage render --model sgi --gpu 0
.venv/bin/python tools/run_sparse27.py --stage report
```

## M1 实现与已验证结果

P0 从已有特征轨迹删除非 27 张视角的观测，保留至少两个有效真实观测，重新三角化，再执行现有重投影、视差、去重和密度检查。P0 为 **4,140 点**，最大观测重投影误差的点间中位数为 **0.606 像素**；27 张相机的共享轨迹图连通。

M1 复用 27 个源的 forward、backward、left、right 共 **108 个视频**，检查早期帧 1–4，共 432 个候选。全部位姿注册使用新的 27 视角支持集和 P0 锚点。每个源、每个方向最多保留一个通过几何验证的帧，优先选择能够提供更多可靠新轨迹的候选；空间区域及高度层轮换处理。

204 个候选通过注册，最终选择 94 帧，贡献 **3,281 个新点**，P1 合计 **7,421 点**。新增点的最大重投影误差中位数为 **1.974 像素**，P90 为 **3.631 像素**。这些是内部几何一致性指标。

| 方向 | 新增点 |
|---|---:|
| forward | 1468 |
| backward | 717 |
| left | 554 |
| right | 542 |

`motion_direction_cosine` 只保留作分析：94 个选中帧中有 14 个为负值，不参与任何拒绝逻辑。`spcgs/m1_geometry.py` 的匹配、注册和几何筛选条件没有改变。

按生成源归属，上部、中部和下部区域分别新增 835、1063、1383 点。56 号末端源没有通过注册的生成帧；由其他源贡献的点中有 124 点在三维位置上最接近末端区域。源归属与点的实际空间归属分别记录，不能混用。

历史 56 视角 P2 为 11,901 点，仅作参考。历史基础 3,979 点的轨迹保留条件与本轮 P0 不同，因此不能仅由基础点数比较稀疏重建优劣。

## 七类监督与训练

类别顺序固定为 piping、equipment、building、structural_frame、access_structure、cable_tray、hvac。每类三个英文提示经冻结 CLIP ViT-B/16 编码、逐提示归一化并取归一化均值，形成 `T: 7×512`。

在 256×128 语义网格上，`F = T[label]`；`S = log(0.95 × onehot + 0.05/7)`，因为原生损失会对 S 做类别 softmax；`W` 表示有效监督像素。缓存的有效像素平均占 93.81%，教师最高类概率约 0.95713。W 同时作用于特征余弦损失和交叉熵，RGB 损失不使用 W。

这是**类别原型监督**。保留 32 维可学习高斯语义、32→512 映射、类别分类头、局部三维正则、原有增密和 OGR。没有加载旧 20 类教师或分类头。原型张成空间包含于初始 32 维正交基，初始语义来自最近的有效标签投影；这只是属性初始化，不代表已解决遮挡。

400 步短训验证梯度、类别和渲染后，从 P1 重新开始 10,000 步正式 SGI。RGB 分辨率 2048×1024，语义分辨率 256×128，seed 42；增密从 500 步到 9000 步、间隔 100 步、阈值 0.0002，OGR 间隔 3000 步、半径 1、最少邻居 5，局部三维项权重 1。

第一轮关闭生成视角语义项 L_GS。生成图像不提供 RGB 监督。P0 RGB 参考使用相同训练 ID、种子、分辨率、步数及增密预算，关闭语义损失和 OGR。它比较的是完整 SGI 流程与普通初始化，不能将差异单独归因于 Wan360。

## 产物与后续加载

| 目录 | 内容 |
|---|---|
| `00_protocol` | 训练 ID、相机、分组、标签和来源快照 |
| `01_p0` | 重新三角化点云、观测、连接图和质量统计 |
| `01_sgi_sfm` | 缓存来源、注册、匹配、选帧、逐点轨迹和 P1 |
| `03_semantic/cache` | 七类 F/S/T/W、提示、映射基和教师预览 |
| `02_sgi_short` | 短训、预览和独立续训检查 |
| `02_sgi_gaussians` | 正式 SGI、各项损失、检查点、OGR 和渲染 |
| `02_rgb_p0_reference` | 同预算 RGB 参考 |
| `07_report` | HTML 报告、完整 JSON 指标和对比图 |

`final/model.pt` 保存高斯、32 维语义、映射与分类头、Adam 和随机状态；PLY 及语义 sidecar 同时保存。短训完整模型已通过独立目录的两步续训检查。

后续 SPC 需要沿用本 run 的七类缓存和 `sgi_config.json` 中的显式 `train_names`。标准 `pipeline.prepare` 支持这些实际 ID，`train_stage` 会核对教师 ID 集合，场景尺度由所选相机计算。不要改回默认 56 视角或旧类别配置。本任务不执行后续 SPC。

RGB 报告分别汇总训练、未见 29 张、验证和测试视角。语义 mIoU 只衡量 27 张训练图上的官方标签拟合；未见视角语义图用于检查，不当作有真值的泛化指标。
