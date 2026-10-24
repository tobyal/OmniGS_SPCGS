# 27 视角 SGI → SPC 第二阶段

本入口复用 `sparse27_official_sgi_20261024/opacity_lr_fixed`，不执行 M1/M2，不生成 Wan360 视频，不构造新的 CLIP 教师。正式输出为 `runs/industrial_spcgs/spc27_hieral_20261024`。

## 输入与阶段初始化

来源 SGI 为 10,000 步、287,787 个高斯，完整归档 SHA256：`edc330a064a41443efe6e11cd6bbb7f711182c8414c3e96678728897decfa362`。七类、27 张实际训练 ID、F/S/T/W 和标签版本快照全部复用来源 run。

`stage_reset` 保留位置、零阶 SH 颜色、32 维语义及 decoder、bias、classifier、cbias、projection、text。尺度根据 SGI 位置的近邻距离重新计算；旋转设为单位四元数，不透明度设为 0.1，高阶 SH 和增密统计清零。高斯与映射优化器重新创建，不载入 SGI 的 Adam 状态。C0/C1/C2 各自直接读取同一 SGI，不能串联。

`tools/check_stage2_initialization.py` 读取实际 LibTorch 归档，逐张量核对继承、重置和各组初始参数一致性。当前独立初始化、C0、C2 的继承误差均为零；C1 启动后再次纳入同一检查。

## SAM2 与 ISP

使用原始 **SAM2 Hiera-L**，不是 SAM2.1 tiny。权重从 [Meta 官方下载脚本所列地址](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh) 获取：

- 权重：`weights/sam2_hiera_large.pt`，SHA256 `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`。
- 配置：`vendor/sam2/sam2/configs/sam2/sam2_hiera_l.yaml`。
- 官方代码提交：`2b90b9f5ceec907a1c18123530e92e794ad901a4`。
- 实际加载参数 224,430,130 个，全部冻结。

每次 ISP 取当前真实训练图像及当前模型在线渲染的两个伪视角。伪位置沿当前相机至最近两个真实相机之间的线段采样，插值比例 0.1–0.3，加上近邻距离 0.02 倍的高斯噪声；ERP 旋转保留当前相机方向。相机选择只使用本轮 27 张。

真实图像上均匀随机采点；三帧一起水平循环平移，将提示置于图像中间。SAM2 传播后将掩码逆平移，再用最近邻缩放至 256×128 语义网格。AMG 也映射回原 ERP 坐标。保留原来的非空、面积和比例检查，并记录失败原因；这些检查不是对应准确率。

训练进程复用一个常驻 SAM2 模型，视频预测与自动区域提议共享冻结骨干。原生训练结束后自动释放该进程。逐事件保存真实/伪 RGB、相机、提示、掩码、最大贡献索引、贡献权重和有效性信息。独立 `spcgs.isp` 必须提供含 `train_names` 的配置，不再默认为 56 张。

## 损失与实现范围

对照 [SPC-GS 第 3.3 节与补充材料](https://arxiv.org/html/2503.12535v1#S3.SS3)，保留现有 `inter()`、`contrastive()`、`kl3d()`：

- Inter 使用当前训练视角自渲染语义的区域均值和主导类别作为停止梯度的提示；官方标签仅进入真实视角基础监督。
- Intra 使用 SAM2 自动区域统一自渲染类别，再构造同类正样本、异类负样本，温度 0.2。
- 三维 SPC 对掩码执行 5×5 腐蚀，水平边界循环、垂直边界无效。使用 CUDA 渲染时实际 `alpha×T` 最大的高斯，源区域概率均值约束伪视角高斯。对 32 维特征作 softmax 得到合法 KL 分布，保留现有按像素平均的实现。
- `local3d` 仍是单独的近邻语义正则，不能视作三维 SPC。L_GS 为零，伪视角不提供 RGB 真值。

本轮沿用工程的计算预算：每 100 步执行一次 ISP（每次两个伪视角），局部正则采样 128 个高斯；Intra 改为从随机 128×128 ERP 区域内采样最多 512 个像素。它保留论文损失目的，但不是每步全 128×128 像素的对比计算，也不是论文的 800 点局部采样。三组使用相同基础配置，SPC 权重在 4,000 步内线性增至 1。单独的采样 RNG 避免对比采样额外消耗基础训练随机流。

逐 ISP 事件记录三项损失各自进入高斯语义参数的梯度范数。诊断反向传播产生的屏幕梯度会清除，避免污染随后真实联合损失的增密统计。原生检查覆盖 KL 数值梯度、停止梯度方向、ERP 腐蚀、最大贡献高斯非最近点、语义增密与归档复载。

## 执行

在服务器工程根目录执行：

```bash
.venv/bin/python tools/run_spc27.py --stage initialize --gpu 0
.venv/bin/python tools/check_stage2_initialization.py runs/industrial_spcgs/spc27_hieral_20261024
.venv/bin/python tools/run_spc27.py --stage isp --view-index 14 --seed 2 --gpu 0
.venv/bin/python tools/run_spc27.py --stage short --gpu 2
.venv/bin/python tools/run_spc27.py --stage c0 --gpu 2
.venv/bin/python tools/run_spc27.py --stage c1 --gpu 2
.venv/bin/python tools/run_spc27.py --stage c2 --gpu 0
```

200 步短训已完成，10 次 ISP 中 13/20 对通过基本检查。Inter、Intra、三维 SPC 的最大语义梯度范数分别为 1.0393、0.09387、0.01683；完整短训模型已重新加载并渲染。初始重置会使 RGB 暂时退化，不能把初始伪视角的质量当作正式训练完成后的表现。第一批预检查也保留了细小提示传播失败及边界碎裂的案例。

## 比较与输出

```bash
.venv/bin/python tools/run_spc27.py --stage render --model c0 --gpu 2
.venv/bin/python tools/run_spc27.py --stage render --model c1 --gpu 2
.venv/bin/python tools/run_spc27.py --stage render --model c2 --gpu 0
.venv/bin/python tools/prepare_spc27_evaluation.py runs/industrial_spcgs/spc27_hieral_20261024
.venv/bin/python -m spcgs.region_evaluation runs/industrial_spcgs/spc27_hieral_20261024 --events-manifest runs/industrial_spcgs/spc27_hieral_20261024/06_comparison/evaluation_correspondences/manifest.json
.venv/bin/python -m spcgs.spc27_evaluate runs/industrial_spcgs/spc27_hieral_20261024 --action lpips --model c2 --gpu 0
.venv/bin/python -m spcgs.spc27_evaluate runs/industrial_spcgs/spc27_hieral_20261024 --action query --model c2 --gpu 0
.venv/bin/python -m spcgs.spc27_evaluate runs/industrial_spcgs/spc27_hieral_20261024 --action report
```

`lpips` 与 `query` 对每组分别执行；报告会要求三组均已完成、重新渲染并通过查询检查，不会把部分结果写成正式完成。LPIPS 复用已安装的 v0.1 AlexNet 与缓存权重，在完整 2048×1024 ERP 上计算。查询复用原七类冻结文本向量，绕过已学习的类别校准头，不重新运行 CLIP。`tools/finish_spc27_evaluation.py` 的两个队列分别负责 C0 完成后在 GPU 2 启动 C1，以及 C2 完成后在 GPU 0 渲染 C2/C0；仅适用于本次已启动的 C0/C2。

固定评估对应使用 1、14、27、48、56 号空间锚点、各自最近的两个未见真实相机，以及预定均匀随机种子 2、16、42。三帧都是实际拍摄 RGB；掩码只用于评估，不参与任何优化。所有模型用同样的区域评价，汇总三者共同可评估的区域，同时报告覆盖率、类别分布与熵，避免以共同错误类别造成的一致性冒充正确率。没有独立语义真值时，不报告未见视角 mIoU。

| 目录 | 内容 |
|---|---|
| `03_stage_initialization` | 重置后的初始归档、继承记录、初始 RGB/语义 |
| `04_isp/preflight` | 独立 ISP 实际样例与失败原因 |
| `short/05_spc/c2` | 200 步短训及非零梯度检查 |
| `05_spc/c0,c1,c2` | 各组完整初始/最终模型、检查点和日志 |
| `05_spc/c1/isp,c2/isp` | 在线伪视角、SAM2、贡献高斯和逐事件诊断 |
| `06_comparison/evaluation_correspondences` | 独立固定评估对应、原始 RGB 与掩码 |
| `06_comparison/fixed_region_consistency` | 同区域二维/三维一致性与覆盖统计 |
| `05_spc/*/renders_all` | 全部 78 个真实视角的 RGB、语义、贡献索引及指标 |
| `05_spc/*/text_query` | 独立加载后的七类文本查询与验证 |
| `07_report` | HTML 报告、全部视角浏览、固定对比图、损失曲线和定性检查 |

正式 C0/C2 已启动；最终步数、有效 SPC 次数及比较结果以各组 `outputs.json`、`losses.jsonl` 和最终报告为准。此处不把运行中的任务写成已完成。
