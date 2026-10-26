# IPSGR：修复后 IRIS 20/6 七类实验

本轮已完成，结果见[最终中文报告](ipsgr_repaired20_6/10_report/report.md)及[六图对照与交互浏览说明](ipsgr_repaired20_6/README.md)。

当前运行目录：`runs/industrial_ipsgr_20train_6test_repaired7_20261025`。
权威协议是该目录下的 `00_dataset/dataset_manifest.json`，包含独立复制的 20 张训练、6 张测试 RGB/标签、相机与 SHA256。训练不读取测试图像或测试标签。标签是官方标注经过既定七类合并及修复得到的版本，并非未经修改的官方原始标签。

## 执行入口

使用仓库 `.venv/bin/python tools/run_ipsgr.py --run <绝对运行目录> --stage <阶段>`。

1. `audit`：检查并冻结当前修复数据、类别配置、相机和训练参数。
2. `p0`：重新提取训练图像特征、匹配全部 190 对、构建轨迹并三角化。
3. `wan --gpu 0`：重新生成全部 20×4 视频。独立新运行使用独立缓存路径；同一运行可按签名恢复。
4. `m1`：使用本轮视频和 P0 注册、选帧、三角化生成 P1。
5. `teacher --gpu 3`：仅从 20 张训练标签生成七类文本原型及 F/S/T/W。
6. `train --model B-RGB`、`train --model B-SEM` 使用 P0；`train --model G-AUG`、`train --model ISLB` 使用 P1。
7. ISLB 完成后先运行 `train --model smoke-IPSGR-Full`；通过后分别执行 `train --model L-BOOT`、`CV-2D`、`IPSGR-Full`。三个正式模型独立读取同一 ISLB 最终模型，不能互相续训。
8. 为六个正式模型分别执行 `evaluate --model <名称>`。评价前必须已通过 `spcgs.ipsgr_analysis.freeze_regions(run)` 固定本轮 ROI 和对象身份记录；该函数拒绝在结果出现后重新生成协议。
9. 原生 512×256 补充检查由 `spcgs.ipsgr_evaluate.high_resolution_check(run, cfg)` 执行；`spcgs.ipsgr_analysis.resolution_audit(run)` 记录尺度诊断。最后执行 `report`。

每个训练阶段可用 `--gpu` 指定物理 GPU。正式最终阶段均为 10,000 步，ISLB 的额外 10,000 步单独计入成本。

当前长实验由 `tools/ipsgr_pipeline.py` 管理后续阶段依赖。它接续本轮已启动的 Wan360 和两个基线，**不是通用的无人值守新实验启动器**；重启会依据 `runner_events.json` 接管本轮仍存活的阶段进程，并检查已退出阶段的完成证据；记录不匹配时应先核对进程与输出。L-BOOT 在 B-SEM 完成后使用 GPU 3，避免与 GPU 0 的 G-AUG 争用。

## 本轮并行记录

Wan360 最初单卡启动，随后采用 GPU 0/1/2 三个独立推理进程，GPU 3 同时训练基线。因为共享 GPU 的速度不同，队列末尾未启动任务在空闲卡之间转移；生成模型、参数、种子、源图像和任务签名均未改变。实际分配及变更记录保存在 `01_geometry/Wan360/parallel_assignment*.json`，汇总在 `parallel_completed.json`。`tools/ipsgr_wan_parallel.py` 是本轮队列接管器，包含本轮已启动进程的接管逻辑；不要把它直接用于其他运行。

`tools/ipsgr_geometry_prefetch.py` 在 CPU 上提前计算本轮已完成视频候选帧的特征与匹配。最终注册、选帧和点云过滤仍使用原几何实现。相机运动方向余弦只作为分析指标，不作为接受门槛。

## 评价约定

- RGB：6 张测试相机，完整 2048×1024 ERP；PSNR、SSIM、LPIPS-AlexNet v0.1 使用保存的浮点渲染。
- 主语义：原生 256×128；GT 最近邻采样；0–6 七类。有效 GT 但无高斯贡献的预测进入混淆矩阵第 8 列作为错误，不增加模型类别，也不剔除该像素。
- 总体 mIoU 来自累计混淆矩阵；逐图 mIoU 和图均值另存。
- 学习分类头与固定 CLIP 文本原型查询分开统计；后者仍经过学习的 32→512 映射，不声称完整开放词汇图像蒸馏。
- 阀门属于 `piping`，只评价其所在 ROI，不报告虚构的独立阀门类别 IoU。
- SAM2 的面积检查有效率不等于对象对应准确率；视觉核验的少量对象另行报告正确一致、错误一致和不一致。
- 修复底部和已合并的标签按冻结协议纳入评价；不依据测试结果重新改标签、切分或评价范围。

## 证据位置

`01_geometry` 保留全新 P0/P1、全部候选、注册/三角化证据和 80 个视频；`02_semantic_teacher` 保存教师与掩码梯度检查；`03_layout_bootstrap` 保存 ISLB 初始、中间和最终模型、OGR；`05_models` 保存六个独立模型和实际梯度日志；`04_cross_view` 关联 ICSC 中间数据；`06_evaluation` 保存全部测试浮点输出和指标；`07_industrial_analysis` 保存固定 ROI、对象身份与尺度诊断；`08_paper_figures`、`09_paper_tables` 保存图表及注册表；`10_report` 保存完成审计、复现环境与中文报告。

完成状态以模型 `outputs.json`、`training_audit.json`、实际评价文件和最终 `completion_audit.json` 为准，不以是否已启动进程为准。原始图像相关旧缓存和旧模型均不用于本轮训练。
