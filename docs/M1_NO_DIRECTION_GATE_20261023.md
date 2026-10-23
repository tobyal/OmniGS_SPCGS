# M1：删除运动方向硬筛选的缓存对照实验

删除方向门槛后，left 第 1 帧被选中并参与三角化，最终贡献 59 个通过现有过滤的新增点。整体有效新增点由 233 增至 288（净增 55，+23.61%），重投影误差中位数由 1.6394 px 变为 1.6404 px。本次只运行 M1，没有生成新视频，也没有运行 M2–M5。

## 实现范围

- `spcgs/m1_geometry.py` 删除方向余弦低于阈值时拒绝注册的分支。
- `spcgs/m1_wan360.py` 删除默认 `min_direction_cosine` 配置。
- 保留 `motion_direction_cosine` 为分析指标，不参与筛选或评分；同步明确输出中的 `conditioning_role`。
- 特征匹配、位姿拟合、留出验证、每视频选帧评分、多视角三角化、共享轨迹去重和密度过滤保持不变。没有新增警告状态或其他筛选规则。

## 对照设置与复用

工程：`/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023`。

- 修改前 run：`m1_wan360_early`，代码提交 `cd58821fb6c32d9fd6779b82b4d8b2ba2853882d`。
- 修改后 run：`m1_wan360_early_no_direction_gate_20261023`。
- 同一配置 `configs/m1_wan360_early.json`，同一真实训练图 `2023-05-11_10.18.46`，同一 56 张真实训练几何支持集。
- 同一 forward/backward/left/right 四条视频，各取第 1–4 帧，共 16 个候选，seed=42。
- 4 条视频、16 组候选特征和 96 组真实—生成描述子匹配全部复用。

现有匹配缓存的键包含整个 `m1_geometry.py` 的文件哈希。因此本次先确认匹配函数 AST、特征指纹、阈值、种子、OpenCV 版本及旧缓存键完全一致，再将原匹配缓存复制到新文件哈希对应的键。该一次性操作保存在 `logs/prepare_m1_direction_replay.py`，没有更改产品缓存逻辑。执行后逐组核对 96 组匹配数组一致。

执行命令：

```bash
cd /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023
.venv/bin/python -m spcgs.pipeline \
  --run m1_wan360_early_no_direction_gate_20261023 \
  --config configs/m1_wan360_early.json --stage m1
```

## 实测结果

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 新生成视频 | 0 | 0 |
| 复用视频 | 4 | 4 |
| 通过注册的候选帧 | 7 / 16 | 9 / 16 |
| 最终选中生成视角 | 3 | 4 |
| 描述子匹配总数 | 29,675 | 29,675 |
| 选中视角的几何匹配数 | 4,492 | 6,575 |
| 三角化候选点 | 372 | 453 |
| 密度过滤前新增点 | 318 | 384 |
| 密度过滤剔除点 | 85 | 96 |
| 有效新增点 | 233 | **288** |
| 原始点数 | 3,979 | 3,979 |
| 增强点云总数 | 4,212 | **4,267** |
| 重投影误差中位数 | 1.6394 px | **1.6404 px** |
| 重投影误差 P90 | 3.3282 px | 3.3559 px |
| 重投影误差最大值 | 4.6442 px | 4.6442 px |

误差口径：对每个最终保留的新点，取其全部真实与生成观测中的最大球面角重投影误差，再在所有保留点上统计分位数；统一换算为宽 2048 像素的单位。原有三角化误差上限仍为 5 px。此结果说明本次小规模实验新增点数增加且误差统计基本持平，不代表完整训练后的渲染质量已经改善。

## left 的实际参与情况

| left 候选 | 注册内点 / 对应数 | 留出内点 / 总数 | 方向余弦（仅记录） | 修改后结果 |
|---|---:|---:|---:|---|
| 第 1 帧 | 64 / 89 | 17 / 22 | −0.4447 | 通过，按原评分选中 |
| 第 2 帧 | 37 / 63 | 6 / 15 | −0.5661 | 通过；同视频只选一帧 |
| 第 3 帧 | 16 / 54 | 4 / 13 | −0.7767 | 内点比例不足 |
| 第 4 帧 | 9 / 44 | 1 / 11 | −0.8436 | 内点不足、留出验证失败 |

选中帧为 forward 1、backward 2、left 1、right 2。left 第 1 帧产生 95 个三角化候选，59 个点在最终过滤后保留；这些点的重投影误差中位数为 1.5458 px，P90 为 3.0336 px，最大为 4.6176 px。

| 点的生成视角归属 | 修改前保留点 | 修改后保留点 |
|---|---:|---:|
| forward | 166 | 171 |
| backward | 41 | 41 |
| left | 0 | 59 |
| right | 26 | 17 |
| 合计 | 233 | 288 |

left 贡献 59 点与总体净增 55 点是不同口径。既有共享轨迹去重按所选视角顺序处理，密度过滤也会利用合并后的邻域；加入 left 会影响其他方向最终保留点的归属和数量。这些逻辑没有修改。

## 验证与输出

- 现有 2 个球面几何单元测试通过。
- 16 个候选的拟合位姿、内点、留出对应及全部数值诊断与旧 run 完全一致；注册拒绝原因仅移除了方向不一致项。
- 其他几何配置完全一致；所有 96 组匹配数组一致，16 组候选特征从缓存读取。
- 旧 run 的 JSON 输出和增强 F32 哈希未变；四条视频及其轨迹文件哈希未变。
- 原始 3,979 点的 XYZRGB 逐项相等，新点坐标有限；新 run 仅有 `01_sgi_sfm` 阶段。
- 新增强 F32 SHA256：`df5faac38ec71f653f30175d3d9f7cfe3f9a64f4340adac07930a6d3b38b5bdd`。

机器可读对照数据：[results/m1_no_direction_gate_20261023.json](results/m1_no_direction_gate_20261023.json)。

服务器输出目录：

```text
/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023/runs/industrial_spcgs/m1_wan360_early_no_direction_gate_20261023/01_sgi_sfm/
```

其中 `comparison_with_direction_gate.json` 保存核对结果；`selected_views.json`、`registration/`、`observations.json` 与 `triangulation.json` 保存视角及点级证据；`new_points.ply`、`enhanced.ply`、`enhanced_points.f32` 为点云输出。`preview/2023-05-11_10.18.46_left_f0001_real_generated.png` 和对应 `_matches.png` 可查看 left 图像与匹配，`preview/point_distribution.png` 展示新增点分布。

日志：`logs/m1_wan360_early_no_direction_gate_20261023.log`、`logs/m1_no_direction_gate_comparison_check.log`、`logs/m1_wan360_early_no_direction_gate_20261023_cache_audit.json`。
