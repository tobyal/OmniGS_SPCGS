# M1 全视角批处理

`tools/run_m1_batch.py` 在 Linux 服务器执行三个阶段：先生成全部源视角的视频和图像，再并行执行各源视角的独立 M1，最后使用相同几何算法执行全局 M1。运动方向余弦仍只记录，不参与筛选。

## 当前正式任务

- Run ID：`m1_wan360_56_batch_20261023`。
- 56 个真实训练源视角，四方向，共 224 条视频；启动时复用 4 条，待生成 220 条。
- 每条视频 1024×512、49 帧、30 步，保存全部无损 PNG；SfM 使用第 1–4 帧，共 896 个候选。
- 生成使用空闲 GPU 0、2，各一个常驻 Wan360 模型进程，避免每个源视角重复加载权重。GPU 1、3 当时正在运行其他任务。
- 全部 224 条视频完成并检查对应图像和轨迹存在后，才允许进入 SfM。
- 当前 SfM 是 OpenMVG / OpenCV / SciPy CPU 实现，按源视角并行启动 4 个进程；每个进程继续使用全部 56 个真实训练视角作为允许的几何支持集及原有邻居选择。
- 最后执行全局 M1，复用独立阶段的特征与匹配缓存，保留原有跨源视角的轨迹去重、点去重和密度过滤。独立点云不直接拼接。
- 只执行 M1，不启动 M2–M5。

任务启动不代表实验完成。实际状态：

```text
/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023/runs/industrial_spcgs/m1_wan360_56_batch_20261023/batch_status.json
```

`status` 表示控制器状态，`phase` 表示当前阶段；`completed_videos/total_videos` 表示视频进度，`sfm_finished/sfm_failed` 表示独立重建进度，`global_metrics` 表示全局阶段进度或最终指标。只有 `status=completed` 且 `phase=completed` 才表示完整三阶段结束。

## 执行与恢复

```bash
cd /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023
.venv/bin/python -u tools/run_m1_batch.py \
  --run m1_wan360_56_batch_20261023 \
  --config configs/m1_wan360_56.json --gpus 0,2 --sfm-workers 4
```

正式任务通过独立进程会话在服务器后台运行，关闭 SSH 不会停止任务。顶层日志为 `logs/m1_wan360_56_batch_20261023.log`，启动命令与 PID 见 run 内的 `launch.json`。

同一 run 有进程锁，防止重复启动。中断后用同一配置重新执行上述命令，会复用完整的视频缓存和已完成的独立重建；失败视角保留输出和原因。全局 M1 始终通过现有管线计算，不能用独立点云数量相加代替。

`--stage generation` 只完成图像阶段；`--stage sfm` 要求所有图像已完成，否则在任何 SfM 子进程启动之前退出。SfM 子进程不分配 CUDA 设备。

## 每个源视角的独立数据

以下路径均相对于该 run：

| 路径 | 内容 |
|---|---|
| `views/<真实视角名>/source_manifest.json` | 源图、相机、四方向任务、候选帧索引和配置 |
| `views/<真实视角名>/generation/{forward,backward,left,right}/` | 对应方向的视频、49 张 PNG、条件轨迹和生成信息 |
| `views/<真实视角名>/01_sgi_sfm/` | 独立匹配、注册、选帧、三角化、过滤、点云和可视化 |
| `batch/view_configs/<真实视角名>.json` | 每个源视角的完整运行配置 |
| `batch/sfm_logs/<真实视角名>.log` | 对应独立 SfM 进程日志 |
| `batch/gpu_0.log`、`batch/gpu_2.log` | 两个 GPU 生成进程日志 |
| `batch/generation_complete.json` | 全部图像完成的阶段记录 |
| `batch/source_results.json` | 全部独立重建的成功/失败状态与指标 |
| `batch/global_sfm.log` | 全局去重重建日志 |
| `01_sgi_sfm/` | 全局 M1 输出，包含最终增强点云和每个新点的观测归属 |

生成目录通过符号链接指向工程内持久缓存 `runs/m1_cache/wan360/<signature>/`；视频和 PNG 实体保留在该缓存中，不删除或覆盖旧 run。不同源视角的内容使用独立签名。每个独立重建的 `enhanced.ply` 都包含原始点云，因此不能累加这些文件的点数作为总新增点数。

独立视角若没有通过现有几何条件的新点，其 M1 状态保留为 `failed`，其他视角继续。全局结果和独立失败数量分别记录，不能把失败视角计为成功。

## 启动前验证

验证记录：[results/m1_batch_validation_20261023.json](results/m1_batch_validation_20261023.json)。

- 已通过现有 2 个球面几何单元测试。
- 实际缓存 smoke run `m1_batch_cached_smoke_20261023` 完成三阶段。逐视角与全局增强 F32 均与此前 288 新点实验哈希完全一致，且没有生成新视频。
- `m1_batch_barrier_check_20261023` 是明确的失败路径测试：缺少 220 条视频时，`--stage sfm` 在启动任何独立或全局 SfM 前退出。该测试目录不计入正式实验结果。

批处理只将原生成函数拆分出无推理的任务规划入口，推理 worker、缓存签名和几何算法保持不变。
