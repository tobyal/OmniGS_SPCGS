# IRIS 语义审核运行说明

当前服务器工程：`/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023`。
当前数据：`runs/iris_semantics_20261024`。全部 56 张训练图均已保存候选；不要对这个目录重新初始化。
本流程仅处理二维标签，不执行 CLIP、SGI、SPC 或高斯训练。

## 启动 Web

服务器现有 `.venv`、`vendor/sam2` 和 `weights/sam2.1_hiera_tiny.pt` 可直接复用：

```bash
cd /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/iris_review_server.py --run runs/iris_semantics_20261024 --port 8876
```

保持 SSH 终端打开；需要后台运行时：

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/iris_review_server.py --run runs/iris_semantics_20261024 --port 8876 > logs/iris_review_8876.log 2>&1 &
echo $! > logs/iris_review_8876.pid
```

仅在服务没有运行时启动，避免重复占用端口。无 GPU 时可用 `--no-sam` 查看、手动编辑、审核及导出；SAM2 按需加载，使用约 1 个缓存透视 embedding。

Windows 本机另开 PowerShell 建立隧道（保持窗口或进程运行）：

```powershell
ssh -F NUL -N -i C:/Users/tobyal/.ssh/A6000/kxliu_ed25519 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8876:127.0.0.1:8876 kxliu@100.111.100.200
```

打开 <http://127.0.0.1:8876>。服务只监听服务器回环地址；无需开放公网端口。当前任务已经启动服务和本机隧道。

## 查看、编辑与审核

1. 总览按类别、来源、状态筛选，卡片显示官方六面覆盖、当前有效标签比例及问题数。有效覆盖不是准确率。
2. 打开图像，切换 RGB、当前叠加、彩色标签、官方映射、官方原始 16 类或新增/修改区域。右侧可隐藏单类；用缩放和平移检查细节。局部透视显示**已保存版本**，修改后先保存再查看。
3. 选择目标类，再用画笔、区域改类、多边形或擦除（255 ignore）。区域改类操作相同标签的连通域；ERP 左右边界连通。多边形双击闭合。撤销/重做保存最近 25 次操作。
4. SAM2 点选：左键正点、右键负点；提示应位于同一个 100°透视范围。点击“生成 SAM2”，检查白色预览，再点“应用掩码”。应用会使用所选目标类，可覆盖已有标签；错误可撤销。保存后才写入数据。
5. “保存修改”创建新 revision，保留旧版并撤销旧批准状态。未保存离开有提示。版本历史可恢复旧版为新的 revision。
6. 人工逐区域检查后，填写审核人姓名，点击“人工确认通过”并确认。通过只绑定当前保存版；任何后续保存都会失效。可标记问题、保存备注或撤销通过。自动补全和 Codex 视觉检查不算人工审核。
7. 默认仅导出通过版本；另可明确选择“全部候选（含未审核）”。当前无人工通过图像，因此默认包有 0 张标签是预期行为。

## 重现分割

不需要重跑已生成的 56 张。若要复现实验，使用**新目录**：

```bash
cd /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023
.venv/bin/python tools/prepare_iris_segmentation.py --run runs/iris_semantics_reproduce
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/complete_iris_segmentation.py --run runs/iris_semantics_reproduce
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/refine_iris_segmentation.py --run runs/iris_semantics_reproduce
.venv/bin/python tools/finalize_iris_segmentation.py --run runs/iris_semantics_reproduce
.venv/bin/python tools/report_iris_segmentation.py --run runs/iris_semantics_reproduce
```

初始化从原始 16 类 COCO RLE 导入，保留原始官方文件。后续步骤读取已直接观察 RGB 后写定的点提示和修正决定；已有完成标记的图像会跳过。修正脚本拒绝重建存在人工修改或批准的图像，人工审核开始后使用 Web 编辑。

提示及审核决策：

- `docs/iris_segmentation_prompts.json`：56 张逐图原生视觉点提示，ERP 像素坐标。
- `docs/iris_segmentation_corrections.json`：第一次候选检查，撤回错误与增加提示。
- `docs/iris_segmentation_final_decisions.json`：第二次检查及逐图三脚架决定。

分割调用真实 SAM2.1 Hiera tiny；点/框仅作为提示，正式标签边界来自模型二值掩码。不同类别候选重叠冲突保留 ignore；已映射官方标签优先保留。正式标签使用最近邻投影。三脚架处理只使用经检查的 SAM2 掩码中包含正点的连通分量，过滤掉不相连的误选物体；无可靠掩码的图像保留待修正。

## 导出与校验

```bash
.venv/bin/python tools/export_iris_segmentation.py --run runs/iris_semantics_20261024 --scope approved --output runs/iris_semantics_20261024/exports/approved.zip
.venv/bin/python tools/export_iris_segmentation.py --run runs/iris_semantics_20261024 --scope candidates --output runs/iris_semantics_20261024/exports/UNREVIEWED_candidates.zip
.venv/bin/python tools/report_iris_segmentation.py --run runs/iris_semantics_20261024
.venv/bin/python -m pytest tests/test_iris_segmentation.py -q
```

数据协议 `iris_erp_semantic_v1`：

```text
run/
  classes.json                 七类 ID/中英文/颜色/范围/官方映射/来源编码
  manifest.json                56 个训练 ID、RGB 路径和 SHA256
  coverage_audit.json           每面/每对象的原始覆盖、漏标和冲突信息
  projection_alignment.json    六面原始 RGB 对齐检查
  quality_report.json          逐图和逐类质量统计
  images/<id>/
    official_raw.png           原始官方 ID 的 ERP 投影，不修改
    official_mapped.png        初始七类 ERP 映射，不修改
    sam_masks/*.png            实际 SAM2 二值候选，包括被撤回版本
    sam_candidates.json        输入 SHA、checkpoint SHA、点、面和模型分数
    refinement.json            追加提示与修正记录
    finalization.json          最终候选整理决定
    revisions/<revision>/
      labels.png               正式标签：L / uint8 / 2048×1024 / ID 0–6、255
      source.png               同尺寸像素来源：0未知、1官方、2SAM、3人工、4视觉检查后的SAM修正
      metadata.json            父版本、时间、作者、操作、统计与 SHA
    state.json                 当前 revision、问题、审核状态、批准的 revision
    review_events.jsonl        明确人工审核事件（未审核图不存在）
  qa/                          RGB、候选与下视掩码检查图，不用作正式标签
```

导出 ZIP 包含 `labels/`、`source/`、`metadata/`、`audit/`、`classes.json`、`manifest.json`。manifest 明确每张 `human_approved`、当前 revision、RGB 与标签 SHA256；候选包明确包含未审核版本。RGB 原图不重复打包，按 ID 和 SHA 对应当前数据集。

导出检查尺寸、离散类别、来源和 RGB 对应；保存使用 revision 乐观锁，过期编辑返回冲突，须重开再处理。所有数据只新增版本，不覆盖原始官方标注。`labels.png` 不是彩色可视化，不能替换为叠加图。

## 本次验证

测试覆盖六面/极点/ERP 接缝、RLE 解码、不可变版本、过期保存/批准、编辑使批准失效、恢复、非法 ID、CSRF 与 RGB 变更拒绝导出。真实浏览器编辑与 SAM2 操作只在隔离副本 `runs/iris_ui_test_20261024` 执行，真实 56 张没有自动批准。

完整类别、56 张覆盖和质量限制见 [质量报告](IRIS_SEGMENTATION_REPORT.md)。
