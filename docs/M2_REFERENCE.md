# M2 实现依据与差异

本实现是基于公开材料的全景适配，不能称为 SPC-GS 官方代码的完整复现。2026-10-23 检查 [SPC-GS 仓库](https://github.com/gbliao/SPC-GS) 时，公开内容为 README 和展示资源，没有完整训练及预处理实现。README 的预处理数据下载链接也不等于代码。论文依据为 [SPC-GS §3.1、附录 B.2](https://arxiv.org/html/2503.12535v1)。

## 可逐项核对的代码

[3D-OVS](https://github.com/Kunhao-Liu/3D-OVS/tree/f77e2fc3fc475fe9299ceb5c3c9c87228a2b19b3) 的三个参考文件及 MIT 许可保存在 `references/3d_ovs/`，版本及 Git blob ID 见 `source.json`。这些文件不作为运行时依赖。

| 原实现 | 本项目 | 保留内容/差异 |
|---|---|---|
| `extract_clip_feature.py`：图像缩小 8 倍 | `spcgs/clip_dense.py::visual_cache` | 对每个透视面执行同样的 PIL resize；默认 1024→128 |
| 三个 patch 尺度 `min(H,W)//[5,7,10]` | `patch_boxes` | 原样采用，默认为 25/18/12 |
| 步长 `p//4`，补足最后一行/列 | `patch_boxes` | 原循环边界；四边独立 `randint(0,p//4)`，本项目固定 seed |
| 原始 CLIP CLS 向量加到整个矩形，按覆盖次数除 | `accumulate_rectangles` | 矩形差分与二维前缀和等价实现；float64 累加避免相消，单元测试对照逐像素加法 |
| OpenAI CLIP ViT-B/16 | `Encoder` | 保留现有本地 HF ViT-B/16 权重与 Processor；没有另换模型。HF 与 OpenAI 预处理包装不声称逐位一致 |
| `llff.py::read_clip_features_and_relevancy_maps` | `unit`、`spatial_relevancy` | 每尺度按 512 维归一化，乘归一化 T，逐类别、逐尺度对全空间 min-max；本项目增加常量图的零分母保护 |
| 保存所有尺度 | `raw_multiscale.npy`、`maps.npz` | 保留未做通道归一化的覆盖均值及每尺度 cosine/relevancy |
| `train_seg.py` 学习 scale selector | `visual_cache`、`relevancy_cache` | **未移植 selector**；为现有单个 512 维监督接口采用三尺度等权平均。F 平均归一化尺度后再归一化；S 平均各尺度 relevancy |
| `train_seg.py` 特征余弦 + relevancy JS | `spcgs_native.cpp::semantic_loss` | 使用 SPC-GS 的 `Lcos + Lce`；不声称复制了 3D-OVS 的完整训练方法，不移植 DINO 相关损失 |

## 全景适配

ERP 转为前、右、后、左、上、下六个 100° 透视图。以像素中心采样，+Y 向下，ERP 水平补一列周期边界，垂直夹到有效像素。每面分别提取三尺度密集特征，再按球面方向反采样回 ERP，重叠区以透视深度 `z^4` 权重融合。这里的双线性插值用于几何重投影；每个面内的特征来自真实重叠矩形累积，不是 5×5 整图向量网格插值。

默认每面 361+841+1600 个 patch，每 ERP 共 16,812 个。保存逆映射网格、面融合权重、全部 patch 边界和覆盖计数。双向投影测试包含接缝和两极；方向通道重建误差上限 0.0006，另一个直接逆映射测试界限 0.0011（128 像素面的半像素边缘夹取）。这是坐标正确性检查，不是语义准确率。

额外的 2048 像素透视面诊断保留在独立 run 中；由于整数 patch/stride 改变，每 ERP 为 15,822 个 patch。它不替代默认配置，不能据此宣称质量普遍改善。

`clip_fov` 可配置，默认仍为 100°，并参与 F 的缓存签名。额外 150° 诊断只改变视野，其 patch 构造保持不变；更宽视野增加上下文，同时也降低物体在面内的空间分辨率。两种 FOV 均经过实际正反投影测试。诊断中 raw cosine 的部分柱/管响应改善，但最终 S 仍明显混淆，所以没有将诊断配置用于正式完整训练。

## F / T / S 与训练

- F：`[512,128,256]`，float16；三尺度原始视觉均值另存 `[3,512,128,256]`。缓存键只依赖 RGB、模型和提取配置，不含词表或位姿。
- T：逐 prompt 保存原始和归一化 512 维向量；类内平均归一化 prompt 向量后再归一化，得到 `[20,512]` float32。保存 class ID、矩阵行号、prompt 行号。
- S：`[20,128,256]` float16 连续 relevancy；另存原始 cosine、三个尺度、概率、argmax、熵、前两类差值。F+T 独立缓存，修改文本不重新编码图片。
- C++ 明确读取 S；默认先插值 S 至语义渲染分辨率，再对类别做 temperature=1 softmax，使用软目标交叉熵。这借用了 3D-OVS 的类概率转换，同时使用 SPC-GS 的 CE，**是公开实现缺失时的明确近似**。
- 旧提取器为兼容既有实验输出独立 S 文件，显式标记 `hard_argmax`，保留旧目标；C++ 同样从磁盘 S 取目标，不再隐式重算 F/T 标签。
- 保留每高斯 32 维、32→512 映射、原有分类映射头、RGB/语义联合渲染、文本查询、SGI/后续 SPC 接口。

SPC-GS 附录给出语义参数 lr=0.0025、映射头 lr=0.0005，SGI 与后续语义阶段各 10k。本任务只修改/验证 M2 及接入 SGI；不重做 M3–M5。

## 数据依据与适用边界

已查阅 [IRIS 官方介绍](https://centreborelli.github.io/iris-vis/) 及 [IRIS-VIS 论文作者公开全文](https://www.researchgate.net/publication/390610296_IRIS-VIS_A_New_Dataset_for_Visibility_Estimation_in_an_Industrial_Environment)。原 CVF PDF 与 HAL 当次直接请求受限，因此论文 §3 的场景/数据内容通过作者上传的全文读取。论文介绍工业场景中密集管线、阀门和仪表导致的遮挡；CAD 与点云也会因临时物体、简化几何和遮挡空洞而不一致。这些说明支持使用官方类别作命名参考，不能把 CAD/官方 mask 当成本文 CLIP 监督。

本地使用 78 张 `images_xphase_1` ERP、56/6/16 划分和 RMX 位姿。它们是本项目实际目录/协议的检查结果，不将后续 IRIS-v2 的数量混入。只读取标注 JSON 的类别名称、ID 和图像关系，未读取分割 mask 用于训练。类别依据是 Codex 直接查看的 56 张训练联系表（ERP+六透视面）与 6 处放大复查。
