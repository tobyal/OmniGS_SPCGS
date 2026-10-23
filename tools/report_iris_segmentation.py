"""Validate all current ERP labels and generate reproducible quality statistics."""
import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.segmentation import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(DEFAULT_RUN));a=p.parse_args();s=Store(a.run)
    observations={}
    for f in sorted((ROOT/'docs/industrial_scene_observations').glob('authored*.json')):
        for r in read(f)['records']:observations[r['name']]=r
    rows=[];total=2048*1024*len(s.items)
    for name,item in s.items.items():
        labels,source,state=s.arrays(name);f=s.folder(name);validate(labels,s.config)
        assert sha(item['rgb'])==item['rgb_sha256']
        assert Image.open(f/'revisions'/state['revision']/'labels.png').mode=='L'
        assert Image.open(item['rgb']).size==(2048,1024)
        assert np.isin(source,range(5)).all()
        assert sha(f/'revisions'/state['revision']/'labels.png')==read(f/'revisions'/state['revision']/'metadata.json')['labels_sha256']
        official=np.array(Image.open(f/'official_mapped.png'));coverage=read(f/'coverage.json')
        # All changed official labels must be explicit, inspected artifact removal.
        changed=(official!=255)&(labels!=official)
        assert np.all((labels[changed]==255)&(source[changed]==4))
        rows.append({**item,**state,'official_faces':sum(c['has_record'] for c in coverage),
          'face_mapped_coverage':{c['face']:c['mapped_coverage'] if c['has_record'] else None for c in coverage},
          'official_erp_coverage':float((official!=255).mean()),'official_changed_pixels':int(changed.sum()),
          'sam_labeled_pixels':int(((source==2)&(labels!=255)).sum()),
          'tripod_applied':read(f/'finalization.json')['tripod_applied'],
          'human_approved':state['review_state']=='approved' and state['approved_revision']==state['revision']})
    classes=[]
    for c in s.config['classes']:
        cl=str(c['id']);classes.append({**c,'rgb_observation_images':sum(bool(set(c['observation_keys'])&set(observations[n]['objects'])) for n in s.items),
          'candidate_images':sum(r['class_pixels'][cl]>0 for r in rows),'candidate_pixels':sum(r['class_pixels'][cl] for r in rows)})
    official_rows=[r for r in rows if r['official_faces']];new_rows=[r for r in rows if not r['official_faces']]
    summary={'images':len(rows),'official_images':len(official_rows),'no_official_images':len(new_rows),
      'official_faces':sum(r['official_faces'] for r in rows),'mean_ignore_fraction':float(np.mean([r['ignore_fraction'] for r in rows])),
      'official_group_ignore':float(np.mean([r['ignore_fraction'] for r in official_rows])),
      'new_group_ignore':float(np.mean([r['ignore_fraction'] for r in new_rows])),
      'approved_images':sum(r['human_approved'] for r in rows),'tripod_corrections':sum(r['tripod_applied'] for r in rows),
      'sam_labeled_pixels':sum(r['sam_labeled_pixels'] for r in rows),
      'class_stats':classes,'images_detail':rows}
    write(s.root/'quality_report.json',summary)
    out=['# IRIS 56 张全景语义整理与审核交付（2026-10-24）','',
      f"56/56 已落盘；27 张有官方六面记录，29 张无官方记录。当前 {summary['approved_images']} 张人工通过。",
      f"像素平均 ignore **{summary['mean_ignore_fraction']:.2%}**；有官方组 **{summary['official_group_ignore']:.2%}**，无官方组 **{summary['new_group_ignore']:.2%}**。",
      f"SAM2 最终保留 {summary['sam_labeled_pixels']:,} 个有效候选像素；{summary['tripod_corrections']} 张应用了视觉检查后的三脚架去除掩码。",
      '', '**这是覆盖全训练集的初步候选版本，不是完整人工真值。** 29 张新生成图像仍有较多未覆盖区域，尤其设备背面、细杆、复杂管群。没有像素级人工真值，不能报告 IoU 或把模型分数当准确率。',
      '', '## 类别分析', '',
      '已直接检查 56 张 ERP RGB、候选叠加图、官方梁/土建/机械/支撑示例和下视三脚架掩码。复用此前逐图原生视觉观察记录，并重新对照官方标签，最终取 7 个宽类，不沿用原 20 类。',
      '官方文件含 748 个 CAD 对象名称，归属于 16 个类别 ID；1500×1500 cubemap 使用 COCO 未压缩 RLE（列优先）。杂项 9 与消防 13 不做强制统一映射。RGB 频率为逐图观察记录中的出现次数，不是像素真值；候选频率另列。', '',
      '| ID | 类别（英文） | RGB 观察频率 | 候选含类图像 | 像素占全数据 | 官方 ID |',
      '|---|---|---:|---:|---:|---|']
    for c in classes:out.append(f"| {c['id']} | {c['zh']} / {c['name']} | {c['rgb_observation_images']}/56 | {c['candidate_images']}/56 | {c['candidate_pixels']/total:.2%} | {','.join(map(str,c['official_ids']))} |")
    out += ['', 'ID 255 为 ignore，不属于背景类别。出现频率由 `authored*.json` 的对象键汇总；固定灯具并入电气设备，土建红柱、台阶并入建筑；通风风道与输送管线分开。']
    for c in classes:out += ['',f"### {c['id']} · {c['zh']}",c['definition']+'。',f"纳入：{c['include']}。排除：{c['exclude']}。",c['rationale']]
    out += ['', '## 官方覆盖与逐图状态', '',
      '以下面覆盖按 F/R/B/L/U/D 顺序，为官方掩码映射到七类后在 1500² 面上的覆盖率；— 表示无面记录。覆盖不是准确率：CAD 标注可能忽略拍摄设备或实拍遮挡。详细每面、每对象 bbox、原始类别、未映射与冲突像素见 `coverage_audit.json`。', '',
      '| 序号 | 图像 ID | 官方面覆盖 F/R/B/L/U/D | 当前 ignore | 审核 |', '|---:|---|---|---:|---|']
    for r in rows:
        face='/'.join('—' if v is None else f'{v:.0%}' for v in r['face_mapped_coverage'].values())
        out.append(f"| {r['index']} | {r['id']} | {face} | {r['ignore_fraction']:.2%} | {'已通过' if r['human_approved'] else '待修正/待审核'} |")
    out += ['', '## 质量、修正与限制', '',
      '- 初次 SAM2 点提示、追加提示、被撤回的错误掩码及理由均记录在三个 `iris_segmentation_*json` 文档和逐图 audit 中。真实边界来自 SAM2.1 Hiera tiny，未使用矩形标签、CLIP 推断或训练。',
      '- 正确官方标签原样保留。跨面候选冲突留 ignore；仅填官方缺失区域。56 号白色柜体经直接视觉确认后补入原不可映射区域。三脚架修正使用实际 SAM2 二值掩码的提示点连通分量，不覆盖原始官方文件。',
      '- 已撤回的错误包括：台阶误作检修设施、红柱顶误作桥架、柜体下部误作地面、容器误作风道、管道误作设备、纸张误作柜体。',
      '- 仍明显不足：单点常只选中一扇柜门或一段管；100°透视边缘存在截断；细桥架、栏杆与遮挡处需要逐区域人工修正；23 张三脚架候选没有可靠采纳，仍需处理。官方组覆盖高不等于边界正确。',
      '- 56 张均有候选标签、像素来源、不可变版本和问题记录。没有替用户做人工通过；训练前应逐图确认，默认导出仅收人工通过版本。',
      '', '## 投影与验证', '',
      '正式标签始终为原始 ERP 2048×1024 单通道 uint8 PNG；ID 为 0–6 与 255。面标签与掩码使用最近邻取样，ERP 水平周期连续。RGB 展示允许双线性插值。',
      '用第 1 张六面原始参考 RGB 校验朝向，相关系数 0.987523–0.997320、MAE 1.20–3.29/255。这验证朝向和像素几何，不证明全部 CAD 边界准确。',
      '逐图检查尺寸、ID 范围、RGB SHA256、标签 SHA256、来源范围、官方修改范围；回归测试验证 ERP 六面/极点/接缝、RLE、保存并发冲突、审核失效和导出。Web 独立副本验证擦除、撤销/重做、保存、重开和真实 SAM2 预览/应用。',
      '', '## 数据与操作', '',
      '数据目录：`runs/iris_semantics_20261024/`。逐图完整审计、prompt、revision 元数据在服务器；候选导出包包含 labels、source、classes、manifest、metadata 和 audit。RGB 使用 manifest 的 ID 与 SHA256 对应，不混入彩色可视化文件。',
      '启动、补全、版本与导出命令见 [运行说明](IRIS_SEGMENTATION_REVIEW.md)。类别定义见 `configs/iris_segmentation_classes.json`；本次未启动 CLIP 特征提取或任何 SGI/SPC/Gaussian 训练。', '']
    path=ROOT/'docs/IRIS_SEGMENTATION_REPORT.md';path.write_text('\n'.join(out),encoding='utf-8')
    print({k:v for k,v in summary.items() if k not in ['class_stats','images_detail']})

if __name__=='__main__':main()
