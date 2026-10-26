"""Export a review-only IRIS 18 train / 6 test proposal; never modify active splits.

The explicit candidate IDs prioritize perimeter/height coverage. Measurements
below use ONLY the proposed 18 training cameras as test-support references.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.analyze_iris_camera_coverage import font_setup, style, annotate_points, finish

TRAIN = "1-01 1-02 1-07 1-08 1-19 1-21 1-28 1-29 1-37 1-40 1-52 1-53 1-58 1-59 1-66 1-67 1-78 2-01".split()
TEST = "1-13 1-18 1-26 1-44 1-61 1-71".split()
BLUE, ORANGE, GRAY, GREEN = "#216CB0", "#D56A2F", "#8292A5", "#438C7A"
REASONS = {
    "1-01": "大Y端右侧低层覆盖", "1-02": "大Y端右侧高层覆盖",
    "1-07": "大Y端左侧高层覆盖；缺官方语义，保留待补标",
    "1-08": "大Y端左侧低层覆盖；缺官方语义，保留待补标",
    "1-19": "大Y段高层向中段过渡", "1-21": "大Y段低层向中段过渡",
    "1-28": "中段右侧高层，连接右侧支路", "1-29": "中段右侧低层，连接右侧支路",
    "1-37": "中段左侧低层覆盖", "1-40": "中段左侧高层覆盖",
    "1-52": "右侧支路高层端点", "1-53": "右侧支路低层端点",
    "1-58": "小Y段中右侧高层连接", "1-59": "小Y段中右侧低层连接",
    "1-66": "小Y段左侧高层边缘", "1-67": "小Y段左侧低层边缘",
    "1-78": "小Y端高层覆盖", "2-01": "小Y端低层补充；与1-78近同XY而高度不同",
    "1-13": "大Y段低层插值测试候选", "1-18": "大Y段高层插值测试候选",
    "1-26": "中段低层插值测试候选", "1-44": "中段高层插值测试候选",
    "1-61": "小Y段高层插值测试候选", "1-71": "小Y段低层插值测试候选",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def xyz(rows):
    return np.array([[r[k] for k in ["x_m", "y_m", "z_m"]] for r in rows])


def inside_hull(hull, points):
    return np.all(points @ hull.equations[:, :-1].T + hull.equations[:, -1] <= 1e-8, axis=1)


def zone(row):
    return "大Y段" if row["y_m"] >= 15 else "中段" if row["y_m"] >= 12.5 else "小Y段"


def analyze(input_path, verify_sources):
    inventory = json.loads(input_path.read_text(encoding="utf-8"))
    rows, meta = inventory["cameras"], inventory["metadata"]
    lookup = {r["camera_id"]: r for r in rows}
    assert len(TRAIN) == len(set(TRAIN)) == 18
    assert len(TEST) == len(set(TEST)) == 6 and not set(TRAIN) & set(TEST)
    assert all(n in lookup for n in TRAIN + TEST)
    train, test = [lookup[n] for n in TRAIN], [lookup[n] for n in TEST]
    assert all(r["is_main_core"] or r["candidate_category"] == "near_supplement" for r in train)
    assert all(r["is_main_core"] and r["has_official_semantics"] for r in test)
    cut = meta["core_height_band_cut_m"]
    T, Q = xyz(train), xyz(test)
    tlow, qlow = T[:, 2] < cut, Q[:, 2] < cut
    assert (int(tlow.sum()), int(qlow.sum())) == (9, 3)
    verified = []
    if verify_sources:
        assert sha(meta["annotation_source"]) == meta["annotation_source_sha256"]
        for row in train + test:
            assert Path(row["rgb_source"]).is_file()
            assert sha(row["rmx_source"]) == row["rmx_sha256"], row["camera_id"]
            verified.append(row["camera_id"])
    hxy, h3 = ConvexHull(T[:, :2]), ConvexHull(T)
    assert inside_hull(hxy, Q[:, :2]).all() and inside_hull(h3, Q).all()
    supports = []
    for row, q in zip(test, Q):
        distances = np.linalg.norm(T - q, axis=1)
        same = np.flatnonzero((np.abs(T[:, 2] - q[2]) <= .35) & (distances <= 2.5))
        angles = np.sort(np.arctan2(T[same, 1] - q[1], T[same, 0] - q[0]))
        angular_gap = float(np.rad2deg(np.diff(np.r_[angles, angles[0] + 2*np.pi])).max())
        assert len(same) >= 3 and angular_gap < 180, row["camera_id"]
        triangles = []
        for ids in itertools.combinations(same, 3):
            a = np.vstack([T[list(ids), :2].T, np.ones(3)])
            if abs(np.linalg.det(a)) < 1e-6:
                continue
            weights = np.linalg.solve(a, np.r_[q[:2], 1])
            if weights.min() > 0:
                # Prefer local enclosing triangles; for equal radii prefer larger interior margin.
                triangles.append((float(distances[list(ids)].max()), -float(weights.min()),
                                  tuple(int(i) for i in ids), weights))
        assert triangles, row["camera_id"]
        radius, _, tri, weights = min(triangles, key=lambda t: (round(t[0], 7), t[1]))
        nearest = int(distances.argmin())
        same_nearest = int(same[np.argmin(distances[same])])
        supports.append({
            "test_id": row["camera_id"], "image_id": row["image_id"],
            "height_band": row["height_band_relative_to_core"], "zone": zone(row),
            "has_official_semantics": row["has_official_semantics"],
            "nearest_train_id": TRAIN[nearest], "nearest_train_3d_m": float(distances[nearest]),
            "nearest_same_height_train_id": TRAIN[same_nearest],
            "nearest_same_height_train_3d_m": float(distances[same_nearest]),
            "same_height_neighbors_2_5m": [TRAIN[i] for i in same],
            "same_height_neighbor_count": len(same),
            "same_height_neighbor_distances_3d_m": [float(distances[i]) for i in same],
            "max_xy_angular_gap_deg": angular_gap,
            "enclosing_triangle_train_ids": [TRAIN[i] for i in tri],
            "triangle_distances_3d_m": [float(distances[i]) for i in tri],
            "triangle_max_distance_3d_m": radius,
            "triangle_max_abs_dz_m": float(np.abs(T[list(tri), 2] - q[2]).max()),
            "xy_triangle_barycentric_weights": weights.tolist(),
            "inside_train_xy_hull": True, "inside_train_xyz_hull": True,
            "note": "Support measured against the proposed 18 training views only; position evidence, not feature-match/reconstruction validation",
        })
    core = [r for r in rows if r["is_main_core"]]
    X = xyz(core)
    D = cdist(X, T)
    same_layer_D = np.where((X[:, 2, None] < cut) == tlow[None, :], D, np.inf)
    other = np.array([r["camera_id"] not in TRAIN for r in core])
    # Deterministic MST is a distance diagnostic, not a match graph.
    mst = minimum_spanning_tree(cdist(T, T)).tocoo()
    metrics = {
        "proposal_status": "draft_for_user_review_not_activated", "validation_count": 0,
        "train_count": 18, "test_count": 6,
        "train_official_count": sum(r["has_official_semantics"] for r in train),
        "test_official_count": sum(r["has_official_semantics"] for r in test),
        "train_missing_official_ids": [r["camera_id"] for r in train if not r["has_official_semantics"]],
        "train_height_counts": {"lower": int(tlow.sum()), "upper": int((~tlow).sum())},
        "test_height_counts": {"lower": int(qlow.sum()), "upper": int((~qlow).sum())},
        "height_band_cut_world_z_m": cut,
        "train_zone_counts": {z: sum(zone(r) == z for r in train) for z in ["大Y段", "中段", "小Y段"]},
        "test_zone_counts": {z: sum(zone(r) == z for r in test) for z in ["大Y段", "中段", "小Y段"]},
        "source_group_counts": {"train": {"images_xphase_1": 17, "images_xphase_2": 1}, "test": {"images_xphase_1": 6}},
        "core_count": len(core), "nontraining_core_count": int(other.sum()),
        "core_nearest_train_quantile_probabilities": [0, .5, .9, 1],
        "core_nearest_train_3d_quantiles_m": np.quantile(D.min(1), [0, .5, .9, 1]).tolist(),
        "nontraining_core_nearest_train_3d_quantiles_m": np.quantile(D[other].min(1), [0, .5, .9, 1]).tolist(),
        "core_nearest_same_layer_train_3d_quantiles_m": np.quantile(same_layer_D.min(1), [0, .5, .9, 1]).tolist(),
        "worst_core_nearest_train_id": core[int(D.min(1).argmax())]["camera_id"],
        "core_inside_training_xy_hull_count": int(inside_hull(hxy, X[:, :2]).sum()),
        "core_xy_hull_area_m2": float(ConvexHull(X[:, :2]).volume),
        "train_xy_hull_area_m2": float(hxy.volume),
        "train_mst_max_edge_3d_m": float(mst.data.max()),
        "train_mst_edges": [{"a": TRAIN[i], "b": TRAIN[j], "distance_3d_m": float(d)} for i,j,d in zip(mst.row, mst.col, mst.data)],
        "test_nearest_train_range_m": [min(s["nearest_train_3d_m"] for s in supports), max(s["nearest_train_3d_m"] for s in supports)],
        "test_nearest_same_height_train_range_m": [min(s["nearest_same_height_train_3d_m"] for s in supports), max(s["nearest_same_height_train_3d_m"] for s in supports)],
        "test_enclosing_triangle_max_radius_m": max(s["triangle_max_distance_3d_m"] for s in supports),
        "test_all_inside_train_xy_and_xyz_hulls": True,
        "test_support_analysis_radius_m": 2.5, "same_height_abs_dz_tolerance_m": .35,
        "support_threshold_note": "Descriptive proposal QA only; no training/SfM acceptance rules changed. Previous full-pool analysis used 2m; this sparse-subset review explicitly uses 2.5m and reports all measured distances.",
        "source_inventory": str(input_path.resolve()), "source_inventory_sha256": sha(input_path),
        "source_annotation_sha256": meta["annotation_source_sha256"],
        "source_project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "analysis_script_sha256": sha(__file__),
        "source_verified_selected_ids": verified,
        "training_started": False, "active_protocol_modified": False,
    }
    selected = []
    for role, names in [("train", TRAIN), ("test", TEST)]:
        for index, name in enumerate(names, 1):
            r = lookup[name]
            selected.append({"candidate_role": role, "role_index": index, "camera_id": name,
                             "image_id": r["image_id"], "source_group": r["source_group"],
                             "x_m": r["x_m"], "y_m": r["y_m"], "z_m": r["z_m"],
                             "height_band": r["height_band_relative_to_core"], "zone": zone(r),
                             "has_official_semantics": r["has_official_semantics"],
                             "official_face_count": r["official_face_count"],
                             "candidate_reason": REASONS[name], "rgb_source": r["rgb_source"],
                             "rmx_source": r["rmx_source"], "rmx_sha256": r["rmx_sha256"],
                             "previous_split_reference_only": r["previous_split_reference_only"],
                             "previous_train_index": r["previous_train_index"]})
    return inventory, train, test, supports, metrics, selected


def legend(fig, y=.895):
    handles = [Line2D([], [], marker="o", ls="none", color=BLUE, markerfacecolor=BLUE, label="训练候选：有官方标注", markersize=8),
               Line2D([], [], marker="o", ls="none", color=BLUE, markerfacecolor="white", label="训练候选：待补标", markersize=8),
               Line2D([], [], marker="*", ls="none", color=ORANGE, markerfacecolor=ORANGE, label="测试候选：全部有官方标注", markersize=13),
               Line2D([], [], marker="s", ls="none", color=BLUE, markerfacecolor=BLUE, label="方形：第二组补充 2-01", markersize=8)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, y), ncol=4, frameon=False, fontsize=10)


def training_points(ax, train, X, ids, size=70, coordinates=(0, 1)):
    for i in ids:
        row = train[i]
        ax.scatter(X[i, coordinates[0]], X[i, coordinates[1]], s=size,
                   marker="s" if row["group"] == 2 else "o",
                   facecolors=BLUE if row["has_official_semantics"] else "white",
                   edgecolors=BLUE, linewidths=1.6, zorder=4)


def figures(out, inventory, train, test, supports, metrics):
    font_setup()
    T, Q = xyz(train), xyz(test)
    core = [r for r in inventory["cameras"] if r["is_main_core"]]
    X = xyz(core)
    cut = metrics["height_band_cut_world_z_m"]
    low = T[:, 2] < cut
    qlow = Q[:, 2] < cut
    fig, axes = plt.subplots(1, 2, figsize=(16, 11))
    fig.subplots_adjust(left=.07, right=.97, bottom=.13, top=.81, wspace=.19)
    for ax, lower, name in zip(axes, [True, False], ["较低一层", "较高一层"]):
        style(ax, True)
        ix = np.flatnonzero(low == lower); iq = np.flatnonzero(qlow == lower)
        background = X[(X[:, 2] < cut) == lower]
        ax.scatter(background[:, 0], background[:, 1], c="#D6DEE7", s=23, zorder=1)
        hull = ConvexHull(T[ix, :2]); vertices = T[ix[hull.vertices], :2]
        ax.add_patch(Polygon(vertices, facecolor=BLUE, edgecolor=BLUE, alpha=.06, lw=1.5, zorder=0))
        ax.plot(*np.vstack([vertices, vertices[0]]).T, color=BLUE, alpha=.4, lw=1, ls="--", zorder=2)
        training_points(ax, train, T, ix)
        ax.scatter(Q[iq, 0], Q[iq, 1], s=200, marker="*", c=ORANGE, edgecolors="white", linewidths=.5, zorder=5)
        ax.set_xlim(24.8, 31.8); ax.set_ylim(8.6, 18.7)
        ax.set_title(f"{name} · 训练 9 张 / 测试 3 张", loc="left")
        positions = np.vstack([T[ix, :2], Q[iq, :2]])
        labels = [TRAIN[i] for i in ix] + [TEST[i] for i in iq]
        annotate_points(ax, positions, labels, [BLUE] * len(ix) + [ORANGE] * len(iq), fontsize=10)
        for y in [12.5, 15]:
            ax.axhline(y, color=GRAY, ls=":", lw=.7)
    legend(fig)
    finish(fig, out, "01_candidate_split_xy", "IRIS 候选划分 v1 · 18 张训练 + 6 张测试 · 无验证集",
           "训练：17 张第一组 + 1 张第二组；16 张有官方标注。测试：6 张均为第一组官方标注视角。仅候选，尚未启用。",
           "空心训练点 1-07 / 1-08 优先保证大Y端左侧两层覆盖，后续可补标。灰色为未选的第一组核心相机；虚线为同层训练 XY 凸包。\n6 张测试均位于实际18张训练相机的 XY 与 XYZ 凸包内；图中编号沿用相机分布分析，完整图像名见清单。")

    fig, axes = plt.subplots(2, 3, figsize=(19, 13))
    fig.subplots_adjust(left=.05, right=.97, top=.83, bottom=.16, wspace=.25, hspace=.42)
    train_lookup = {name: i for i, name in enumerate(TRAIN)}
    for ax, support, q in zip(axes.flat, supports, Q):
        ids = [train_lookup[n] for n in support["enclosing_triangle_train_ids"]]
        points = T[ids, :2]
        style(ax, True)
        ax.add_patch(Polygon(points, facecolor=GREEN, edgecolor=GREEN, alpha=.13, zorder=0))
        ax.plot(*np.vstack([points, points[0]]).T, color=GREEN, lw=1)
        for idx, d in zip(ids, support["triangle_distances_3d_m"]):
            ax.plot([q[0], T[idx, 0]], [q[1], T[idx, 1]], color=ORANGE, lw=1.1, ls="--", zorder=2)
        training_points(ax, train, T, ids, size=80)
        ax.scatter(q[0], q[1], marker="*", s=240, c=ORANGE, edgecolors="white", linewidths=.7, zorder=5)
        all_pos = np.vstack([points, q[:2]])
        xmin, ymin = all_pos.min(0) - .55; xmax, ymax = all_pos.max(0) + .55
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        labels = support["enclosing_triangle_train_ids"] + [support["test_id"]]
        annotate_points(ax, all_pos, labels, [BLUE]*3 + [ORANGE], fontsize=10)
        height = "低层" if support["height_band"] == "lower" else "高层"
        ax.set_title(f"{support['test_id']} · {support['zone']} / {height}\n最近训练 {support['nearest_train_3d_m']:.2f}m；三角支持最远 {support['triangle_max_distance_3d_m']:.2f}m", loc="left", fontsize=11)
        ax.text(.02, -.18, "3D 距离：" + " / ".join(f"{d:.2f}m" for d in support["triangle_distances_3d_m"]), transform=ax.transAxes, fontsize=8.8, color=GRAY)
    finish(fig, out, "02_test_training_support", "逐个检查测试支持 · 仅使用拟选的18张训练图像",
           "每颗橙色星对应一张测试图像；绿色三角形由三个近似同高度的实际训练候选构成，测试 XY 在其内部。",
           "三角顶点与测试之间的距离均为3D欧氏距离；六组最远支持距离≤2.43m，最大同组高度差见 test_support.csv。\n本次采用2.5m邻域、|ΔZ|≤0.35m描述稀疏候选支持，区别于上一份完整候选池的2m统计；不是增加训练筛选条件，未运行匹配或训练。")

    fig, (ax, distances_ax) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1.7, 1]})
    fig.subplots_adjust(left=.07, right=.97, top=.82, bottom=.14, hspace=.43)
    style(ax)
    ax.scatter(X[:, 1], X[:, 2], c="#D9E0E8", s=22, zorder=1)
    training_points(ax, train, T, np.arange(18), coordinates=(1, 2))
    ax.scatter(Q[:, 1], Q[:, 2], c=ORANGE, marker="*", s=170, zorder=5)
    ax.set_xlim(8.6, 18.7); ax.set_ylim(151.54, 152.74)
    ax.set_xlabel("世界坐标 Y / m"); ax.set_ylabel("世界坐标 Z / m")
    ax.set_title("上下层均有训练9张、测试3张；两层分别覆盖大Y / 中 / 小Y段", loc="left")
    annotate_points(ax, np.vstack([T[:, [1,2]], Q[:, [1,2]]]), TRAIN + TEST, [BLUE]*18 + [ORANGE]*6, fontsize=8.4)
    style(distances_ax)
    positions = np.arange(6)
    distances_ax.bar(positions-.2, [s["nearest_train_3d_m"] for s in supports], width=.38, color=BLUE, label="最近训练相机（3D距离）")
    distances_ax.bar(positions+.2, [s["triangle_max_distance_3d_m"] for s in supports], width=.38, color=GREEN, label="包围三角形的最远训练相机")
    distances_ax.set_xticks(positions, [s["test_id"] for s in supports]); distances_ax.set_ylim(0, 2.8)
    distances_ax.set_ylabel("距离 / m"); distances_ax.set_xlabel("测试候选编号")
    distances_ax.legend(loc="upper right", frameon=False, ncol=2)
    legend(fig)
    finish(fig, out, "03_height_and_support_distances", "高度覆盖与测试邻距 · 候选划分复核",
           "较低层训练高程151.643–151.822m；较高层训练高程152.509–152.592m。Z为世界高程，并非相机离地高度。",
           "6张测试到最近训练相机为0.66–1.39m；每张测试都有3个近似同高度的训练位置在XY上包围它。\n位置包围关系不等于无遮挡或匹配成功；本次仅提出候选，后续可结合这些图像再调整。")


def contacts(out, rows, name, title, source_root):
    columns, nrows = 3, (len(rows)+2)//3
    fig, axes = plt.subplots(nrows, columns, figsize=(18, nrows*3.55 + 1.6), squeeze=False)
    fig.subplots_adjust(left=.035, right=.97, top=.84 if nrows == 2 else .89, bottom=.07, hspace=.34, wspace=.1)
    evidence = []
    for ax, row in zip(axes.flat, rows):
        path = source_root / "datasets/iris_full_A_official_2k/images" / (row["image_id"] + ".png")
        assert path.is_file(), path
        with Image.open(path) as im:
            im.thumbnail((800, 400))
            ax.imshow(np.asarray(im.convert("RGB")))
        official = "官方语义" if row["has_official_semantics"] else "待补标"
        layer = "低层" if row["height_band_relative_to_core"] == "lower" else "高层"
        ax.set_title(f"{row['camera_id']} · {layer} · {official}\n{row['image_id']}", fontsize=10, loc="left", color=BLUE if row["camera_id"] in TRAIN else ORANGE)
        ax.axis("off")
        evidence.append({"camera_id": row["camera_id"], "rgb_2k_preview_source": str(path), "sha256": sha(path)})
    finish(fig, out, name, title, "服务器已有2K全景RGB，仅缩略显示；相机编号对应分布图，图像名可对照清单。", "所有图像仅作为候选展示。本次没有运行任何 M1–M5 训练、标签生成或重建。")
    return evidence


def reports(out, selected, supports, m):
    train_rows = [r for r in selected if r["candidate_role"] == "train"]
    test_rows = [r for r in selected if r["candidate_role"] == "test"]
    def table(rows):
        return "\n".join(f"| {r['role_index']:02d} | {r['camera_id']} | `{r['image_id']}` | {'低' if r['height_band']=='lower' else '高'} | {'有' if r['has_official_semantics'] else '**无，待补标**'} | {r['candidate_reason']} |" for r in rows)
    support_table = "\n".join(f"| {s['test_id']} | {s['nearest_train_id']} | {s['nearest_train_3d_m']:.3f} | {'、'.join(s['enclosing_triangle_train_ids'])} | {s['triangle_max_distance_3d_m']:.3f} | {s['triangle_max_abs_dz_m']:.3f} |" for s in supports)
    text = f"""# IRIS 18张训练 + 6张测试候选划分 v1

**状态：供用户查看的候选方案，未启用。无验证集，未启动训练，未修改现有 data/protocol.json、数据集划分、模型或标签。**

## 选择结果

- 训练18张：低层9、高层9；17张来自第一组，1张来自第二组。16张有官方语义，1-07/08缺官方语义，保留待补标。
- 测试6张：低层3、高层3；全部来自第一组，全部有官方语义。
- 沿Y轴三个空间段，训练预算为6/4/8，测试为2/2/2。小Y段安排8张，是为了同时覆盖左侧边缘、右侧支路和小Y端，不机械按每段6张分配。
- 官方状态来自原始CAD投影COCO，未使用SAM或自动候选掩码作为官方标注；有官方记录不代表完整像素真值。

## 选择理由与覆盖

先保留大Y端左右两侧、两层中段连接、右侧支路、左下侧和小Y端，再放置位于内部的测试视角。训练1-07/08是大Y端左侧的一对上下层位置，即使缺官方语义也保留，避免因追求标注率而丢失外围覆盖。第二组仅使用2-01，补足第一组1-78相近XY处的低层拍摄位置；第三、四组未纳入。

18张训练位置相对于第一组连续核心76台相机的最近邻3D距离最大为 **{m['core_nearest_train_3d_quantiles_m'][-1]:.3f}m**。排除本方案已经选作训练的17台第一组相机后，剩余59台核心相机的最近训练距离中位数 **{m['nontraining_core_nearest_train_3d_quantiles_m'][1]:.3f}m**、90%分位数 **{m['nontraining_core_nearest_train_3d_quantiles_m'][2]:.3f}m**。同层最近训练距离的最大值也为 {m['core_nearest_same_layer_train_3d_quantiles_m'][-1]:.3f}m。76台核心中68台位于训练XY凸包内，其余边缘点仍保留在分析数据中；这里没有声称全部原始视角都被严格包围。

训练相机的空间最小生成树最长边为 {m['train_mst_max_edge_3d_m']:.3f}m，仅反映采样间隔。它不是特征匹配图，本次没有计算几何匹配或确认重建成功。

## 训练候选名单

| 序号 | 相机编号 | 图像ID（原始文件后缀.jpg） | 层 | 官方语义 | 候选理由 |
|---:|---|---|---|---|---|
{table(train_rows)}

## 测试候选名单

| 序号 | 相机编号 | 图像ID（原始文件后缀.jpg） | 层 | 官方语义 | 候选理由 |
|---:|---|---|---|---|---|
{table(test_rows)}

## 测试支持：仅使用这18张训练候选

6张测试均位于拟选训练相机的 **XY凸包和三维XYZ凸包内部**。各测试还有三个近似同高度的训练位置在XY上形成包围三角形，三角形重心坐标均为正，且这些训练点与测试的3D距离均不超过2.43m。

| 测试 | 最近训练 | 最近距离/m | XY包围三角形的训练编号 | 三角最远3D距离/m | 三角最大高度差/m |
|---|---|---:|---|---:|---:|
{support_table}

所有邻居都来自TRAIN清单，未把其余测试视角或未选视角算作训练支持。完整邻居、距离、XY三角形权重及凸包检查保存在 `test_support.csv/json`。

为描述稀疏18张子集，本次统计邻域使用2.5m、近似同高度使用 |ΔZ|≤0.35m；前一份完整76台候选池分析使用的是2m，因此不能把两份候选支持计数直接等同。这里明确报告实际距离，阈值只用于分析展示，不写入SfM或训练的拒绝规则。XY局部三角包围与全体训练相机的XYZ凸包内部检查分别计算，并未声称每个同层三角形在三维中都包含测试点。

Z来自原始RMX毫米位置换算为米，是测量坐标高程；低/高层分界为 {m['height_band_cut_world_z_m']:.3f}m，不是相机离地高度。所有选择均为2023-05-11同日拍摄。

## 文件与使用范围

- `01_candidate_split_xy.png/pdf`：两层分别标注训练、测试及待补标点。
- `02_test_training_support.png/pdf`：逐张展示6个测试视角的训练包围三角形。
- `03_height_and_support_distances.png/pdf`：高度覆盖和测试距离。
- `04_train_rgb_contact_sheet.png/pdf`、`05_test_rgb_contact_sheet.png/pdf`：本次24张实际RGB缩略图。
- `selected_cameras.csv`：24行完整图像名、来源、XYZ、角色、官方状态、候选理由及旧划分追溯。
- `candidate_train18.txt`、`candidate_test6.txt`：候选图像ID列表；当前没有接入数据集读取配置，不创建val.txt。
- `candidate_split.json`：完整候选与指标；`coverage_metrics.json`、`test_support.csv/json`：可复核测量。

本次只提出一版方案，不声称它是全局最优划分，也不以这些位置统计替代可见结构、遮挡或实际匹配质量。审核后若决定采用新划分，旧模型和旧稀疏点云包含旧训练视角信息，需按新训练集建立独立协议和初始化，不能直接作为这6张新测试的无泄漏评估结果。

复现：

```bash
cd /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023
.venv/bin/python tools/propose_iris_sparse_split.py \\
  --inventory runs/iris_camera_analysis_20261025/camera_inventory.json \\
  --source-root /mnt/sda/ubuntu/lkx/od \\
  --output runs/iris_camera_analysis_20261025/candidate_18_6_NEW_ID
```
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    def html_table(rows):
        body = "".join(f"<tr><td>{r['camera_id']}</td><td>{html.escape(r['image_id'])}.jpg</td><td>{'低层' if r['height_band']=='lower' else '高层'}</td><td>{'有' if r['has_official_semantics'] else '无，待补标'}</td><td>{html.escape(r['candidate_reason'])}</td></tr>" for r in rows)
        return f"<table><thead><tr><th>相机编号</th><th>图像名</th><th>高度</th><th>官方语义</th><th>候选理由</th></tr></thead><tbody>{body}</tbody></table>"
    figures = [("01_candidate_split_xy", "候选相机分布"), ("02_test_training_support", "6张测试的训练支持"),
               ("03_height_and_support_distances", "高度与距离复核"),
               ("04_train_rgb_contact_sheet", "18张训练候选图像"), ("05_test_rgb_contact_sheet", "6张测试候选图像")]
    gallery = "".join(f'<section><h2>{title}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy" alt="{title}"></a><p><a href="{name}.png">PNG 原图</a> · <a href="{name}.pdf">PDF</a></p></section>' for name,title in figures)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS 18+6 候选划分 v1</title><style>
body{{margin:0;background:#f3f6fa;color:#24394f;font:16px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1380px;margin:auto;padding:28px}}section{{background:white;border-radius:12px;padding:20px;margin:24px 0}}a{{color:#216cb0}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #dce3ec;padding:9px;text-align:left}}th{{background:#edf2f8}}nav a{{display:inline-block;margin-right:20px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.8 "Microsoft YaHei",sans-serif}}.summary{{background:white;border-left:5px solid #216cb0;padding:14px 20px}}</style>
<main><h1>IRIS 候选划分 v1：18张训练 + 6张测试</h1><p>无验证集 · 仅供审核 · 未启动训练 · 未修改当前数据集协议</p><div class="summary"><b>训练上下层各9张，测试上下层各3张。</b><p>训练16/18张有官方语义，测试6/6张有官方语义。1-07/08优先保留空间覆盖，后续可补标；第二组仅补充2-01。每个测试都有这18张训练中的局部位置支持。</p></div>
<nav><a href="selected_cameras.csv">24张图像清单CSV</a><a href="candidate_split.json">完整JSON</a><a href="candidate_train18.txt">训练名单</a><a href="candidate_test6.txt">测试名单</a><a href="test_support.csv">测试支持CSV</a><a href="REPORT.md">详细报告</a></nav>
<section><h2>训练候选18张</h2>{html_table(train_rows)}<h2>测试候选6张</h2>{html_table(test_rows)}</section>{gallery}<section><h2>分析全文</h2><pre>{html.escape(text)}</pre></section></main></html>'''
    (out / "report.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    inventory, train, test, supports, metrics, selected = analyze(a.inventory, verify_sources=True)
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "coverage_metrics.json", metrics)
    write_json(a.output / "test_support.json", supports)
    write_csv(a.output / "test_support.csv", supports)
    write_csv(a.output / "selected_cameras.csv", selected)
    write_json(a.output / "candidate_split.json", {"metadata": metrics, "selected_cameras": selected, "test_support": supports})
    for name, rows in [("candidate_train18.txt", train), ("candidate_test6.txt", test)]:
        (a.output / name).write_text("".join(r["image_id"] + "\n" for r in rows), encoding="utf-8")
    figures(a.output, inventory, train, test, supports, metrics)
    evidence = contacts(a.output, train, "04_train_rgb_contact_sheet", "18张训练候选 · 实际全景图像", a.source_root)
    evidence += contacts(a.output, test, "05_test_rgb_contact_sheet", "6张测试候选 · 全部具有官方语义", a.source_root)
    write_json(a.output / "rgb_preview_provenance.json", evidence)
    reports(a.output, selected, supports, metrics)
    print(json.dumps({"output": str(a.output), "train": TRAIN, "test": TEST,
                      "train_official": metrics["train_official_count"], "test_official": metrics["test_official_count"],
                      "test_nearest_distance_range": metrics["test_nearest_train_range_m"],
                      "max_core_nearest_train_distance": metrics["core_nearest_train_3d_quantiles_m"][-1]}, indent=2))


if __name__ == "__main__":
    main()
