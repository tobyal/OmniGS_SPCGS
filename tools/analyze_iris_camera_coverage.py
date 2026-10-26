"""Read-only IRIS camera/official-label inventory and static spatial analysis.

No training, reconstruction, label generation, or train/test split is performed.
All cross-group distances use original official RMX translations in meters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image

COLORS = {1: "#226BBA", 2: "#DF8732", 3: "#8D65B5", 4: "#279784"}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
CAT_COLORS = {"main_core": "#226BBA", "near_supplement": "#D38025",
              "transition_review": "#A36CAB", "main_outlier": "#BD4B4B",
              "remote": "#A8B0BB"}
INK, GRAY = "#24394F", "#607287"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_rmx(path):
    b = path.read_bytes()
    encoding = "utf-16" if b[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    s = b.decode(encoding)
    assert "translation vector (millimeters)" in s, path
    a = [list(map(float, line.split())) for line in s.splitlines()
         if line.strip() and not line.lstrip().startswith("#")]
    assert [len(x) for x in a] == [3, 3, 1], path
    assert abs(np.linalg.norm(a[1]) - 1) < 1e-6, path
    assert np.isfinite(np.concatenate(a)).all(), path
    return np.array(a[0]) / 1000, encoding


def max_angular_gap(vectors):
    if len(vectors) < 3:
        return 360.0
    angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return float(np.rad2deg(np.diff(np.r_[angles, angles[0] + 2 * np.pi])).max())


def inventory(root, project):
    source = root / "datasets/images_xphase_1_cubemap_semantics_1500/_source/annotations_segmentation_cubemaps_xphase.json"
    data = json.loads(source.read_text())
    anns = defaultdict(list)
    for ann in data["annotations"]:
        anns[ann["image_id"]].append(ann)
    face_records = defaultdict(list)
    for im in data["images"]:
        stem, face = Path(im["file_name"]).stem.rsplit("_", 1)
        assert face in "FRBLUD"
        face_records[stem].append({"face": face, "official_image_id": im["id"],
                                  "file_name": im["file_name"],
                                  "object_count": len(anns[im["id"]])})
    protocol = json.loads((project / "data/protocol.json").read_text())
    old_split = {name: split for split, names in protocol["splits"].items() for name in names}
    old_index = {name: i for i, name in enumerate(protocol["splits"]["train"], 1)}
    rows = []
    for group in range(1, 5):
        images = sorted((root / f"images_xphase_{group}").glob("*.jpg"))
        assert len(images) == 78, (group, len(images))
        for index, image in enumerate(images, 1):
            pose = root / "images_xphase_camera_info" / (image.stem + ".rmx")
            xyz, encoding = read_rmx(pose)
            fs = sorted(face_records.get(image.stem, []), key=lambda x: x["face"])
            labeled = [x for x in fs if x["object_count"] > 0]
            assert len({x["face"] for x in fs}) == len(fs)
            status = ("six_faces_with_objects" if len(labeled) == 6 else
                      "partial_faces" if labeled else "no_official_annotation")
            rows.append({"camera_id": f"{group}-{index:02d}", "image_id": image.stem,
                         "source_group": f"images_xphase_{group}", "group": group,
                         "group_index": index, "capture_date": image.stem[:10],
                         "x_m": float(xyz[0]), "y_m": float(xyz[1]), "z_m": float(xyz[2]),
                         "rgb_source": str(image), "rmx_source": str(pose),
                         "rmx_sha256": sha(pose), "rmx_encoding": encoding,
                         "has_official_semantics": bool(labeled), "official_status": status,
                         "official_face_records": fs, "official_face_count": len(fs),
                         "official_faces_with_objects": len(labeled),
                         "official_object_count": sum(x["object_count"] for x in fs),
                         "official_source": str(source),
                         "official_provenance": "IRIS official CAD-projected COCO RLE; SAM/review masks not consulted",
                         "official_pixel_completeness": "not_assumed_from_record_presence",
                         "previous_split_reference_only": old_split.get(image.stem, "outside_previous_protocol"),
                         "previous_train_index": old_index.get(image.stem)})
    assert len({r["image_id"] for r in rows}) == len(rows) == 312
    assert set(face_records).issubset({r["image_id"] for r in rows})
    X = np.array([[r[k] for k in ["x_m", "y_m", "z_m"]] for r in rows])
    G = np.array([r["group"] for r in rows])
    O = np.array([r["has_official_semantics"] for r in rows])
    main = np.flatnonzero(G == 1)
    dm = cdist(X[main], X[main])
    _, components = connected_components(dm <= 1.5, directed=False)
    core = main[components == np.bincount(components).argmax()]
    core_mask = np.isin(np.arange(len(rows)), core)
    excluded = main[~np.isin(main, core)]
    assert len(core) == 76 and len(excluded) == 2
    nn = dm.copy()
    np.fill_diagonal(nn, np.inf)
    d_all, d_core = cdist(X, X[main]), cdist(X, X[core])
    xy_all, xy_core = cdist(X[:, :2], X[main, :2]), cdist(X[:, :2], X[core, :2])
    nearest_core = core[d_core.argmin(axis=1)]
    z_sorted = np.sort(X[core, 2])
    gap_index = np.argmax(np.diff(z_sorted))
    height_cut = float((z_sorted[gap_index] + z_sorted[gap_index + 1]) / 2)
    low = X[:, 2] < height_cut
    hull = ConvexHull(X[core, :2])
    inside = np.all(X[:, :2] @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-8, axis=1)
    min_core = d_core.min(axis=1)
    category = np.array(["main_core" if i in core else "main_outlier" if G[i] == 1 else
                         "near_supplement" if min_core[i] <= 2 else
                         "transition_review" if min_core[i] <= 3 else "remote"
                         for i in range(len(rows))])
    # This is potential support in the candidate pool, NOT support by a selected training set.
    # Core references only, target excluded, same-height band; no annotation preference here.
    for i, row in enumerate(rows):
        refs = core[(core != i) & (d_core[i] <= 2.0) & (np.abs(X[core, 2] - X[i, 2]) <= 0.35)]
        angular_gap = max_angular_gap(X[refs, :2] - X[i, :2])
        interpolation = bool(len(refs) >= 3 and angular_gap < 180 and
                             category[i] in ["main_core", "near_supplement"])
        z_nearest = float(X[i, 2] - X[nearest_core[i], 2])
        if category[i] == "main_core":
            note = "主区域覆盖候选；优先覆盖各空间段与两层高度"
        elif category[i] == "near_supplement":
            note = "距连续主区域≤2米；优先核对RGB重叠后作为补充"
        elif category[i] == "transition_review":
            note = "距连续主区域2–3米；扩边复核候选，暂不作为默认测试"
        elif category[i] == "main_outlier":
            note = "第一组孤立相机；不用于扩大默认主区域边界"
        else:
            note = "距连续主区域>3米；本次紧凑实验不优先纳入"
        if interpolation:
            note += "；候选池存在同高度周向支持，可优先复核为插值测试或保留训练"
        if not O[i]:
            note += "；无官方语义，若选用可后续补标"
        if row["capture_date"] != "2023-05-11":
            note += "；2024跨日期拍摄，需另核对场景变化"
        row.update({"is_main_core": bool(core_mask[i]), "candidate_category": str(category[i]),
                    "z_relative_to_151_7_m": float(X[i, 2] - 151.7),
                    "height_band_relative_to_core": "lower" if low[i] else "upper",
                    "distance_to_xphase1_3d_m": float(d_all[i].min()),
                    "distance_to_xphase1_xy_m": float(xy_all[i].min()),
                    "nearest_xphase1_id": rows[main[d_all[i].argmin()]]["camera_id"],
                    "distance_to_main_core_3d_m": float(min_core[i]),
                    "distance_to_main_core_xy_m": float(xy_core[i].min()),
                    "nearest_main_core_id": rows[nearest_core[i]]["camera_id"],
                    "dz_to_nearest_main_core_m": z_nearest,
                    "inside_main_core_xy_hull": bool(inside[i]),
                    "candidate_core_support_count": len(refs),
                    "candidate_core_support_ids": [rows[j]["camera_id"] for j in refs],
                    "candidate_core_max_angular_gap_deg": angular_gap,
                    "interpolation_review_candidate": interpolation,
                    "candidate_notes": note})
    groups = []
    for g in range(1, 5):
        mask = G == g
        groups.append({"group": g, "total": int(mask.sum()), "official": int(O[mask].sum()),
                       "xyz_min_m": X[mask].min(axis=0).tolist(), "xyz_max_m": X[mask].max(axis=0).tolist(),
                       "minimum_distance_to_core_m": float(min_core[mask].min()),
                       "candidate_counts": dict(Counter(category[mask])),
                       "candidate_official_counts": dict(Counter(category[mask & O])),
                       "acquisition_dates": dict(Counter(r["capture_date"] for r in rows if r["group"] == g))})
    zones = []
    # Descriptive coverage budget zones, not an assignment of images.
    for name, mask in [("大Y段 Y≥15m", core_mask & (X[:, 1] >= 15)),
                       ("中段 12.5≤Y<15m", core_mask & (X[:, 1] >= 12.5) & (X[:, 1] < 15)),
                       ("小Y段 Y<12.5m", core_mask & (X[:, 1] < 12.5))]:
        zones.append({"name": name, "count": int(mask.sum()), "official": int((mask & O).sum()),
                      "lower": int((mask & low).sum()), "upper": int((mask & ~low).sum()),
                      "lower_official": int((mask & low & O).sum()),
                      "upper_official": int((mask & ~low & O).sum()),
                      "interpolation_pool": sum(r["interpolation_review_candidate"] for i, r in enumerate(rows) if mask[i]),
                      "interpolation_official": sum(r["interpolation_review_candidate"] for i, r in enumerate(rows) if mask[i] and O[i])})
    sensitivity = {}
    for threshold in [1, 1.25, 1.5, 2, 2.5, 3]:
        _, lab = connected_components(dm <= threshold, directed=False)
        sensitivity[str(threshold)] = sorted(np.bincount(lab).tolist(), reverse=True)
    summary = {"schema": "iris_camera_spatial_analysis_v1", "source_project_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
               "annotation_source": str(source), "annotation_source_sha256": sha(source),
               "official_definition": "Original COCO face records with non-empty annotation objects; CAD projection, not SAM or human pixel-perfect masks",
               "official_panorama_count": int(O.sum()), "official_face_records": len(data["images"]),
               "official_annotation_objects": len(data["annotations"]),
               "position_definition": "Original RMX translation in common survey coordinates; millimeters / 1000; no per-group normalization",
               "height_definition": "World Z elevation, NOT height above floor; plots may subtract declared 151.7 m datum",
               "camera_id_definition": "group-number, chronological sorted filename within group; old train index saved separately",
               "analysis_only": True, "final_split_created": False, "training_started": False,
               "groups": groups, "core_camera_ids": [rows[i]["camera_id"] for i in core],
               "core_count": len(core), "core_official_count": int(O[core].sum()),
               "core_bounds_xyz_m": [X[core].min(0).tolist(), X[core].max(0).tolist()],
               "main_outlier_ids": [rows[i]["camera_id"] for i in excluded],
               "core_graph_radius_m": 1.5, "core_components_sensitivity": sensitivity,
               "main_nearest_neighbor_3d_m_quantiles": np.quantile(nn.min(1), [0, .25, .5, .75, .9, 1]).tolist(),
               "candidate_distance_bands_m": {"near_supplement": [0, 2], "transition_review": [2, 3], "remote": [3, None]},
               "distance_band_note": "Descriptive analysis bands, not pipeline rejection rules; source images are unchanged",
               "core_height_band_cut_m": height_cut,
               "core_lower_count": int(low[core].sum()), "core_upper_count": int((~low[core]).sum()),
               "core_lower_median_z_m": float(np.median(X[core[low[core]], 2])),
               "core_upper_median_z_m": float(np.median(X[core[~low[core]], 2])),
               "interpolation_pool_definition": ">=3 OTHER core cameras within 2m 3D and |dZ|<=0.35m, largest XY bearing gap <180deg. Pool-level only; recompute after actual 18 training cameras are chosen. Does not prove visual overlap or occlusion-free support.",
               "interpolation_pool_count": sum(r["interpolation_review_candidate"] for r in rows),
               "interpolation_pool_official": sum(r["interpolation_review_candidate"] and r["has_official_semantics"] for r in rows),
               "coverage_zones": zones}
    return rows, summary, X, G, O, core, category, low, hull


def font_setup():
    for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/msyh.ttc"]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "pdf.fonttype": 42,
                         "axes.titlesize": 13, "axes.labelsize": 11, "savefig.facecolor": "#F6F8FB"})


def style(ax, xy=False):
    ax.set_facecolor("white")
    ax.grid(color="#E7ECF2", linewidth=.65)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#CFD8E2")
    ax.tick_params(colors=GRAY)
    if xy:
        ax.set_xlabel("世界坐标 X / m")
        ax.set_ylabel("世界坐标 Y / m")
        ax.set_aspect("equal", adjustable="box")


def finish(fig, out, name, title, subtitle, footer):
    fig.patch.set_facecolor("#F6F8FB")
    fig.text(.06, .962, title, fontsize=21, weight="bold", color=INK)
    fig.text(.06, .926, subtitle, fontsize=10.5, color=GRAY)
    fig.text(.06, .025, footer, fontsize=9, color=GRAY, va="bottom")
    fig.savefig(out / f"{name}.png", dpi=190)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def scatter_group(ax, X, G, O, ids, size=42, alpha=1):
    ids = np.asarray(ids, dtype=int)
    for g in range(1, 5):
        for official in [False, True]:
            ix = ids[(G[ids] == g) & (O[ids] == official)]
            if len(ix):
                ax.scatter(X[ix, 0], X[ix, 1], s=size, marker=MARKERS[g],
                           facecolors=COLORS[g] if official else "white", edgecolors=COLORS[g],
                           linewidths=1.05, alpha=alpha, zorder=4 if official else 3)


def annotate_points(ax, points, labels, colors=None, fontsize=8):
    """Place compact labels near markers with deterministic screen-space collision scoring."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    locations = ax.transData.transform(points)
    boxes = []
    axis_box = ax.get_window_extent(renderer)
    offsets = [(7, 5), (7, -11), (-26, 5), (-26, -11), (7, 15), (-26, 15),
               (15, -22), (-35, -22), (18, 2), (-40, 2), (0, 25), (0, -30),
               (30, 20), (-48, 20), (30, -27), (-48, -27)]
    scale = ax.figure.dpi / 72
    for i in np.argsort(locations[:, 1], kind="stable"):
        width = max(11, len(labels[i]) * fontsize * .56) * scale
        height = fontsize * 1.2 * scale
        best = None
        for dx, dy in offsets:
            x, y = locations[i] + np.array([dx, dy]) * scale
            box = (x - 1, y - 2, x + width + 1, y + height + 1)
            overlap = sum(max(0, min(box[2], b[2]) - max(box[0], b[0])) *
                          max(0, min(box[3], b[3]) - max(box[1], b[1])) for b in boxes)
            marker_hits = sum(box[0] - 3 < px < box[2] + 3 and box[1] - 3 < py < box[3] + 3
                              for j, (px, py) in enumerate(locations) if i != j)
            outside = max(0, axis_box.x0 - box[0]) + max(0, box[2] - axis_box.x1) + max(0, axis_box.y0 - box[1]) + max(0, box[3] - axis_box.y1)
            score = overlap * 5 + marker_hits * 70 + outside * 80 + abs(dx) + abs(dy)
            if best is None or score < best[0]:
                best = (score, dx, dy, box)
        _, dx, dy, box = best
        boxes.append(box)
        ax.annotate(labels[i], points[i], xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, color=colors[i] if colors is not None else INK,
                    ha="left", va="bottom", zorder=6,
                    arrowprops={"arrowstyle": "-", "color": "#9AA7B5", "lw": .45, "shrinkA": 1, "shrinkB": 3},
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])


def hull_patch(ax, X, core, hull):
    ax.add_patch(Polygon(X[core[hull.vertices], :2], closed=True, facecolor="#226BBA",
                         edgecolor="#226BBA", alpha=.09, lw=1.5, zorder=0))


def legends(fig, group=True, y=.89):
    handles = [Line2D([], [], marker=MARKERS[g], color=COLORS[g], linestyle="none",
                      markerfacecolor=COLORS[g], label=f"xphase_{g}", markersize=7) for g in range(1, 5)] if group else []
    handles += [Line2D([], [], marker="o", linestyle="none", color=INK, markerfacecolor=INK, label="实心：有官方语义"),
                Line2D([], [], marker="o", linestyle="none", color=INK, markerfacecolor="white", label="空心：无官方语义")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, y), frameon=False,
               ncol=len(handles), fontsize=10)


def plots(out, rows, s, X, G, O, core, cat, low, hull):
    font_setup()
    n = len(rows)
    all_ids = np.arange(n)
    core_mask = np.isin(all_ids, core)
    near = np.flatnonzero(cat == "near_supplement")
    review = np.flatnonzero(cat == "transition_review")
    interp = np.array([r["interpolation_review_candidate"] for r in rows])
    footer = "原始 RMX 坐标统一换算为米；官方标注仅依据原始 COCO，未使用 SAM。编号为组别-组内时间顺序，不是旧训练集序号。"

    fig, (ax, detail) = plt.subplots(1, 2, figsize=(17, 10), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(left=.06, right=.97, top=.82, bottom=.13, wspace=.18)
    for a in [ax, detail]:
        style(a, True)
        hull_patch(a, X, core, hull)
        scatter_group(a, X, G, O, all_ids, 30)
    ax.set_xlim(8, 37); ax.set_ylim(-.5, 20.5)
    detail.set_xlim(24.2, 32.1); detail.set_ylim(8.1, 19.5)
    ax.set_title("全区域 312 台相机 · 131 台有官方标注", loc="left")
    detail.set_title("连续主区域：76 台 / 44 台有标注", loc="left")
    for label, xy, textxy in [
        ("1-30：右侧孤立点", X[29, :2], (32.5, 6.7)),
        ("1-55：下方孤立点", X[54, :2], (12.0, 1.3)),
        ("xphase_1 连续主区域", (27.7, 15), (18.3, 19.2)),
        ("xphase_4 跨日期点\n2024-01-23（12 台）", (34.8, 16), (30.5, 20))]:
        ax.annotate(label, xy=xy, xytext=textxy, fontsize=10, color=INK,
                    arrowprops={"arrowstyle": "->", "color": GRAY}, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    detail.text(.03, .03, "俯视重叠点通常对应不同高度\n逐台编号见 02 / 06 图", transform=detail.transAxes, fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#D0D8E2", "alpha": .95})
    legends(fig)
    finish(fig, out, "01_xy_overview", "IRIS · 四组相机空间分布与官方标注覆盖",
           "xphase_1 是主要研究区域；淡蓝多边形仅表示主区域相机 XY 凸包，不代表墙体或可见空间边界。", footer)

    fig, axes = plt.subplots(1, 2, figsize=(15, 11))
    fig.subplots_adjust(left=.07, right=.96, top=.82, bottom=.12, wspace=.2)
    for a, mask, name in zip(axes, [low, ~low], ["较低一层", "较高一层"]):
        ids = core[mask[core]]
        style(a, True); hull_patch(a, X, core, hull)
        scatter_group(a, X, G, O, ids, 55)
        a.set_xlim(24.8, 31.8); a.set_ylim(8.6, 18.65)
        a.set_title(f"{name}：{len(ids)} 台 / 官方 {int(O[ids].sum())} 台", loc="left")
        annotate_points(a, X[ids, :2], [rows[i]["camera_id"] for i in ids], fontsize=9)
    legends(fig, group=False)
    finish(fig, out, "02_xy_main_labeled", "主区域 76 台相机 · 分高度俯视与逐台编号",
           f"按核心点 Z 的最大间隙分层：分界 Z={s['core_height_band_cut_m']:.3f} m；两台孤立点 1-30 / 1-55 见总图及全量图册。",
           footer + "\n分层仅帮助看清上下位置；不是删除某一高度或建立训练/测试划分。")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, left=.07, right=.96, top=.82, bottom=.13,
                         height_ratios=[1.55, 1, 1], hspace=.42, wspace=.17)
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    for official in [False, True]:
        ids = core[O[core] == official]
        ax.scatter(X[ids, 1], X[ids, 2], s=43, facecolors=COLORS[1] if official else "white", edgecolors=COLORS[1], zorder=3)
    ax.axhline(s["core_height_band_cut_m"], color=GRAY, ls="--", lw=.8)
    ax.set_xlabel("世界坐标 Y / m"); ax.set_ylabel("世界坐标 Z / m")
    ax.set_ylim(151.5, 152.8)
    ax.set_title(f"A  主区域 Y–Z：低层中位数 {s['core_lower_median_z_m']:.3f} m，高层 {s['core_upper_median_z_m']:.3f} m，相差 {s['core_upper_median_z_m']-s['core_lower_median_z_m']:.3f} m", loc="left")
    for g in range(1, 5):
        a = fig.add_subplot(gs[1 + (g - 1) // 2, (g - 1) % 2]); style(a)
        ids = np.flatnonzero(G == g)
        a.axhspan(X[core, 2].min(), X[core, 2].max(), color=COLORS[1], alpha=.07)
        for official in [False, True]:
            ii = ids[O[ids] == official]
            a.scatter([rows[j]["group_index"] for j in ii], X[ii, 2], s=27, marker=MARKERS[g],
                      facecolors=COLORS[g] if official else "white", edgecolors=COLORS[g], zorder=3)
        a.set_xlim(-1, 80); a.set_ylim(149.8, 153.8)
        a.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 78])
        a.set_ylabel("Z / m"); a.set_xlabel(f"xphase_{g} 组内编号")
        a.set_title(f"{g}  |  Z 范围 {X[ids, 2].min():.2f}–{X[ids, 2].max():.2f} m", loc="left", fontsize=11)
        if g == 1:
            a.annotate("1-55", (55, X[54, 2]), xytext=(50, 153.55), fontsize=9, arrowprops={"arrowstyle": "-", "color": GRAY})
        if g == 4:
            a.axvspan(66.5, 78.5, color="#EFC46B", alpha=.18)
            a.text(67, 153.5, "2024", fontsize=8)
    legends(fig)
    finish(fig, out, "03_height_distribution", "不能忽略的高度维度 · 世界坐标 Z",
           "上图放大主区域；下方四组共用相同 Z 轴范围。淡蓝带为连续主区域高度范围，黄色为跨日期拍摄。",
           "Z 是测量坐标系中的高程，不是相机离地高度；未提供地面高程，不能声称相机架设高度为 151–153 米。\n主区域约 0.80 米的上下层差异应进入后续覆盖设计；同 XY 的上下视角不自动等同于有充分插值支持。")

    fig, (ax, detail) = plt.subplots(1, 2, figsize=(17, 10), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.06, right=.97, top=.80, bottom=.13, wspace=.2)
    for a in [ax, detail]:
        style(a, True); hull_patch(a, X, core, hull)
        for category, color in CAT_COLORS.items():
            ids = np.flatnonzero(cat == category)
            for official in [False, True]:
                ii = ids[O[ids] == official]
                if category == "main_outlier":
                    a.scatter(X[ii, 0], X[ii, 1], s=48, marker="x", color=color, linewidths=1.3, zorder=4)
                else:
                    a.scatter(X[ii, 0], X[ii, 1], s=23 if category == "remote" else 40,
                              marker="o", facecolors=color if official else "white", edgecolors=color,
                              linewidths=1, alpha=.5 if category == "remote" else 1, zorder=3)
    ax.set_xlim(8, 37); ax.set_ylim(-.5, 20.5)
    detail.set_xlim(22.5, 33.0); detail.set_ylim(6.5, 20)
    ax.set_title("区域关系：孤立点不会扩张主区域", loc="left")
    detail.set_title("补充候选：优先橙色，紫色需复核", loc="left")
    ids = np.r_[near, review]
    # Merge almost coincident upper/lower XY only for this overview; exact IDs remain in tables.
    label_pos, label_text, label_color = merged_labels(ids, X, rows, threshold=.10, categories=cat)
    annotate_points(detail, label_pos, label_text, label_color, fontsize=7.4)
    for i in [29, 54]:
        ax.annotate(rows[i]["camera_id"], X[i, :2], xytext=(8, 8), textcoords="offset points", color=CAT_COLORS["main_outlier"], weight="bold")
    cat_names = {"main_core": "主区域", "near_supplement": "近邻补充 ≤2m", "transition_review": "扩边复核 2–3m", "main_outlier": "第一组孤立点", "remote": "较远 >3m"}
    handles = [Line2D([], [], marker="o", ls="none", color=color, label=f"{cat_names[c]} ({int((cat==c).sum())})") for c, color in CAT_COLORS.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .89), ncol=3, frameon=False, fontsize=10)
    finish(fig, out, "04_candidate_regions", "候选区域分析 · 到连续主区域的三维距离",
           "距离为最近主区域相机的 3D 欧氏距离；2m / 3m 是本次分析带宽，不是训练筛选规则或最终入选结论。实心有官方语义。",
           "近邻补充：xphase_2 共 20 台（官方 6）；扩边复核：xphase_2 共 13 台、xphase_3 共 2 台。xphase_4 最近仍为 3.57m。\n某些点靠近 1-30 / 1-55，但离连续主区域较远；不能用“到任意第一组相机很近”替代区域判断。")

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    fig.subplots_adjust(left=.07, right=.97, top=.80, bottom=.13, wspace=.17)
    eligible = core_mask | (cat == "near_supplement")
    for a, layer, name in zip(axes, [low, ~low], ["较低一层", "较高一层"]):
        style(a, True); hull_patch(a, X, core, hull)
        ids = np.flatnonzero(eligible & layer)
        scatter_group(a, X, G, O, ids, size=40)
        ii = np.flatnonzero(eligible & layer & interp)
        a.scatter(X[ii, 0], X[ii, 1], marker="o", s=160, facecolors="none", edgecolors="#41967F", linewidths=1.5, zorder=2)
        a.set_xlim(23.7, 32.5); a.set_ylim(7.0, 20.1)
        for y in [12.5, 15]:
            a.axhline(y, color=GRAY, ls="--", lw=.65)
        a.set_title(f"{name}：绿环 {len(ii)} 台有候选池插值支持", loc="left")
        annotate_points(a, X[ids, :2], [rows[i]["camera_id"] for i in ids], [COLORS[G[i]] for i in ids], fontsize=7.6)
    legends(fig, y=.89)
    fig.text(.06, .836, "绿环＝潜在插值测试复核候选，也可留作训练；其余点可作为空间覆盖候选。所有点均未划入固定训练/测试集。", fontsize=10, color="#2D7B68")
    finish(fig, out, "05_potential_train_test_support", "覆盖训练候选与潜在插值测试候选 · 尚未划分",
           "沿 Y 轴的三个覆盖段均应保留上下层训练支持；绿环来自其他核心相机的局部几何包围关系，不依赖有无语义标注。",
           "绿环定义：其他核心相机中，2m 内且 |ΔZ|≤0.35m 的邻居≥3，XY 最大方向缺口<180°。这是完整候选池支持，非所选18张的支持。\n选择训练集后须重新计算测试支持并检查 RGB 共享结构与遮挡；仅相机距离/凸包无法保证 M1/M2 匹配及重建质量。")

    fig, axes = plt.subplots(2, 2, figsize=(20, 17))
    fig.subplots_adjust(left=.055, right=.97, top=.85, bottom=.08, hspace=.22, wspace=.1)
    for g, a in enumerate(axes.flat, 1):
        style(a, True); hull_patch(a, X, core, hull)
        a.scatter(X[:, 0], X[:, 1], s=7, c="#D9DFE7", zorder=1)
        ids = np.flatnonzero(G == g)
        scatter_group(a, X, G, O, ids, size=42)
        a.set_xlim(8.2, 37.3); a.set_ylim(-.3, 20.5)
        a.set_title(f"xphase_{g}：78 台 / 官方 {int(O[ids].sum())} 台", loc="left")
        pos, labels, colors = merged_labels(ids, X, rows, threshold=.10, short=True)
        annotate_points(a, pos, labels, colors, fontsize=8.3)
    legends(fig, y=.90)
    finish(fig, out, "06_all_camera_ids_atlas", "全量编号图册 · 每组 78 台相机",
           "各图使用相同坐标范围；标签 01/02 表示该组两台 XY 距离≤0.10m 的相机，精确位置和高度分别保留在 CSV/JSON。",
           footer + "\n实心/空心逐个绘制，重叠点的官方状态请以 02 / 05 分层图及数据表为准。淡灰点为其他组位置。")


def merged_labels(ids, X, rows, threshold=.1, short=False, categories=None):
    remaining = set(int(i) for i in ids)
    positions, texts, colors = [], [], []
    while remaining:
        first = min(remaining)
        cluster = [i for i in sorted(remaining) if rows[i]["group"] == rows[first]["group"] and
                   np.linalg.norm(X[i, :2] - X[first, :2]) <= threshold and
                   (categories is None or categories[i] == categories[first])]
        remaining.difference_update(cluster)
        positions.append(X[cluster, :2].mean(axis=0))
        prefix = "" if short else f"{rows[first]['group']}-"
        texts.append(prefix + "/".join(f"{rows[i]['group_index']:02d}" for i in cluster))
        colors.append(COLORS[rows[first]["group"]] if categories is None else CAT_COLORS[categories[first]])
    return np.array(positions), texts, colors


def contact_sheet(out, rows, root):
    names = ["1-01", "1-02", "1-35", "1-37", "1-74", "1-78", "2-01", "2-03",
             "2-45", "2-52", "3-10", "4-50", "1-30", "1-55", "4-65", "4-67"]
    lookup = {r["camera_id"]: r for r in rows}
    fig, axes = plt.subplots(4, 4, figsize=(20, 13))
    fig.subplots_adjust(left=.035, right=.97, top=.87, bottom=.075, hspace=.34, wspace=.08)
    evidence = []
    for name, ax in zip(names, axes.flat):
        row = lookup[name]
        path = root / "datasets/iris_full_A_official_2k/images" / (row["image_id"] + ".png")
        if not path.exists():
            path = Path(row["rgb_source"])
        with Image.open(path) as im:
            im.thumbnail((640, 320))
            ax.imshow(np.asarray(im.convert("RGB")))
        official = "有官方" if row["has_official_semantics"] else "无官方"
        ax.set_title(f"{name}  |  {official}  |  Z={row['z_m']:.2f}m\n距核心 {row['distance_to_main_core_3d_m']:.2f}m  ·  {row['image_id']}", fontsize=8.5, loc="left", color=INK)
        ax.axis("off")
        evidence.append({"camera_id": name, "rgb_preview_source": str(path), "sha256": sha(path)})
    finish(fig, out, "07_rgb_context", "实际 RGB 参考 · 主区域、邻近补充与偏远点",
           "复用服务器已有 2K 全景 RGB，仅缩略展示，不生成语义预测；图像用于核对场景上下文，不能替代特征匹配。",
           "每张图标注相机 ID、官方语义状态、高程与到核心的 3D 距离。下排含两台第一组孤立相机、右侧其他组点和 2024 跨日期点。")
    save_json(out / "rgb_evidence.json", evidence)


def report(out, rows, s):
    group_table = "\n".join(f"| xphase_{g['group']} | {g['total']} | {g['official']} | {g['candidate_counts'].get('near_supplement', 0)} | {g['candidate_counts'].get('transition_review', 0)} |" for g in s["groups"])
    zone_table = "\n".join(f"| {z['name']} | {z['count']} | {z['official']} | {z['lower']} / {z['upper']} | {z['interpolation_official']} |" for z in s["coverage_zones"])
    official_near = [r for r in rows if r["candidate_category"] == "near_supplement" and r["has_official_semantics"]]
    near_table = "\n".join(f"| {r['camera_id']} | {r['image_id']} | {r['distance_to_main_core_3d_m']:.3f} | {r['nearest_main_core_id']} |" for r in official_near)
    text = f"""# IRIS 相机空间分布与官方语义覆盖分析

分析日期：2026-10-25。服务器项目：`/mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023`，读取时提交 `{s['source_project_commit']}`。

## 主要结论

建议以 **xphase_1 的连续 76 个相机位置**为约 18 张训练 + 6 张测试、无验证集实验的主要候选区域。其 X 范围 25.501–31.016 m、Y 范围 9.184–18.008 m，包围盒约 5.52×8.82 m；这是相机位置范围，不是场景墙体边界。该区域包含 **44 张官方标注**，不必为凑语义标注而扩展到相距较远的第三、四组。

本次读取 312 份原始 RMX 和原始官方标注 JSON，生成图表与候选说明。**没有启动 M1–M5、没有生成或修改语义标签、没有创建最终训练/测试/验证清单，也没有改动旧划分或模型。**

## 数据与官方语义核对

| 来源组 | 图像数 | 有官方语义 | 距核心≤2m补充 | 距核心2–3m复核 |
|---|---:|---:|---:|---:|
{group_table}

合计 131/312 张具有官方语义，均有 6 面记录且每面存在标注对象：786 面、24,549 个对象。官方状态直接按原始 COCO 的图像时间戳和 F/R/B/L/U/D 六面匹配，不读取 SAM 结果或审核后的自动补全标签。官方来源是 **CAD 投影标注**；有记录不等于每个像素、遮挡物或全部工业类别都完整准确。

第一组的 44 张包括旧训练集 27 张、旧验证集 6 张、旧测试集 11 张。因此此前“27 张官方标注”指旧 56 张训练子集，本次 44 张指第一组全部 78 张，两者并不矛盾。若后续重新划分，这些旧划分身份仅作追溯，必须按新协议重新初始化和训练，旧训练结果不能直接作为新测试评估结果。

官方来源：`{s['annotation_source']}`。

SHA256：`{s['annotation_source_sha256']}`。

## 真实位置、高度与主区域定义

RMX 明确写明 translation vector (millimeters)，统一除以 1000 得到米；跨组距离使用共同原始坐标，没有混用旧 OpenMVG 归一化位置或分别居中的组坐标。Z 是测量坐标高程，**没有地面高程，不能换算成离地高度**。

第一组的最近邻 3D 距离中位数为 {s['main_nearest_neighbor_3d_m_quantiles'][2]:.3f} m。用 1.5 m 邻接图识别连续主区域，最大连通分量为 76 台；阈值在 1.5、2、2.5、3 m 时均保持 76+1+1 的结构。1.0/1.25 m 会把主区域切成较小分量。这里的连通区域只用于说明空间结构，不是修改 SfM 或训练的筛选规则。

两个孤立点为：

- **1-30**：`2023-05-11_10.39.59`，距连续核心 3.185 m，无官方语义；对应旧训练编号 22。
- **1-55**：`2023-05-11_11.05.01`，距连续核心 9.800 m，无官方语义；对应旧训练编号 39。

保留它们的完整记录，但不让其扩张默认主区域。尤其部分第四组相机离 1-55 很近，却仍离连续主区域很远。CSV 同时保存“到全部第一组”和“到连续核心”的距离，避免这种误判。

主区域分为较低层 37 台（官方 20）和较高层 39 台（官方 24）。Z 中位数分别为 **151.728 m、152.530 m，相差 0.802 m**；两层 XY 经常几乎重合。图 02 / 05 分层标注相机 ID，完整高度见图 03。分层界限取核心 Z 最大间隙中点 {s['core_height_band_cut_m']:.3f} m，仅用于展示。

## 补充候选

第二组 20 张距核心不超过 2 m，另有 13 张处于 2–3 m。2 m 约为第一组典型相机间距的 2.9 倍，是本次便于解释的近邻范围；2–3 m 为扩边复核范围。完整连续距离均保留，后续应结合 RGB 重叠和实际几何质量判断，不把这些距离带变成固定拒绝规则。

优先查看下列 6 张有官方语义的第二组近邻。它们更适合补充覆盖边缘，尚不宜直接当作被核心训练视角包围的插值测试点。

| 相机编号 | 图像 ID | 距核心 3D / m | 最近核心相机 |
|---|---|---:|---|
{near_table}

其中 2-01 与 1-78 接近同一 XY、但高度不同，可补足小 Y 端的低层视角。2-02/03 和 2-06/07 延伸该端覆盖；2-52 位于大 Y 段的左侧边缘。2-45/46 距核心仅 0.30/0.28 m，空间上值得保留，虽然没有官方语义；是否选用仍以可见结构和采样冗余为依据。

第三组仅 **3-10/11** 进入 2–3 m 复核范围，距核心约 2.70/2.71 m，均无官方语义。其 49 张官方标注全部位于更远区域，当前不应仅为标注覆盖而纳入。第四组最近为 **4-50，3.571 m**；本次不优先加入。第四组 4-67 至 4-78 是 2024-01-23 拍摄，共 12 张且无官方标注，和其余 2023-05-11 图像有日期差异。

## 约 18+6 的空间可行性建议（未选择最终图像）

| 核心覆盖段 | 总候选 | 官方 | 低层 / 高层 | 有官方的潜在插值候选 |
|---|---:|---:|---:|---:|
{zone_table}

可以从“沿 Y 轴三段 × 两种高度”的覆盖结构开始规划 18 张训练预算，初步按每段约 6 张、同时覆盖两种高度，再根据设备遮挡和共享结构调整。这只是空间预算思路，未选定任何 18 张清单。图中上/下、左/右指坐标图方向，不推断地理南北。

训练优先照顾外围和转折覆盖，例如大 Y 端左侧 1-07/08 周边、右侧支路 1-52/53 周边、小 Y 端 1-76/77/78 与 2-01 周边；不要只选择中央官方标注密集的位置。位置合适而无官方标签的覆盖点可以保留待补标。

约 6 张测试应分布在同一核心区域内部，并有相应高度的实际训练邻居；可先在三段各考虑约 2 张的预算，再根据所选训练视角调整。本次有 **54 张核心视角**具备候选池局部插值支持，其中 **33 张有官方语义**，三个段分别有 8、14、11 张官方候选，留有选择余地。无需为了达到 6 张测试而把测试移到远区。

图 05 绿环的明确含义：目标之外的核心相机中，3D 距离≤2 m、|ΔZ|≤0.35 m 的邻居至少 3 台，且这些邻居在 XY 上的最大方向缺口<180°。这只证明**完整候选池中存在局部包围关系**，不证明未来所选 18 张能支持该测试点，也不证明遮挡、共享纹理或重建质量。所有绿环视角仍可留作训练；未标绿环也不构成算法拒绝。选定训练候选后必须重新检查这 6 张测试的实际邻居、共同可见设备、两层覆盖与重叠图像。

已抽查图 07 的 16 张服务器真实 RGB 缩略图：核心和第二组近邻均能看到管道、设备柜、支撑结构等工业内容；第二组小 Y 端仍有共同的大型管道/设备背景，同时视角变化带来遮挡差异。这支持将其作为补充复核候选，但本次没有跑特征匹配来证明几何可重建性。

## 图表与数据

- `01_xy_overview.png/pdf`：四组总体 XY 分布、官方状态与核心范围。
- `02_xy_main_labeled.png/pdf`：核心 76 台分高度逐台编号。
- `03_height_distribution.png/pdf`：主区域放大 Y–Z 与四组统一尺度 Z 分布。
- `04_candidate_regions.png/pdf`：主区域、近邻补充、扩边复核与远区。
- `05_potential_train_test_support.png/pdf`：覆盖候选与潜在插值测试候选，未固定划分。
- `06_all_camera_ids_atlas.png/pdf`：312 台相机全量编号图册。
- `07_rgb_context.png/pdf`：16 张实际 RGB 场景参考。
- `camera_inventory.csv/json`：312 行，图像 ID、来源组、XYZ、官方六面状态、源路径/SHA、两类 3D/XY 距离、最近核心编号、相对高度、候选说明、潜在支持列表及旧划分追溯。
- `analysis_summary.json`：统计、方法参数、来源与限制；`rgb_evidence.json`：所看 RGB 路径与 SHA256。

相机 ID 如 `2-01` 表示第二组按文件名时间排序的第 1 张，完整时间戳在 CSV/JSON。图册中 `01/02` 表示俯视几乎重合的多台相机；其官方状态及高度不能从重叠符号推断，需看分层图和明细。

复现（使用新的输出目录）：

```bash
cd /mnt/sda/ubuntu/lkx/od/OmniGS_SPCGS_20261023
.venv/bin/python tools/analyze_iris_camera_coverage.py \\
  --source-root /mnt/sda/ubuntu/lkx/od \\
  --output runs/iris_camera_analysis_NEW_RUN_ID
```
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    figures = [("02_xy_main_labeled", "主区域分高度编号图"), ("03_height_distribution", "高度分布"),
               ("04_candidate_regions", "补充候选与远区"), ("05_potential_train_test_support", "潜在训练/测试支持"),
               ("01_xy_overview", "四组总体分布"), ("06_all_camera_ids_atlas", "全量编号图册"), ("07_rgb_context", "实际 RGB 参考")]
    sections = "".join(f'<section><h2>{title}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy" alt="{title}"></a><p><a href="{name}.png">PNG 原图</a> · <a href="{name}.pdf">矢量 PDF</a></p></section>' for name, title in figures)
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS 相机与官方标注覆盖分析</title>
<style>body{{margin:0;background:#f3f6fa;color:#24394f;font:16px/1.7 "Microsoft YaHei",sans-serif}}main{{max-width:1350px;margin:auto;padding:30px}}h1{{font-size:30px}}a{{color:#1768a8}}section{{background:white;border-radius:12px;padding:20px;margin:25px 0}}img{{width:100%;height:auto}}.summary{{border-left:5px solid #226bba;padding:12px 22px;background:white}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.8 "Microsoft YaHei",sans-serif}}nav a{{display:inline-block;margin-right:22px}}</style>
<main><h1>IRIS 相机空间分布与官方语义覆盖</h1><p>2026-10-25 · 服务器真实 RMX / 官方 CAD 投影 COCO · 只做分析，未启动训练，未确定划分</p>
<div class="summary"><b>建议以第一组连续 76 个相机位置为核心，优先构建同一区域内的约 18+6 实验。</b><p>核心有官方标注 44 张；第二组有 20 张近邻补充，其中官方 6 张。上下高度层相差约 0.80m。测试优先考虑核心内部、且由最终训练视角实际支持的位置。</p></div>
<nav><a href="camera_inventory.csv">312 台 CSV</a><a href="camera_inventory.json">完整 JSON</a><a href="analysis_summary.json">统计 JSON</a><a href="REPORT.md">详细报告 Markdown</a></nav>
{sections}<section><h2>方法、统计与建议全文</h2><pre>{html.escape(text)}</pre></section></main></html>'''
    (out / "report.html").write_text(doc, encoding="utf-8")


def write_outputs(output, source_root, project):
    rows, summary, X, G, O, core, category, low, hull = inventory(source_root, project)
    save_json(output / "camera_inventory.json", {"metadata": summary, "cameras": rows})
    save_json(output / "analysis_summary.json", summary)
    fields = list(rows[0])
    with (output / "camera_inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    plots(output, rows, summary, X, G, O, core, category, low, hull)
    contact_sheet(output, rows, source_root)
    report(output, rows, summary)
    print(json.dumps({"output": str(output), "groups": summary["groups"],
                      "zones": summary["coverage_zones"], "core_count": len(core),
                      "interpolation_pool_count": summary["interpolation_pool_count"],
                      "interpolation_pool_official": summary["interpolation_pool_official"]}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=False)
    write_outputs(a.output, a.source_root, project)


if __name__ == "__main__":
    main()
