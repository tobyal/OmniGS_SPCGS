"""Reproduce the user's ROI3-only F05 layout without editing model pixels.

Run with Python + Pillow + numpy + reportlab + Poppler. On first execution,
pass --source-report and --screenshot; subsequent executions use frozen inputs.
All output paths stay beside this script. Original F05 files are read-only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
RED = HexColor('#E02424')
INK = HexColor('#171717')
WIDTH = 180.0
MARGIN = 1.5
GAP = 2.0
COL = (WIDTH - 2 * MARGIN - 2 * GAP) / 3
MAIN_H = COL * 360 / 640
ZOOM_H = 22.0
HEADER_H = 9.0
ZOOM_LABEL_H = 2.8
ROW_GAP = 1.0
MAIN_ZOOM_GAP = 1.5
PANEL_GAP = 4.5
LEGEND_H = 7.0
DETAIL_ROWS = {'a': [[1], [2, 3], [4]], 'b': [[1, 2]]}
DETAIL_GAP = 1.4

# Exact detected red-stroke extents in the user's latest 2000 x 954 screenshot.
# The screenshot scales the same 640 x 360 source slightly differently in a/b.
# Mapping uses the centerline of the 4-pixel-wide red stroke, rounded once to
# native pixel edges. No ROI is selected from model performance.
SCREEN = {
    'a': {'image_xyxy': [670, 62, 1320, 427],
          'boxes_xyxy': [[1132, 72, 1272, 144], [1042, 153, 1136, 235],
                         [836, 281, 998, 422], [1031, 284, 1307, 423]]},
    'b': {'image_xyxy': [671, 562, 1324, 929],
          'boxes_xyxy': [[815, 702, 936, 808], [1110, 763, 1294, 923]]},
}
PANELS = {
    'a': {'title': '(a) Geometry-Augmented Layout Reconstruction',
          'columns': ['Ground Truth', 'L0 (P0 Layout)', 'L1 (P1 Layout)'],
          'files': ['ROI3_GT_RGB.png', 'ROI3_L0_RGB.png', 'ROI3_L1_RGB.png']},
    'b': {'title': '(b) Sparse Semantic Anchoring',
          'columns': ['Semantic Ground Truth', 'L1 (w/o Anchor)', 'L2 (w/ Anchor)'],
          'files': ['ROI3_GT_Semantic.png', 'ROI3_L1_Semantic.png', 'ROI3_L2_Semantic.png']},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(source, screenshot):
    INPUT.mkdir(exist_ok=True)
    if not (INPUT / 'input_manifest.json').exists():
        assert source and screenshot, 'First run requires --source-report and --screenshot.'
        source = Path(source).resolve()
        assert source != ROOT and not source.is_relative_to(ROOT)
        records = []
        for p in [source / 'figures/panels' / name for panel in PANELS.values() for name in panel['files']]:
            q = INPUT / p.name
            shutil.copy2(p, q)
            records.append({'source': str(p), 'file': q.name, 'sha256': sha(p)})
            assert Image.open(q).size == (640, 360)
        shutil.copy2(screenshot, INPUT / 'user_latest_red_boxes.png')
        records.append({'source': str(Path(screenshot).resolve()), 'file': 'user_latest_red_boxes.png', 'sha256': sha(screenshot)})
        m = json.loads((source / 'data/dataset_manifest.json').read_text(encoding='utf-8'))
        classes = [{k: c[k] for k in ['id', 'name', 'color']} for c in m['classes']['classes']]
        save_json(INPUT / 'official_class_mapping.json', classes)
        records.append({'source': str(source / 'data/dataset_manifest.json'), 'file': 'official_class_mapping.json', 'sha256': sha(INPUT / 'official_class_mapping.json')})
        save_json(INPUT / 'input_manifest.json', {'files': records, 'source_report': str(source)})
    m = json.loads((INPUT / 'input_manifest.json').read_text(encoding='utf-8'))
    for r in m['files']:
        assert sha(INPUT / r['file']) == r['sha256'], r['file']
    return m


def mapped_boxes():
    result = {}
    for key, s in SCREEN.items():
        ox, oy, ex, ey = s['image_xyxy']
        scale = np.array([640 / (ex - ox), 360 / (ey - oy)] * 2)
        origin = np.array([ox, oy, ox, oy])
        rows = []
        for i, outer in enumerate(s['boxes_xyxy'], 1):
            center = np.array(outer) + [2, 2, -2, -2]
            box = np.rint((center - origin) * scale).astype(int).tolist()
            assert 0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 360
            rows.append({'detail': i, 'screenshot_red_outer_xyxy': outer,
                         'screenshot_red_centerline_xyxy': center.tolist(),
                         'source_ROI3_xyxy': box,
                         'source_full_ERP_xyxy': (np.array(box) + [520, 160, 520, 160]).tolist()})
        result[key] = rows
    return result


def fonts():
    candidates = [(Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')),
                  (Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'), Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'))]
    regular, bold = next((a, b) for a, b in candidates if a.exists() and b.exists())
    pdfmetrics.registerFont(TTFont('Paper', str(regular)))
    pdfmetrics.registerFont(TTFont('PaperBold', str(bold)))
    return {'regular': str(regular), 'bold': str(bold)}


def panel_height(key, boxes):
    n = len(DETAIL_ROWS[key])
    return HEADER_H + MAIN_H + MAIN_ZOOM_GAP + n * (ZOOM_LABEL_H + ZOOM_H) + (n - 1) * ROW_GAP


def text(c, h, x, top, value, size=7.4, bold=False, color=INK, align='left'):
    c.setFillColor(color)
    c.setFont('PaperBold' if bold else 'Paper', size)
    fn = c.drawCentredString if align == 'center' else c.drawString
    fn(x * mm, (h - top) * mm, value)


def rectangle(c, h, x, top, w, ht, color, linewidth=.55, fill=None):
    c.setStrokeColor(color)
    c.setLineWidth(linewidth)
    if fill is not None:
        c.setFillColor(fill)
    c.rect(x * mm, (h - top - ht) * mm, w * mm, ht * mm, stroke=1, fill=int(fill is not None))


def picture(c, h, im, x, top, w, ht):
    # Lossless embedded pixels, no filtering, enhancement, or altered labels.
    c.drawImage(ImageReader(im), x * mm, (h - top - ht) * mm,
                w * mm, ht * mm, mask=None, preserveAspectRatio=True)


def panel(c, h, key, top, boxes, crops):
    info = PANELS[key]
    text(c, h, MARGIN, top + 3.1, info['title'], 9.0, True)
    main_top = top + HEADER_H
    for col, (name, filename) in enumerate(zip(info['columns'], info['files'])):
        x = MARGIN + col * (COL + GAP)
        text(c, h, x + COL / 2, top + 7.1, name, 7.6, align='center')
        im = Image.open(INPUT / filename).convert('RGB')
        picture(c, h, im, x, main_top, COL, MAIN_H)
        for item in boxes[key]:
            x0, y0, x1, y1 = item['source_ROI3_xyxy']
            rx, ry = x + x0 * COL / 640, main_top + y0 * MAIN_H / 360
            rectangle(c, h, rx, ry, (x1 - x0) * COL / 640, (y1 - y0) * MAIN_H / 360, RED)
            # Small index outside the selected content; no long arrows.
            c.setFillColor(white)
            c.rect((rx - 2.6) * mm, (h - ry - 2.6) * mm, 2.4 * mm, 2.6 * mm, stroke=0, fill=1)
            text(c, h, rx - 1.4, ry + 2.0, str(item['detail']), 6.3, True, RED, 'center')
    y = main_top + MAIN_H + MAIN_ZOOM_GAP
    for details in DETAIL_ROWS[key]:
        slot_w = (COL - DETAIL_GAP * (len(details) - 1)) / len(details)
        for col, filename in enumerate(info['files']):
            for local_col, detail in enumerate(details):
                x = MARGIN + col * (COL + GAP) + local_col * (slot_w + DETAIL_GAP)
                text(c, h, x, y + 2.0, f'Detail {detail}', 6.3, color=RED)
                crop = Image.open(crops[(key, detail, filename)]).convert('RGB')
                iw, ih = crop.size
                factor = min((slot_w - .6) / iw, (ZOOM_H - .6) / ih)
                dw, dh = iw * factor, ih * factor
                px, py = x + (slot_w - dw) / 2, y + ZOOM_LABEL_H + (ZOOM_H - dh) / 2
                # Pair the similarly shaped narrow crops inside each model
                # column. Same height and magnification as the taller draft,
                # but remove two unnecessary rows and their empty padding.
                rectangle(c, h, x, y + ZOOM_LABEL_H, slot_w, ZOOM_H, HexColor('#D8D8D8'), .25)
                picture(c, h, crop, px, py, dw, dh)
        y += ZOOM_LABEL_H + ZOOM_H + ROW_GAP


def legend(c, h, top, classes):
    names = [('HVAC' if row['name'] == 'hvac' else row['name'].replace('_', ' ').capitalize()) for row in classes]
    size = 6.8
    swatch = 2.4
    widths = [swatch + 1.0 + pdfmetrics.stringWidth(n, 'Paper', size) / mm for n in names]
    gap = 2.8
    total = sum(widths) + gap * (len(names) - 1)
    assert total <= WIDTH - 2 * MARGIN
    x = (WIDTH - total) / 2
    for row, name, w in zip(classes, names, widths):
        color = Color(*[v / 255 for v in row['color']])
        c.setFillColor(color)
        c.rect(x * mm, (h - top - 2.3) * mm, swatch * mm, 2.3 * mm, stroke=0, fill=1)
        text(c, h, x + swatch + 1.0, top + 1.95, name, size)
        x += w + gap


def build(args):
    frozen = prepare(args.source_report, args.screenshot)
    font_info = fonts()
    boxes = mapped_boxes()
    classes = json.loads((INPUT / 'official_class_mapping.json').read_text(encoding='utf-8'))
    crop_dir = ROOT / 'crops'
    crop_dir.mkdir(exist_ok=True)
    crops = {}
    crop_checks = []
    for key, info in PANELS.items():
        for item in boxes[key]:
            for filename in info['files']:
                original = Image.open(INPUT / filename).convert('RGB')
                cropped = original.crop(item['source_ROI3_xyxy'])
                p = crop_dir / f"{key}_detail{item['detail']}_{filename}"
                cropped.save(p)
                assert np.array_equal(np.array(Image.open(p)), np.array(cropped))
                crops[(key, item['detail'], filename)] = p
                crop_checks.append({'file':str(p.relative_to(ROOT)), 'sha256':sha(p), 'size':list(cropped.size), 'exact_source_crop':True})
    ha, hb = panel_height('a', boxes), panel_height('b', boxes)
    outputs = []
    for stem, keys in [('F05_Combined', ['a', 'b']), ('F05a_Geometry', ['a']), ('F05b_Semantic', ['b'])]:
        height = 2 * MARGIN + sum(panel_height(k, boxes) for k in keys) + PANEL_GAP * (len(keys) - 1) + (LEGEND_H if 'b' in keys else 0)
        p = ROOT / (stem + '.pdf')
        c = canvas.Canvas(str(p), pagesize=(WIDTH * mm, height * mm), pageCompression=1, invariant=1)
        c.setTitle(stem + ' - ROI3 / View 1-60')
        c.setAuthor('OmniGS_SPCGS experiment visualization')
        c.setSubject('Original model outputs; user-selected detail crops; no image enhancement')
        c.setFillColor(white);c.rect(0,0,WIDTH*mm,height*mm,stroke=0,fill=1)
        y = MARGIN
        for key in keys:
            panel(c, height, key, y, boxes, crops)
            y += panel_height(key, boxes)
            if key != keys[-1]:y += PANEL_GAP
        if 'b' in keys:legend(c, height, y + 2.1, classes)
        c.showPage();c.save()
        # Rasterize this exact vector layout. /Interpolate is false by default;
        # semantic label colors stay categorical and no RGB filter is applied.
        subprocess.run([args.pdftoppm, '-singlefile', '-r', str(args.dpi), '-png', str(p), str(ROOT/stem)], check=True)
        im = Image.open(ROOT/(stem+'.png'))
        outputs.append({'name':stem,'width_mm':WIDTH,'height_mm':height,'PNG_pixels':list(im.size),'dpi':args.dpi,'PDF':p.name,'PNG':stem+'.png'})
    save_json(ROOT/'roi_coordinates.json', {'camera_id':'1-60','view':'2023-05-11_11.08.41','source_full_ERP_ROI3_xywh':[520,160,640,360], 'screenshot_dimensions':[2000,954], 'screenshot_panels':SCREEN,'details':boxes,'mapping':'Four-pixel red stroke centerline mapped to original 640x360 source and rounded once to native pixel edges. No new detail regions selected.'})
    save_json(ROOT/'generation_manifest.json', {'status':'generated','source_inputs':frozen,'outputs':outputs,'fonts':font_info,'crop_checks':crop_checks,'color_mapping':classes,'preservation':{'original_files_unchanged':True,'generated_predictions_changed':False,'RGB_filters_applied':False,'semantic_colors_changed':False,'source_RGB_and_semantic_images_losslessly_embedded':True},'layout':{'width_mm':WIDTH,'main_image_mm':[COL,MAIN_H],'detail_slot_height_mm':ZOOM_H,'detail_rows':DETAIL_ROWS,'red_line_width_pt':.55,'same_crop_and_scale_across_models':True,'aspect_ratio_preserved_with_white_padding':True}})
    caption = ('Figure F05. Qualitative comparison at ROI3, view 1-60 (crossing pipes, cable trays and frames). '
               '(a) Ground-truth RGB, L0 initialized from P0, and L1 initialized from P1 at the end of layout pre-optimization. '
               '(b) Semantic ground truth, L1 without sparse semantic anchors, and L2 with sparse semantic anchors. '
               'Numbered red rectangles identify the user-selected regions; corresponding magnified crops appear below each panel. '
               'Crop coordinates and display scales are identical across the three columns for each numbered region. '
               'All predictions and class colors are unchanged.\n')
    (ROOT/'figure_caption.txt').write_text(caption,encoding='utf-8')
    (ROOT/'REPRODUCE.txt').write_text('Run: python draw_f05.py --dpi 600\nRequires: Pillow, numpy, reportlab, and pdftoppm on PATH.\nThe inputs/ directory freezes the six original ROI3 images, official class colors, and user screenshot.\nNo model training, inference, RGB enhancement, semantic relabeling, or ROI reselection is performed.\nSource crops retain their aspect ratios inside uniform display slots.\nOutputs: three lossless PNGs at 600 DPI and three PDFs with embedded fonts and lossless image objects.\n',encoding='utf-8')
    for record in frozen['files']:
        assert sha(INPUT/record['file'])==record['sha256']
    print(json.dumps({'outputs':outputs,'native_boxes':{k:[x['source_ROI3_xyxy'] for x in v] for k,v in boxes.items()}},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-report',type=Path)
    p.add_argument('--screenshot',type=Path)
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--pdftoppm',default=shutil.which('pdftoppm') or 'pdftoppm')
    build(p.parse_args())
