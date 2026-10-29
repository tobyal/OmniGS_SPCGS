"""ROI3 F05, revision 4: aligned insets and plain colored dashed connectors.

The user's small rectangles select content; the larger rectangles specify
approximate inset destinations. Source model pixels are never modified.
Run: python draw_f05_insets.py --dpi 600
First run additionally requires --source-report and --reference-image.
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
WIDTH, MARGIN, GAP = 180.0, 1.5, 2.0
COL = (WIDTH - 2 * MARGIN - 2 * GAP) / 3
MAIN_H, HEADER_H, PANEL_GAP, LEGEND_H = COL * 360 / 640, 9.0, 4.5, 7.0
RED, BLUE, INK = HexColor('#E02424'), HexColor('#4875B8'), HexColor('#171717')
FRAME_WIDTH_PT = 1.05
CONNECTOR_WIDTH_PT = 1.1
CONNECTOR_DASH_PT = [3.0, 2.0]
INSET_TOP, INSET_HEIGHT, INSET_RIGHT, INSET_GAP = 200.0, 150.0, 628.0, 24.0
PAIR_COLORS = {'red': RED, 'blue': BLUE}
PANELS = {
    'a': {'title': '(a) Geometry-Augmented Layout Reconstruction',
          'columns': ['Ground Truth', 'L0 (P0 Layout)', 'L1 (P1 Layout)'],
          'files': ['ROI3_GT_RGB.png', 'ROI3_L0_RGB.png', 'ROI3_L1_RGB.png']},
    'b': {'title': '(b) Sparse Semantic Anchoring',
          'columns': ['Semantic Ground Truth', 'L1 (w/o Anchor)', 'L2 (w/ Anchor)'],
          'files': ['ROI3_GT_Semantic.png', 'ROI3_L1_Semantic.png', 'ROI3_L2_Semantic.png']},
}
# All coordinates refer to the unchanged 640 x 360 ROI3 source.
# In the user's reference, the SMALL red box around the pipe label feeds the
# LARGE red box at lower left. The SMALL blue box at upper right feeds the
# LARGE blue box at lower right. Large boxes are destinations, NOT sources.
# Native pixel edges come from the supplied screenshot and preserve aspect.
INSETS = {
    'a': [
        {'id': 1, 'pair_color': 'red', 'source_xyxy': [368, 92, 457, 169],
         'destination_slot': 'left', 'connector_source': [368, 169],
         'connector_destination': 'top_right'},
        {'id': 2, 'pair_color': 'blue', 'source_xyxy': [457, 12, 591, 79],
         'destination_slot': 'right', 'connector_source': [591, 79],
         'connector_destination': 'top_right'},
    ],
    'b': [
        {'id': 1, 'pair_color': 'red', 'source_xyxy': [143, 139, 258, 239],
         'destination_slot': 'right', 'connector_source': [258, 189],
         'connector_destination': 'top_left'},
    ],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(args):
    INPUT.mkdir(exist_ok=True)
    manifest = INPUT / 'input_manifest.json'
    if not manifest.exists():
        assert args.source_report and args.reference_image
        source = Path(args.source_report).resolve()
        assert source != ROOT and not source.is_relative_to(ROOT)
        rows = []
        for panel in PANELS.values():
            for name in panel['files']:
                p = source / 'figures/panels' / name
                shutil.copy2(p, INPUT / name)
                assert Image.open(INPUT / name).size == (640, 360)
                rows.append({'file': name, 'source': str(p), 'sha256': sha(p)})
        shutil.copy2(args.reference_image, INPUT / 'user_inset_placement.png')
        rows.append({'file': 'user_inset_placement.png', 'source': str(args.reference_image.resolve()), 'sha256': sha(args.reference_image)})
        m = json.loads((source / 'data/dataset_manifest.json').read_text(encoding='utf-8'))
        classes = [{k: c[k] for k in ['id', 'name', 'color']} for c in m['classes']['classes']]
        write(INPUT / 'official_class_mapping.json', classes)
        rows.append({'file': 'official_class_mapping.json', 'sha256': sha(INPUT / 'official_class_mapping.json')})
        write(manifest, {'source_report': str(source), 'files': rows})
    data = json.loads(manifest.read_text(encoding='utf-8'))
    for row in data['files']:
        assert sha(INPUT / row['file']) == row['sha256']
    return data


def register_fonts():
    choices = [(Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')),
               (Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'), Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'))]
    regular, bold = next((a, b) for a, b in choices if a.exists() and b.exists())
    pdfmetrics.registerFont(TTFont('Paper', str(regular)))
    pdfmetrics.registerFont(TTFont('PaperBold', str(bold)))
    return [str(regular), str(bold)]


def text(c, height, x, top, value, size=7.6, bold=False, color=INK, center=False):
    c.setFillColor(color)
    c.setFont('PaperBold' if bold else 'Paper', size)
    f = c.drawCentredString if center else c.drawString
    f(x * mm, (height - top) * mm, value)


def image(c, height, im, x, top, width, h):
    c.drawImage(ImageReader(im), x * mm, (height - top - h) * mm,
                width * mm, h * mm, mask=None, preserveAspectRatio=True)


def rect(c, height, x, top, w, h, color, line=FRAME_WIDTH_PT):
    c.setStrokeColor(color);c.setLineWidth(line)
    c.rect(x * mm, (height - top - h) * mm, w * mm, h * mm, stroke=1, fill=0)


def badge(c, height, x, top, label, color):
    c.setFillColor(white)
    c.rect(x * mm, (height - top - 2.4) * mm, 2.4 * mm, 2.4 * mm, stroke=0, fill=1)
    text(c, height, x + 1.2, top + 1.85, str(label), 6.0, True, color, True)


def destination(item):
    x0, y0, x1, y1 = item['source_xyxy']
    # Shared top, bottom and height across both panels. Preserve each crop's
    # aspect ratio. The RGB pair has a fixed gap; right insets share a margin.
    h = INSET_HEIGHT
    w = h * (x1 - x0) / (y1 - y0)
    y = INSET_TOP
    x = INSET_RIGHT - w
    if item['destination_slot'] == 'left':
        blue = INSETS['a'][1]['source_xyxy']
        blue_width = h * (blue[2] - blue[0]) / (blue[3] - blue[1])
        x -= blue_width + INSET_GAP
    assert x >= 0 and y >= 0 and x + w <= 640 and y + h <= 360
    return x, y, w, h


def panel(c, page_h, key, top, crops):
    info = PANELS[key]
    text(c, page_h, MARGIN, top + 3.1, info['title'], 9.0, True)
    main_top = top + HEADER_H
    scale = COL / 640
    for j, (title, filename) in enumerate(zip(info['columns'], info['files'])):
        left = MARGIN + j * (COL + GAP)
        text(c, page_h, left + COL / 2, top + 7.1, title, center=True)
        image(c, page_h, Image.open(INPUT / filename).convert('RGB'), left, main_top, COL, MAIN_H)
        # One colored dashed stroke only. No white edge or under-stroke.
        for item in INSETS[key]:
            dx, dy, dw, dh = destination(item)
            corner = item['connector_destination']
            tx = dx + dw if corner.endswith('right') else dx
            ty = dy + dh if corner.startswith('bottom') else dy
            sx, sy = item['connector_source']
            color = PAIR_COLORS[item['pair_color']]
            c.saveState()
            c.setStrokeColor(color);c.setLineWidth(CONNECTOR_WIDTH_PT)
            c.setDash(CONNECTOR_DASH_PT, 0)
            c.setLineCap(0)
            c.line((left + sx * scale) * mm, (page_h - main_top - sy * scale) * mm,
                   (left + tx * scale) * mm, (page_h - main_top - ty * scale) * mm)
            c.restoreState()
        # Crop from the untouched inputs, then place inside the main image.
        for item in INSETS[key]:
            dx, dy, dw, dh = destination(item)
            px, py = left + dx * scale, main_top + dy * scale
            crop = Image.open(crops[(key, item['id'], filename)]).convert('RGB')
            image(c, page_h, crop, px, py, dw * scale, dh * scale)
            color = PAIR_COLORS[item['pair_color']]
            rect(c, page_h, px, py, dw * scale, dh * scale, color)
            badge(c, page_h, px + .3, py + .3, item['id'], color)
        for item in INSETS[key]:
            x0, y0, x1, y1 = item['source_xyxy']
            px, py = left + x0 * scale, main_top + y0 * scale
            color = PAIR_COLORS[item['pair_color']]
            rect(c, page_h, px, py, (x1-x0)*scale, (y1-y0)*scale, color)
            badge(c, page_h, px - 2.7, py, item['id'], color)


def legend(c, page_h, top, classes):
    labels = ['HVAC' if a['name'] == 'hvac' else a['name'].replace('_', ' ').capitalize() for a in classes]
    swatch, gap, size = 2.4, 2.8, 6.8
    widths = [swatch + 1 + pdfmetrics.stringWidth(n, 'Paper', size)/mm for n in labels]
    total = sum(widths) + (len(labels)-1)*gap
    assert total < WIDTH - 2*MARGIN
    x = (WIDTH-total)/2
    for row, label, w in zip(classes, labels, widths):
        c.setFillColor(Color(*[v/255 for v in row['color']]))
        c.rect(x*mm,(page_h-top-2.3)*mm,swatch*mm,2.3*mm,stroke=0,fill=1)
        text(c,page_h,x+swatch+1,top+1.95,label,size)
        x += w+gap


def build(args):
    frozen = prepare(args)
    font_paths = register_fonts()
    classes = json.loads((INPUT/'official_class_mapping.json').read_text(encoding='utf-8'))
    crop_dir = ROOT/'crops';crop_dir.mkdir(exist_ok=True)
    crops, records = {}, []
    for key, info in PANELS.items():
        for item in INSETS[key]:
            for filename in info['files']:
                crop = Image.open(INPUT/filename).convert('RGB').crop(item['source_xyxy'])
                p = crop_dir/f"{key}_detail{item['id']}_{filename}"
                crop.save(p)
                assert np.array_equal(np.array(crop),np.array(Image.open(p)))
                crops[(key,item['id'],filename)] = p
                records.append({'file':str(p.relative_to(ROOT)),'size':list(crop.size),'sha256':sha(p),'exact_source_crop':True})
    outputs=[]
    for name, keys in [('F05_Combined',['a','b']),('F05a_Geometry',['a']),('F05b_Semantic',['b'])]:
        height = 2*MARGIN + len(keys)*(HEADER_H+MAIN_H) + (len(keys)-1)*PANEL_GAP + (LEGEND_H if 'b' in keys else 0)
        pdf=ROOT/(name+'.pdf')
        c=canvas.Canvas(str(pdf),pagesize=(WIDTH*mm,height*mm),pageCompression=1,invariant=1)
        c.setTitle(name+' - ROI3 / View 1-60 - aligned dashed insets v4')
        c.setSubject('Two RGB source regions and one semantic source region; unchanged model pixels')
        c.setFillColor(white);c.rect(0,0,WIDTH*mm,height*mm,stroke=0,fill=1)
        y=MARGIN
        for key in keys:
            panel(c,height,key,y,crops);y+=HEADER_H+MAIN_H
            if key!=keys[-1]:y+=PANEL_GAP
        if 'b' in keys:legend(c,height,y+2.1,classes)
        c.showPage();c.save()
        subprocess.run([args.pdftoppm,'-singlefile','-r',str(args.dpi),'-png',str(pdf),str(ROOT/name)],check=True)
        im=Image.open(ROOT/(name+'.png'))
        outputs.append({'name':name,'width_mm':WIDTH,'height_mm':height,'PNG_pixels':list(im.size),'dpi':args.dpi,'PDF':pdf.name,'PNG':name+'.png'})
    geometry={}
    for key,items in INSETS.items():
        geometry[key]=[]
        for item in items:
            dx,dy,dw,dh=destination(item);s=item['source_xyxy']
            geometry[key].append({**item,'destination_xywh':[dx,dy,dw,dh],
                                 'magnification':dw/(s[2]-s[0]),
                                 'full_ERP_source_xyxy':[s[0]+520,s[1]+160,s[2]+520,s[3]+160]})
    write(ROOT/'roi_and_inset_coordinates.json',{'view':'1-60','ERP_ROI3_xywh':[520,160,640,360],
          'source_regions_per_panel':{'a':2,'b':1},'definitions':{'red':'small red source to large red inset','blue':'small blue source to large blue inset'},
          'regions':geometry,'frame_width_pt':FRAME_WIDTH_PT,
          'alignment':{'top':INSET_TOP,'height':INSET_HEIGHT,'bottom':INSET_TOP+INSET_HEIGHT,'right':INSET_RIGHT,'RGB_pair_gap':INSET_GAP},
          'adjustments':'Source regions unchanged from v3. RGB red source to lower-left red inset; blue source to lower-right blue inset. All insets share top, bottom and height; right insets share the same right margin. Colored 1.1 pt dashed connectors have no white outline or under-stroke.'})
    write(ROOT/'generation_manifest.json',{'status':'generated','version':4,'outputs':outputs,'font_paths':font_paths,
          'frame_width_pt':FRAME_WIDTH_PT,'same_color_source_destination':True,
          'connectors':{'width_pt':CONNECTOR_WIDTH_PT,'dash_pt':CONNECTOR_DASH_PT,'white_outline':False,'white_understroke':False},
          'source_inputs':frozen,'crop_checks':records,'color_mapping':classes,
          'preservation':{'filters_applied':False,'semantic_labels_or_colors_changed':False,'lossless_PDF_images':True,
                          'original_files_unchanged':True,'insets_are_overlays_not_new_predictions':True}})
    (ROOT/'figure_caption.txt').write_text('Figure F05. Qualitative comparison at ROI3, view 1-60. (a) Ground-truth RGB, L0 (P0 layout), and L1 (P1 layout). The small red region around the pipe label is enlarged in the lower-left red inset; the small blue region at upper right is enlarged in the lower-right blue inset. (b) Semantic ground truth, L1 without sparse semantic anchors, and L2 with anchors. The small red source region is enlarged in the lower-right red inset. Matching colors, numbers and dashed lines indicate correspondence. Insets share an aligned top and bottom; aspect ratios are preserved. Coordinates and magnification are identical across corresponding model columns; all model pixels and semantic colors are unchanged.\n',encoding='utf-8')
    (ROOT/'REPRODUCE.txt').write_text('python draw_f05_insets.py --dpi 600\nRequires Pillow, numpy, reportlab, Poppler (pdftoppm), and Arial or Liberation Sans.\ninputs/ contains frozen original ROI3 outputs, the official class mapping and the corrected user reference.\nOnly two RGB source regions and one semantic source region are used. Enlargements are embedded inside the main images.\nNo training, inference, sharpening, smoothing or semantic editing is performed.\n',encoding='utf-8')
    print(json.dumps({'outputs':outputs,'regions':geometry},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-report',type=Path);p.add_argument('--reference-image',type=Path)
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--pdftoppm',default=shutil.which('pdftoppm') or 'pdftoppm')
    build(p.parse_args())
