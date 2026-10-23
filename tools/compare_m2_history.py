"""Compare saved historical/current SGI renders with separate class legends.

Training budgets and vocabularies differ. These figures are historical context,
not a controlled estimate of the effect of changing M2.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from spcgs.common import ROOT, read, ref, Stage
from spcgs.visuals import PALETTE


def compare(old_reference, new_stage, output):
    old_reference, new_stage, output = map(Path, (old_reference, new_stage, output))
    old = read(old_reference / 'render_job/request.json')
    new = read(new_stage / 'validation/final_job/request.json')
    old_cfg = read(old_reference / 'inputs.json')['source_config']
    new_cfg = read(new_stage / 'config.json')
    assert read(old_reference / 'outputs.json')['status'] == 'completed'
    assert read(new_stage / 'outputs.json')['status'] == 'completed'
    old_views = {v['name']: v for v in old['views']}
    w, h = 768, 384
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()
    note = 'Historical comparison: different training budgets and vocabularies; not an isolated M2 effect.'
    with Stage(output, {'old_render_request': ref(old_reference / 'render_job/request.json'),
                        'new_render_request': ref(new_stage / 'validation/final_job/request.json')},
               {'old_training': old_cfg, 'new_training': new_cfg}) as stage:
        figures = []
        for view in new['views']:
            name = view['name']
            if name not in old_views:
                continue
            assert view['image'] == old_views[name]['image']
            rgb = Image.open(view['image']).convert('RGB').resize((w, h))

            def overlay(labels):
                colors = Image.fromarray(PALETTE[labels]).resize((w, h), Image.Resampling.NEAREST)
                return Image.blend(rgb, colors, .65)

            def labels(request):
                logits = np.fromfile(Path(request['output']) / (name + '_logits.f16'), dtype='f2')
                return logits.reshape(len(request['classes']), request['semantic_height'], request['semantic_width']).argmax(0)

            s = np.fromfile(view['relevancy'], dtype='f2').reshape(len(new['classes']), view['relevancy_height'], view['relevancy_width'])
            panels = [('Real training RGB', rgb),
                      (f"Old SGI RGB: {old_cfg['sgi_iterations']} steps, trained at {old_cfg['width']}x{old_cfg['height']}", Image.open(Path(old['output']) / (name + '_rgb.png')).resize((w, h))),
                      (f"New SGI RGB: {new_cfg['sgi_iterations']} steps, trained at {new_cfg['width']}x{new_cfg['height']}", Image.open(Path(new['output']) / (name + '_rgb.png')).resize((w, h))),
                      (f"New cached S teacher ({len(new['classes'])} classes)", overlay(s.argmax(0))),
                      (f"Old Gaussian class head ({len(old['classes'])} classes)", overlay(labels(old))),
                      (f"New Gaussian class head ({len(new['classes'])} classes)", overlay(labels(new)))]
            canvas = Image.new('RGB', (3 * w, 1130), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), note, fill='black', font=font)
            for i, (title, panel) in enumerate(panels):
                x, y = (i % 3) * w, 40 + (i // 3) * (h + 30)
                draw.text((x + 8, y + 4), title, fill='black', font=font)
                canvas.paste(panel, (x, y + 28))
            for classes, start_x, columns, column_w, title in [
                (old['classes'], 10, 2, 370, 'OLD class legend'),
                (new['classes'], w + 10, 3, 505, 'NEW / TEACHER class legend')]:
                draw.text((start_x, 875), title, fill='black', font=font)
                for i, category in enumerate(classes):
                    x, y = start_x + (i % columns) * column_w, 905 + (i // columns) * 30
                    draw.rectangle((x, y, x + 18, y + 18), fill=tuple(PALETTE[i]))
                    draw.text((x + 26, y), category, fill='black', font=font)
            path = output / (name + '_history.jpg')
            canvas.save(path, quality=94)
            figures.append(str(path))
        assert figures, 'No common saved views'
        stage.done({'figures': figures, 'note': note})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('old_reference')
    parser.add_argument('new_stage')
    parser.add_argument('output')
    args = parser.parse_args()
    compare(ROOT / args.old_reference, ROOT / args.new_stage, ROOT / args.output)
