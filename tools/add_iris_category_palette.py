"""Refresh bilingual color keys in existing reports and ZIPs; never edit masks."""
import argparse
import hashlib
import html
import json
import re
import zipfile
from pathlib import Path

START = '<!-- IRIS_BILINGUAL_PALETTE_START -->'
END = '<!-- IRIS_BILINGUAL_PALETTE_END -->'
STYLE = '''<style>
.category-palette{background:#fff;border:1px solid #c7d7e0;border-radius:10px;padding:14px 18px;margin:0 0 22px;box-shadow:0 3px 12px #17334b12}
.category-palette h2{margin:0 0 8px;font-size:23px}.category-palette p{margin:8px 0;font-size:14px}
.category-palette summary{font-weight:700;color:#123446;cursor:pointer;padding:2px 0}
.category-palette.docked{position:sticky;top:8px;z-index:20}
.category-palette .palette-scroll{overflow:auto;max-height:65vh;margin-top:10px}
.category-palette table{min-width:630px;font-size:14px}.category-palette th{position:sticky;top:0;background:#eaf1f5}
.category-palette td{vertical-align:middle;padding:6px 8px}.palette-swatch{display:inline-block;width:30px;height:20px;border:1px solid #0003;border-radius:3px;vertical-align:middle;margin-right:8px}
.palette-chip{display:inline-flex;align-items:center;gap:5px;border:1px solid #d2dde5;background:#f8fafc;border-radius:5px;padding:3px 7px;margin:3px 4px 3px 0;color:#203040;text-decoration:none;font-size:13px}
.palette-chip .palette-swatch{width:17px;height:17px;margin:0;flex-shrink:0}
html{scroll-padding-top:75px}
@media(max-width:650px){.category-palette{padding:10px}.category-palette h2{font-size:20px}.category-palette summary{font-size:14px}}
</style>'''


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def color(c):
    return '#'+''.join(f'{v:02X}' for v in c['display_color'])


def chip(c, reference):
    label = html.escape(f"{c['id']} · {c['name_zh']} / {c['official_semantic_name']}")
    return f'<a class="palette-chip" href="{reference}#class-{c["id"]}"><span class="palette-swatch" style="background:{color(c)}"></span>{label}</a>'


def palette(classes, reference, docked):
    rows = ''
    for c in classes:
        cid = c['id']
        name = html.escape(c['official_semantic_name'])
        english = name if cid == 255 else f'<a href="{reference}#class-{cid}">{name}</a>'
        rows += f'<tr data-category-id="{cid}"><td>{cid}</td><td>{html.escape(c["name_zh"])}</td><td>{english}</td><td><span class="palette-swatch" style="background:{color(c)}"></span><code>{color(c)}</code></td><td>{", ".join(map(str,c["display_color"]))}</td></tr>'
    content = '<p>16 个 IRIS 原始官方语义类别 + 255（ignore）。ID 与英文名称沿用官方，中文为解释性名称；颜色是本报告统一显示配色，原始 CAD 实例颜色另行保留。</p>'
    content += '<div class="palette-scroll"><table aria-label="IRIS 中英文类别与显示颜色"><thead><tr><th>ID</th><th>中文名称</th><th>Official category</th><th>色块 / HEX</th><th>RGB</th></tr></thead><tbody>'+rows+'</tbody></table></div>'
    content += '<p><b>9 / MISCELLANEOUS（杂项）不是未知或背景。</b>没有可靠标注的像素保留 255 / ignore（未标注／忽略）；没有新增类别 ID。</p>'
    if docked:
        block = '<details id="category-colors" class="category-palette docked"><summary>类别与颜色对照 · 中文 / English · 点击展开</summary>'+content+'</details>'
    else:
        block = '<section id="category-colors" class="category-palette"><h2>类别与颜色对照 · 中文 / English</h2>'+content+'</section>'
    return START+STYLE+block+END


def update_html(path, classes, reference, docked=False):
    body = path.read_text(encoding='utf-8')
    body = re.sub(re.escape(START)+'.*?'+re.escape(END), '', body, flags=re.S)
    assert '<main>' in body, path
    body = body.replace('<main>', '<main>'+palette(classes, reference, docked), 1)
    if docked:
        by_id = {c['id']:c for c in classes}
        def face_classes(match):
            ids = [int(v) for v in re.findall(r'\d+', match.group(1))]
            keys = ''.join(chip(by_id[v], reference) for v in ids) or '无官方类别；全图为 ignore（255）'
            return '源文件类别 ID：'+match.group(1)+'；<span class="face-category-keys">'+keys+'</span>官方合成覆盖'
        body = re.sub(r'源文件类别 ID：([^；]+)；(?:<span class="face-category-keys">.*?</span>)?官方合成覆盖', face_classes, body)
    assert body.count(START) == 1
    assert body.count('data-category-id=') == 17
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write(body)


def refresh(root):
    root = Path(root)
    cfg = read(root/'taxonomy/classes.json')
    classes = sorted(cfg['classes'], key=lambda c:c['id'])
    assert len(classes) == 16 and cfg['ignore_id'] == 255
    classes = classes+[{'id':255, 'official_semantic_name':'ignore', 'name_zh':'未标注／忽略', 'display_color':[35,40,50]}]
    update_html(root/'report.html', classes, 'taxonomy/index.html')
    update_html(root/'taxonomy/index.html', classes, '')
    records = read(root/'package_index.json')
    changed_pages = 2
    for r in records:
        cid = r['camera_id']
        folder = root/'images'/cid
        update_html(folder/'index.html', classes, 'taxonomy/index.html', docked=True)
        update_html(folder/'taxonomy/index.html', classes, '')
        changed_pages += 2
        hashes = read(folder/'checksums.json')
        for name in ['index.html', 'taxonomy/index.html']:
            hashes[name] = digest(folder/name)
        write(folder/'checksums.json', hashes)
        replacements = {str(Path(cid)/n).replace('\\','/'):(folder/n).read_bytes() for n in ['index.html','taxonomy/index.html','checksums.json']}
        archive = root/r['zip']
        temporary = archive.with_suffix('.palette.tmp')
        with zipfile.ZipFile(archive) as old, zipfile.ZipFile(temporary, 'w') as new:
            assert set(replacements) <= set(old.namelist())
            for info in old.infolist():
                data = replacements[info.filename] if info.filename in replacements else old.read(info.filename)
                new.writestr(info, data, compress_type=info.compress_type, compresslevel=3)
        temporary.replace(archive)
        r.update(zip_bytes=archive.stat().st_size, zip_sha256=digest(archive))
        with zipfile.ZipFile(archive) as z:
            for name, data in replacements.items():
                assert z.read(name) == data
        print('Updated palette:', cid, flush=True)
    write(root/'package_index.json', records)
    result={'passed':True, 'scope':'bilingual category/color keys only; original masks and analysis unchanged',
            'html_pages_updated':changed_pages, 'zip_packages_updated':len(records),
            'official_categories':16, 'ignore_id':255, 'languages':['zh','en'],
            'changed_package_files':['index.html','taxonomy/index.html','checksums.json'],
            'archive_replacement_bytes_checked':True}
    write(root/'category_palette_update.json', result)
    print(json.dumps(result,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report_root',type=Path)
    refresh(parser.parse_args().report_root)
