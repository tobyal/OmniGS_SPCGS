"""Offline held-out-view browser built only from saved final evaluations."""
import html,json,shutil
import numpy as np
from PIL import Image
from .common import read,write,ref
from .ipsgr_protocol import protocol,MODEL_NAMES


def build_viewer(run,models=None):
    names=list(MODEL_NAMES if models is None else models);assert set(names)<=set(MODEL_NAMES) and "B-SEM" in names
    from matplotlib import colormaps
    m=protocol(run);reference=run/'06_evaluation/reference';reference.mkdir(exist_ok=True)
    classes=m['classes']['classes'];records=[];low=10.;high=-10.
    for name in [n for n in names if n!='B-RGB']:
        base=run/'06_evaluation'/name
        for v in m['test']:
            scores=np.load(base/(v['name']+'_query_response.npy'));valid=np.fromfile(base/'native'/(v['name']+'_ids.i32'),dtype='i4').reshape(128,256)>=0
            if valid.any():low=min(low,float(scores[:,valid].min()));high=max(high,float(scores[:,valid].max()))
    results={name:read(run/'06_evaluation'/name/'metrics.json') for name in names}
    cmap=colormaps['viridis']
    for v in m['test']:
        from pathlib import Path
        filename=v['camera_id']+Path(v['image']).suffix;shutil.copy2(v['image'],reference/filename)
        row={'id':v['camera_id'],'name':v['name'],'rgb':'../06_evaluation/reference/'+filename,'gt':'../06_evaluation/B-SEM/'+v['name']+'_gt.png','models':{}}
        for name in names:
            base=run/'06_evaluation'/name;prefix='../06_evaluation/'+name+'/';metric=next(x for x in results[name]['views'] if x['id']==v['name'])
            item={'rgb':prefix+'native/'+v['name']+'_rgb.png','metrics':{k:metric[k] for k in ['psnr','ssim','lpips']},'metrics_file':prefix+'metrics.json'}
            if name!='B-RGB':
                item.update(semantic=prefix+v['name']+'_semantic.png',query_label=prefix+v['name']+'_query.png');item['metrics']['miou']=metric['semantic']['miou'];item['queries']=[];item['relative_queries']=[]
                maps=base/'query_maps';maps.mkdir(exist_ok=True);scores=np.load(base/(v['name']+'_query_response.npy'));valid=np.fromfile(base/'native'/(v['name']+'_ids.i32'),dtype='i4').reshape(128,256)>=0
                for k in range(7):
                    pixels=(cmap(np.clip((scores[k]-low)/max(high-low,1e-6),0,1))[...,:3]*255).astype('u1');pixels[~valid]=0;file=v['name']+f'_q{k}.png';Image.fromarray(pixels).save(maps/file);item['queries'].append(prefix+'query_maps/'+file)
                    relative=np.load(base/(v['name']+'_query_probabilities.npy'))[k];relative_pixels=(cmap(relative)[...,:3]*255).astype('u1');relative_pixels[~valid]=0;relative_file=v['name']+f'_qs{k}.png';Image.fromarray(relative_pixels).save(maps/relative_file);item['relative_queries'].append(prefix+'query_maps/'+relative_file)
            row['models'][name]=item
        records.append(row)
    data={'views':records,'models':names,'classes':[{'name':c['name'],'color':c['color']} for c in classes],'query_scale':[low,high]}
    write(run/'10_report/viewer_manifest.json',{'data':data,'script':ref(__file__),'source_metrics':{n:ref(run/'06_evaluation'/n/'metrics.json') for n in names},'note':'Only completed-model held-out final renders; no invented novel camera positions. Semantic images are native256x128, displayed with nearest scaling. All query maps share a range across the available semantic models and six views.'})
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IPSGR · 六图测试浏览</title>
<style>[hidden]{display:none!important}body{margin:0;background:#f3f6f8;color:#21374b;font:16px/1.6 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1{margin:10px 0 4px;font-size:30px}p{margin:6px 0 18px}a{color:#135e94}nav{display:flex;gap:24px;flex-wrap:wrap}label{display:flex;align-items:center;gap:8px}select{font:inherit;padding:8px 16px;border:1px solid #b6c6d2;border-radius:6px;background:white;color:#21374b}.controls{display:flex;gap:24px;flex-wrap:wrap;padding:18px 0;position:sticky;top:0;background:#f3f6f8f5;z-index:2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;background:white;border:1px solid #dce5ed;border-radius:8px;overflow:hidden}figcaption{padding:10px 14px;font-weight:600}img{display:block;width:100%;aspect-ratio:2/1;object-fit:contain;background:#20252b}.semantic{image-rendering:pixelated}.empty{aspect-ratio:2/1;display:grid;place-items:center;background:#e6edf2;color:#53697b}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{display:inline-block;width:16px;height:16px;border-radius:2px}.stats{display:flex;gap:24px;flex-wrap:wrap;padding:12px 16px;background:#e4eef5;border-radius:6px;margin:0 0 20px}h2{margin:32px 0 10px}small{color:#53697b}.scale{height:18px;background:linear-gradient(90deg,#440154,#414487,#2a788e,#22a884,#7ad151,#fde725);margin:12px 0 2px}.ends{display:flex;justify-content:space-between}.query-header{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:15px 0}button{font:inherit;padding:7px 16px;border:1px solid #b6c6d2;border-radius:6px;background:white;cursor:pointer}@media(max-width:850px){main{padding:16px}.grid{grid-template-columns:1fr}.controls{position:static}h1{font-size:25px}}</style>
<main><nav><a id="report-link" href="report.html">← 返回实验报告</a><a id="metrics-link">当前模型原始指标</a></nav><h1>六张测试视角的实际预测</h1><p>20 张训练输入 · 6 张独立测试 · 7 类工业语义。点击图像查看完整文件。<br><strong id="completion"></strong></p>
<div class="controls"><label>测试相机 <select id="view"></select></label><button id="prev">上一张</button><button id="next">下一张</button><label>模型 <select id="model"></select></label></div><div class="stats" id="stats"></div>
<div class="grid"><figure><figcaption>真实测试 RGB（修复版本）</figcaption><a id="rgb-gt-link"><img id="rgb-gt"></a></figure><figure><figcaption>模型 RGB 预测</figcaption><a id="rgb-link"><img id="rgb"></a></figure><figure><figcaption>固定七类语义真值</figcaption><a id="gt-link"><img id="gt" class="semantic"></a></figure><figure><figcaption>学习分类头的语义预测</figcaption><a id="semantic-link"><img id="semantic" class="semantic"></a><div id="no-sem" class="empty" hidden>此模型仅训练 RGB</div></figure></div>
<div class="legend" id="legend"></div><small>RGB 为 2048×1024；主语义为原生 256×128，展示时最近邻放大。黑色未覆盖像素已计入评价错误，不是新增语义类别。</small>
<section id="queries"><h2>固定 CLIP 文本原型查询</h2><div class="query-header"><label>查询类别 <select id="class"></select></label><label>响应显示 <select id="response"><option value="relative">七类内相对响应（Softmax）</option><option value="raw">原始文本相似度</option></select></label></div><div class="grid"><figure><figcaption id="query-caption">当前类别的查询响应</figcaption><a id="query-link"><img id="query" class="semantic"></a></figure><figure><figcaption>七类直接文本查询的最大响应类别</figcaption><a id="query-label-link"><img id="query-label" class="semantic"></a></figure></div><div class="scale"></div><div class="ends"><span id="low"></span><span id="high"></span></div><small id="query-note"></small></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);$('report-link').hidden=data.models.length<6;function options(el,rows){for(const [value,text]of rows){const o=document.createElement('option');o.value=value;o.textContent=text;el.append(o)}}options($('view'),data.views.map((v,i)=>[i,v.id]));options($('model'),data.models.map(n=>[n,n]));$('model').value=data.models.includes('IPSGR-Full')?'IPSGR-Full':data.models.at(-1);$('completion').textContent=`当前已完成 ${data.models.length}/6 个核心模型${data.models.length<6?'，其余实验尚未完成。':'。'}`;options($('class'),data.classes.map((c,i)=>[i,c.name]));for(const c of data.classes){const s=document.createElement('span'),i=document.createElement('i');i.className='swatch';i.style.background=`rgb(${c.color.join(',')})`;s.append(i,document.createTextNode(c.name));$('legend').append(s)}function picture(id,path){$(id).src=path;$(id+'-link').href=path}function render(){const v=data.views[+$('view').value],m=v.models[$('model').value];picture('rgb-gt',v.rgb);picture('rgb',m.rgb);picture('gt',v.gt);$('semantic-link').hidden=!m.semantic;$('no-sem').hidden=!!m.semantic;$('queries').hidden=!m.semantic;if(m.semantic){picture('semantic',m.semantic);picture('query',($('response').value==='relative'?m.relative_queries:m.queries)[+$('class').value]);picture('query-label',m.query_label)}$('metrics-link').href=m.metrics_file;$('stats').replaceChildren();for(const [name,value] of Object.entries(m.metrics)){const el=document.createElement('span');el.textContent=name==='miou'?`mIoU ${(value*100).toFixed(2)}%`:`${name.toUpperCase()} ${value.toFixed(4)}`;$('stats').append(el)}const relative=$('response').value==='relative';$('low').textContent=relative?'0.000':data.query_scale[0].toFixed(3);$('high').textContent=relative?'1.000':data.query_scale[1].toFixed(3);$('query-caption').textContent=relative?'当前类别 / 七类内相对响应（Softmax）':'当前类别 / 原始文本相似度';$('query-note').textContent=relative?'对七个固定文本原型响应作 Softmax，所有模型、视角使用 0–1 色标；它不是经过校准的置信度，也不是学习分类头的概率。':'原始相似度乘以固定温度系数 10，不是概率；所有可浏览模型、视角、类别共享色标。这不是任意开放词汇查询。'}for(const id of ['view','model','class','response'])$(id).addEventListener('change',render);$('prev').onclick=()=>{$('view').value=(+$('view').value+data.views.length-1)%data.views.length;render()};$('next').onclick=()=>{$('view').value=(+$('view').value+1)%data.views.length;render()};render();
</script></html>'''
    # The HTML is self-contained apart from local experiment images/metrics.
    (run/'10_report/viewer.html').write_text(page.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    return run/'10_report/viewer.html'
