"""Publish inspected images, original taxonomy, GPT tasks and self-contained ZIPs.
Does not infer or edit semantic masks. Uses authored visual notes, not model calls.
"""
import collections,csv,html,json,math,shutil,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from prepare_iris_fullclass_gpt import OUT,ROOT,FACES,EXPLAIN,read,write,sha,save_image,font
from iris_fullclass_visual_notes import NOTES

REVIEW={
2:'实拍可见平台下侧密集格栅，类别对应有依据；掩码简化了栅缝和遮挡边界。',
5:'实拍确为网篮桥架及缆束；官方外轮廓近似实体块，不能把内孔、横撑和每根散线都当可靠桥架像素。',
6:'平台主梁及交叉梁可见，类别对应有依据；前景梯和线缆遮挡已部分扣除，细边界仍需复核。',
7:'白色电柜表面可作为类别参考；垂挂橙色接头/黑线被箱体掩码吞并的部分不是可靠箱柜像素。',
9:'当前官方把 transfero 白设备外壳归入杂项；可见外壳内部区域可参考。1-59 down 的纸张、孔洞不是可靠的9类示例。禁止把未知区域一律归9。',
12:'示例裁剪特意避开相机脚架，展示实际可见地坪；原始整面12掩码仍包含脚架，不可据此认为全掩码正确。',
13:'当前没有可视觉认证的可靠实例。两个官方13轮廓位于推车左侧红墙/软管附近，RGB可见带软管的推车式筒状设备，但无法确认消防身份，也可能是清洁/吸尘设备或遮挡。保留原始掩码作为疑点参考，不把整台推车称为消防器材。',
16:'官方实例包括黑色泵/电机连同白圆座，以及蓝色筒体；这反映源数据对象分组，不能自行把电机改为32。蓝筒底部构件的CAD形状与实拍可见几何有偏差，其支撑/连接功能仍需确认。',
19:'可见建筑双门及门把手，类别有依据；前景栏杆、管线遮挡较多，仅可见门面部分可参考，不应按完整CAD门面补齐。',
21:'护栏横杆与立柱确实可见；高竖杆及近旁管支撑的语义范围需要按官方实例关系核查，不把整个框内区域填成栏杆。',
24:'穿孔型钢支架可见，类别匹配；官方掩码把孔洞与部分附着缆线整体填实，不能作为精细边界真值。',
27:'工艺管及保温外壳清楚可见；短端件、红蓝手柄和局部支撑被一起归27，应保留为待复核的类别粒度问题。',
29:'蓝色传动机构和黑手轮可辨，官方29有依据；手轮内部孔洞被填实，1-36手轮外轮廓亦有轻微错位。',
30:'银色通风管和弯头可辨，类别与投影相符；反光和拼接缝不是新类别，边缘仍可能有CAD与实物几何差异。',
31:'爬梯的梯级、立杆与护笼清楚可见，类别有依据；细线和平台梁穿过梯子时应按前后遮挡关系处理。',
32:'当前例子是长条灯具，官方32有明确依据；过曝光晕不属于灯壳，其他视角相似灯体有被桥架5吞并的情况。'}

PROMPT='''IRIS 原始全类别语义补标与纠错任务

你正在分析一张真实工业 ERP 全景。请先列出你实际看到了哪些图像附件（包含 ERP、Front/Right/Back/Left/Up/Down、官方叠加和参考例）。仅看到服务器路径、文件名、JSON 或 ZIP 清单不算看到图像。如果工具能够解压 ZIP，请解压后逐张打开图像；否则请用户把 upload_first 中的图片和相关 faces/*/rgb.jpg 作为图像附件上传。未看到的区域不能宣称检查过。

资料说明：2048×1024 ERP；六面各1500×1500，90度水平/垂直视场。原始类别ID为2,5,6,7,9,12,13,16,19,21,24,27,29,30,31,32；255=ignore，没有背景类。categories中name是CAD名称，supercategory是语义名；同一ID有多个CAD名称，但annotation没有具体CAD实例名。不要把候选实例名称填成事实。

任务：
1. 阅读 taxonomy/classes.json、class_references.json 和类别图片，结合 RGB 实际识别物体。
2. 对每个面比较 RGB、官方语义与叠加图。官方CAD投影不是逐像素人工真值，覆盖率100%不代表无漏标；检查脚架、纸张、散线、袋装物、孔洞及真实遮挡。
3. 审核 tasks.json 的优先对象；框/点只是 AI 初步提示。逐面继续扫描未列出的对象，尤其无官方标注图像，不能只处理已有框。
4. 分别列出：保留官方、建议补充、建议改类、建议修边、保持未知、提出新类建议。对有官方标注的区域提供原始类ID与建议类ID；二者不得混淆。
5. 新类建议与官方16类分表，不能重定义官方类别。优先验证官方9/32等是否已经覆盖，不因为难判断就新建类；相机/拼接伪影优先ignore。
6. 返回符合 decision_schema.json 的结构化JSON。每个对象包含图像ID、face、原始/建议类ID、零起点1500像素xyxy框、正/负点、可见边界描述、证据图片、类别依据、置信程度与未解决问题。不要把描述、框、矩形或语言模型估计伪装成精确掩码。
7. 只有在本视角实际可见的区域提出分割。跨视角引用默认只是相似类别/邻近场景，不能直接复制掩码、不能补出遮挡背面，未经证据不能声称同一CAD实例。
8. 坐标以单张1500面图为准，不是拼图坐标。可用 coordinates.json 和 coordinate_tools.py 转换到ERP；ERP接缝处不能用一个跨全幅的大矩形。裁剪图坐标应先加 task.crop_box_xyxy 左上角。
9. 没有足够依据就保持 proposed_category_id=null、action=keep_unknown；不要强制归背景或9。后续由GPT/用户确认语义及提示，再用SAM2 Hiera-L生成精确像素掩码，本任务不运行SAM或训练。

请先给逐面检查结果，再输出JSON。任何确认仅表示本轮GPT分析结论；human_confirmed必须保持false，直到真实人工确认。
'''

COORD_CODE='''"""Standalone pixel-coordinate helper. Requires numpy. No masks generated."""
import json, numpy as np
from pathlib import Path
D=json.loads((Path(__file__).parent/'coordinates.json').read_text(encoding='utf-8'))
def face_to_erp(face,xy):
 xy=np.asarray(xy,float);q=np.concatenate([(xy+.5)/750-1,np.ones(xy.shape[:-1]+(1,))],-1)
 r=q@np.array(D['rotation_face_to_erp_ray'][face]).T;r/=np.linalg.norm(r,axis=-1,keepdims=True)
 return np.stack([((np.arctan2(r[...,0],r[...,2])/(2*np.pi)+.5)*2048-.5)%2048,(np.arcsin(r[...,1])/np.pi+.5)*1024-.5],-1)
def erp_to_face(face,uv):
 uv=np.asarray(uv,float);lon=((uv[...,0]+.5)/2048-.5)*2*np.pi;lat=((uv[...,1]+.5)/1024-.5)*np.pi
 r=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],-1)@np.array(D['rotation_face_to_erp_ray'][face])
 xy=(r[...,:2]/r[...,2:]+1)*750-.5
 return xy,(r[...,2]>0)&(xy>=-.5).all(-1)&(xy<=1499.5).all(-1)
'''

def f2e(face,xy):
    xy=np.array(xy,float);q=np.c_[(xy+.5)/750-1,np.ones(len(xy))];r=q@dict(FACES)[face].T;r/=np.linalg.norm(r,axis=1)[:,None]
    return np.c_[((np.arctan2(r[:,0],r[:,2])/(2*np.pi)+.5)*2048-.5)%2048,(np.arcsin(r[:,1])/np.pi+.5)*1024-.5].round(3).tolist()

def perimeter(box):
    x1,y1,x2,y2=box;x2-=1;y2-=1
    return [[x,y1] for x in np.linspace(x1,x2,12)]+[[x2,y] for y in np.linspace(y1,y2,12)]+[[x,y2] for x in np.linspace(x2,x1,12)]+[[x1,y] for y in np.linspace(y2,y1,12)]

def img(path,caption=''):
    return f'<figure><a href="{path}" target="_blank"><img loading="lazy" src="{path}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'

STYLE='''body{margin:0;background:#f2f5f8;color:#203040;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:28px}header{background:#123446;color:white;padding:32px}h1{margin:0 0 8px;font-size:30px}h2{margin-top:38px}a{color:#087293}header a{color:#9de4f4}nav{display:flex;gap:20px;flex-wrap:wrap}section,.card,article{background:white;border:1px solid #d8e1e8;border-radius:12px;padding:20px;margin:18px 0}figure{margin:0;min-width:0}img{max-width:100%;height:auto;display:block;border-radius:6px}figcaption{font-size:13px;color:#526779;margin:6px 0 15px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.notice{background:#fff4d9;border-left:5px solid #e2aa38;padding:15px}.badge{display:inline-block;background:#e6f3ed;color:#176746;padding:3px 9px;border-radius:6px;margin-right:8px}.missing{background:#fff0dd;color:#91520b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dbe2e8;padding:8px;text-align:left;vertical-align:top}code,pre{background:#edf2f6}pre{overflow:auto;padding:15px;font-size:13px;white-space:pre-wrap}summary{cursor:pointer}small{color:#526779}.swatch{display:inline-block;width:18px;height:18px;margin-right:8px;vertical-align:middle}.task{border-left:4px solid #387ca0;padding:12px 16px;background:#f4f8fb} @media(max-width:850px){.grid,.pair{grid-template-columns:1fr}main{padding:14px}header{padding:22px}h1{font-size:24px}}'''

def page(p,title,body,nav=''):
    p.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style><header><h1>'+html.escape(title)+'</h1><nav>'+nav+'</nav></header><main>'+body+'</main></html>',encoding='utf-8')

def prep_taxonomy():
    cfg=read(OUT/'taxonomy/classes.json')
    for c in cfg['classes']:
        zh,desc,confuse=EXPLAIN[c['id']];c.update(name_zh=zh,definition_zh=desc,confusion_notes=confuse)
    write(OUT/'taxonomy/classes.json',cfg)
    with (OUT/'taxonomy/classes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['id','official_semantic_name','name_zh','cad_object_count','cad_object_names','definition_zh','confusion_notes'])
        for c in cfg['classes']:
            writer.writerow([c['id'],c['official_semantic_name'],c['name_zh'],c['raw_category_record_count'],';'.join(c['cad_object_names']),c['definition_zh'],c['confusion_notes']])
    refs=read(OUT/'taxonomy/class_references.json');by={x['id']:x for x in cfg['classes']}
    for r in refs:
        r.update(by[r['id']]);r['visual_reference_review']=REVIEW[r['id']];r['reviewer']='Codex visual inspection, not human confirmation'
        r['no_reliable_example']=r['id']==13;r['reliable_scope']='no visually certified example' if r['id']==13 else 'visible corresponding object interiors only; not whole-mask pixel certification'
        for i,e in enumerate(r['examples']):
            e['visual_review']='AI_inspected';e['human_confirmed']=False;e['review_notes']=REVIEW[r['id']]
            e['suitable_as_whole_mask_ground_truth']=False
            e['example_use']='category_appearance_reference_with_boundary_caveats'
            if r['id']==9 and i==0:e['example_use']='known_occlusion_error_example_not_clean_class_reference'
            if r['id']==13:e['example_use']='official_mask_disputed_visual_identity_not_reliable_class_reference'
    write(OUT/'taxonomy/class_references.json',refs)
    return cfg,refs

def decision_schema():
    point={'type':'array','items':{'type':'number','minimum':0,'maximum':1499},'minItems':2,'maxItems':2}
    ids=[2,5,6,7,9,12,13,16,19,21,24,27,29,30,31,32,None]
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':'GPT annotation decisions, not pixel masks','type':'object',
      'required':['camera_id','images_actually_seen','human_confirmed','decisions'],'properties':{
       'camera_id':{'type':'string'},'images_actually_seen':{'type':'array','items':{'type':'string'}},'human_confirmed':{'const':False},
       'decisions':{'type':'array','items':{'type':'object','required':['task_id','face','action','original_category_ids','proposed_category_id','bbox_face_xyxy','positive_points','negative_points','visible_boundary','evidence_images','reason','confidence','uncertainty','is_precise_mask'],
        'properties':{'task_id':{'type':'string'},'face':{'enum':[n for n,_ in FACES]},'action':{'enum':['retain_official','add_candidate','correct_class_candidate','correct_boundary_candidate','keep_unknown','propose_new_class']},
          'original_category_ids':{'type':'array','items':{'enum':[x for x in ids if x is not None]+[255]}},'proposed_category_id':{'enum':ids},
          'new_category_proposal_key':{'type':['string','null']},'bbox_face_xyxy':{'type':['array','null'],'items':{'type':'number','minimum':0,'maximum':1500},'minItems':4,'maxItems':4},
          'positive_points':{'type':'array','items':point},'negative_points':{'type':'array','items':point},'visible_boundary':{'type':'string'},
          'evidence_images':{'type':'array','items':{'type':'string'}},'reason':{'type':'string'},'confidence':{'enum':['low','medium','high']},'uncertainty':{'type':'string'},'is_precise_mask':{'const':False}}}}}}

def make_tasks(folder,meta,refs,neighbors):
    lines=[x.strip() for x in NOTES[folder.name].splitlines() if x.strip()];assert len(lines)==6,(folder.name,len(lines))
    output={'camera_id':folder.name,'image_id':meta['image_id'],'split':meta['split'],'split_status':read(OUT/'manifest.json')['split_status'],
      'review_status':'AI_visual_inspected_awaiting_GPT_and_human','human_confirmed':False,'is_precise_mask':False,'faces':[],
      'references_note':'Class references show similar objects; nearby views show spatial context. No verified same-CAD-instance association.',
      'priority_regions_exhaustive':False,'full_six_face_scan_required':True,'neighbor_references':neighbors}
    cropboard=Image.new('RGB',(1800,1450),'white');draw=ImageDraw.Draw(cropboard)
    closeboard=Image.new('RGB',(1800,1280),'white');cd=ImageDraw.Draw(closeboard)
    for i,((face,_),line) in enumerate(zip(FACES,lines)):
        scene,audit,obj,ids,box,point=line.split('|');ids=[int(x) for x in ids.split(',') if x]
        b=[round(float(x)*1.5) for x in box.split(',')];pt=[min(1499,round(float(x)*1.5)) for x in point.split(',')]
        assert 0<=b[0]<b[2]<=1500 and 0<=b[1]<b[3]<=1500 and b[0]<=pt[0]<b[2] and b[1]<=pt[1]<b[3],(folder.name,face,b,pt)
        d=folder/'faces'/face;rgb=Image.open(d/'rgb.jpg');labels=np.array(Image.open(d/'labels.png'))
        cropbox=[max(0,b[0]-60),max(0,b[1]-60),min(1500,b[2]+60),min(1500,b[3]+60)]
        save_image(d/'priority_crop_rgb.jpg',rgb.crop(cropbox));save_image(d/'priority_crop_overlay.jpg',Image.open(d/'overlay.jpg').crop(cropbox))
        marked=rgb.copy();md=ImageDraw.Draw(marked);md.rectangle(b,outline='#ff007f',width=5);x,y=pt;md.ellipse((x-9,y-9,x+9,y+9),fill='#00ffff',outline='black',width=2)
        for tick in range(0,1500,250):
            md.text((tick+4,4),str(tick),font=font(22),fill='yellow',stroke_width=2,stroke_fill='black')
            md.text((4,tick+25),str(tick),font=font(22),fill='yellow',stroke_width=2,stroke_fill='black')
        save_image(d/'priority_region.jpg',marked)
        close=marked.crop(cropbox);close.thumbnail((590,590))
        cx=(i%3)*600;cy=(i//3)*640
        cd.text((cx+10,cy+5),f'{folder.name} {face} | crop',font=font(24),fill='black')
        closeboard.paste(close,(cx+(600-close.width)//2,cy+45))
        vals,cnts=np.unique(labels[b[1]:b[3],b[0]:b[2]],return_counts=True)
        fm=next(x for x in meta['faces'] if x['face']==face)
        candidates=[{'category_id':v,'status':'visual_hypothesis_not_truth'} for v in ids]
        task={'task_id':folder.name+'_'+face+'_01','object_description':obj,'candidate_categories':candidates,
          'action_requested':'full_face_annotation_plus_priority_object' if not meta['has_official_semantics'] else 'review_official_class_boundary_and_occlusion',
          'bbox_face_xyxy':b,'bbox_is_approximate':True,'positive_points':[pt],'negative_points':[],
          'point_label':[1],'prompt_status':'AI_suggested_verify_on_full_face_before_SAM2','crop_box_xyxy':cropbox,
          'crop_to_face':'add crop_box_xyxy[0:2] to crop coordinates; crops are not resized on disk',
          'erp_positive_points':f2e(face,[pt]),'erp_bbox_perimeter_samples':f2e(face,perimeter(b)),
          'erp_geometry_note':'These are the projected perimeter of a prompt box, NOT an object polygon or mask. Split at ERP seam.',
          'official_id_at_suggested_point':int(labels[pt[1],pt[0]]),
          'official_ids_inside_prompt_box':{str(int(v)):int(n) for v,n in zip(vals,cnts)},
          'visible_boundary_guidance':'以本面 RGB 中的可见外观与实际遮挡为界；框含上下文，不把框全部当目标。'+audit,
          'evidence_images':[f'faces/{face}/rgb.jpg',f'faces/{face}/overlay.jpg',f'faces/{face}/priority_crop_rgb.jpg'],
          'category_reference_images':[f'taxonomy/reference_{v:02d}.jpg' for v in ids],
          'uncertainty':'类别与精细边界尚未确认；提示点也需GPT核查。遮挡背面不推断。',
          'human_confirmed':False,'is_precise_mask':False,'final_category_id':None}
        output['faces'].append({'face':face,'visual_scene_description':scene,'visual_audit':audit,
          'objects_to_inventory':scene.split('，'),'official_label_facts':fm,
          'official_vs_visual_note':'统计是源文件事实，视觉类别候选不改变标签。未逐像素认证。',
          'unlabelled_scope':'entire_face' if not fm['has_official_record'] else 'inspect_real_foreground_missing_from_CAD_even_if_coverage_is_100_percent',
          'priority_tasks':[task]})
        x=(i%3)*600;y=(i//3)*725
        draw.text((x+12,y+6),f'{folder.name} {face} | proposed prompt',font=font(23),fill='black')
        im=marked.copy();im.thumbnail((590,650));cropboard.paste(im,(x,y+40))
    save_image(folder/'priority_regions_board.jpg',cropboard)
    save_image(folder/'priority_crops_board.jpg',closeboard)
    write(folder/'tasks.json',output);return output

def class_html(refs):
    body='<p>16 个官方语义 ID · 748 条原始 CAD 名称。中文定义为解释性释义；原 COCO 没有文字定义。实例名称不能从 annotation.category_id 单独反推。</p>'+img('legend.png','显示色是本报告统一语义色，不替换原始 CAD 颜色。')
    body+='<p><a href="classes.csv">完整类别 CSV</a> · <a href="classes.json">类别 JSON</a> · <a href="raw_categories.json">748 条原始 categories</a> · <a href="class_references.json">出现位置与参考来源</a></p>'
    for r in refs:
        body+=f'<section id="class-{r["id"]}"><h2>{r["id"]} · {r["official_semantic_name"]} · {r["name_zh"]}</h2><p>{r["definition_zh"]}</p><p><b>易混淆：</b>{r["confusion_notes"]}</p><p class="notice">{r["visual_reference_review"]}</p>'+img(f'reference_{r["id"]:02d}.jpg','真实 RGB / 单个原始官方 annotation 掩码叠加。仅类别外观参考，非整幅掩码已认证。')
        body+='<details><summary>官方 CAD 对象名称与关系（'+str(len(r['cad_object_names']))+' 条）</summary><p>'+html.escape(', '.join(r['cad_object_names']))+'</p><p>这些名称共享类别 ID；具体 annotation 的 CAD 名称未知。</p></details>'
        body+='<details><summary>当前视角出现位置（'+str(len(r['target_appearances']))+' 个面）</summary><table><tr><th>相机</th><th>划分</th><th>面</th><th>官方像素数</th></tr>'
        for a in r['target_appearances']:body+=f'<tr><td>{a["camera_id"]}</td><td>{a["split"]}</td><td>{a["face"]}</td><td>{a["pixels"]}</td></tr>'
        body+='</table></details></section>'
    page(OUT/'taxonomy/index.html','IRIS 官方全类别参考',body)

def image_html(folder,meta,tasks):
    flag='有官方六面记录' if meta['has_official_semantics'] else '无官方标注：全图 ignore'
    body=f'<p><span class="badge">{meta["split"]}</span><span class="badge {"" if meta["has_official_semantics"] else "missing"}">{flag}</span>{meta["image_id"]}</p>'
    body+='<p class="notice">官方导入与 AI 视觉分析均不等于人工确认真值。原始标签保持不变；候选框、点和投影周界不是像素掩码。无可靠依据时保持 ignore=255。</p>'
    body+='<nav><a href="START_HERE.txt">上传方法</a><a href="GPT_TASK.txt">标准 GPT 说明</a><a href="tasks.json">逐面任务 JSON</a><a href="taxonomy/index.html">完整类别参考</a><a href="coordinates.json">坐标关系</a><a href="official_annotations.json">原始 COCO RLE</a></nav>'
    body+='<h2>ERP 全景</h2>'+img('erp/rgb.png','完整 ERP RGB · 2048×1024')+'<div class="pair">'+img('erp/semantic.jpg','官方原始类别 · 深灰=ignore')+img('erp/overlay.jpg','官方掩码叠加')+'</div>'
    body+='<p><a href="erp/labels.png">原始 ID 标签 PNG</a> · <a href="erp/provenance.png">来源图</a> · <a href="metadata.json">逐类覆盖与 SHA256</a></p>'
    body+='<h2>整图视觉概述</h2><p>'+html.escape(' '.join(f['visual_scene_description'] for f in tasks['faces'][:4]))+'</p>'
    for f in tasks['faces']:
        n=f['face'];t=f['priority_tasks'][0];facts=f['official_label_facts'];classes=', '.join(str(c['id']) for c in facts['classes']) or '无'
        body+=f'<section id="{n}"><h2>{n.title()} · 90° · 1500×1500</h2><p><b>实拍所见：</b>{html.escape(f["visual_scene_description"])}</p><p><b>视觉核查：</b>{html.escape(f["visual_audit"])}</p><p>源文件类别 ID：{classes}；官方合成覆盖 {facts["exclusive_coverage_fraction"]:.2%}。这表示源掩码覆盖，不表示标注准确率。</p><div class="grid">'
        for kind,cap in [('rgb','RGB'),('semantic','官方原始类别'),('overlay','官方叠加')]:body+=img(f'faces/{n}/{kind}.jpg',cap)
        body+='</div><p><a href="faces/'+n+'/boundaries.jpg">只看原始掩码边界</a> · <a href="faces/'+n+'/statistics.json">类别/annotation 清单</a></p>'
        body+=f'<div class="task"><h3>{html.escape(t["object_description"])}</h3><p>候选 ID：{", ".join(str(c["category_id"]) for c in t["candidate_categories"]) or "未知 / ignore"}；提示点所在官方 ID：{t["official_id_at_suggested_point"]}。候选均未确认。</p><p>局部框 xyxy：{t["bbox_face_xyxy"]}；正点：{t["positive_points"]}；ERP 正点：{t["erp_positive_points"]}。</p><div class="grid">'+img(f'faces/{n}/priority_region.jpg','粉框=粗区域；青点=待复核 SAM 提示；不是新掩码')+img(f'faces/{n}/priority_crop_rgb.jpg','局部 RGB，原始像素裁剪')+img(f'faces/{n}/priority_crop_overlay.jpg','同区域原始官方叠加')+'</div></div></section>'
    body+='<h2>其他视角参考</h2><p>选取邻近相机的真实图片帮助核查遮挡和设备关系。未验证 CAD 实例身份；不可直接复制其掩码。</p><div class="pair">'
    for r in tasks['neighbor_references']:body+=img(r['rgb_board'],f"{r['camera_id']} / {r['split']} / 距离 {r['distance_m']:.2f} m")
    body+='</div><p><a href="taxonomy/proposed_classes.json">独立保存的新增类别建议</a> · <a href="decision_schema.json">GPT 输出结构</a></p>'
    page(folder/'index.html',f'IRIS {folder.name} · GPT 标注资料',body,'<a href="START_HERE.txt">从这里开始</a><a href="taxonomy/index.html">类别库</a>')

def proposals():
    return {'status':'proposals_only_no_official_ids_assigned','official_taxonomy_modified':False,'items':[
      {'proposal_key':'INSTRUMENTS','name_zh':'独立测量仪表','examples':['1-35/left','1-27/front'],'rationale':'圆形压力表可见，部分官方归27；是否需要独立类先确认9/32是否覆盖，不能直接判官方27错误。','fallback':'ignore if neither current taxonomy nor new proposal is confirmed'},
      {'proposal_key':'EXPOSED_CABLES','name_zh':'独立散线与软电缆','examples':['1-24/right','1-56/right','1-70/front'],'rationale':'箱柜外悬线、设备接线环无法等同5桥架；先核查32范围，必要时单独建类。'},
      {'proposal_key':'TEMPORARY_MATERIALS','name_zh':'临时材料与维护物品','examples':['1-25/left','1-59/front','1-56/right'],'rationale':'袋装材料、纸张等真实前景不在静态CAD中；9不应当未知兜底。若不研究这些对象可统一ignore，不强求新类。'}],
      'exclusion_recommendation':{'objects':'camera tripod / panorama stitching artifacts','semantic_class_id':None,'recommended_label':255,'reason':'采集装置不属于需重建的固定工业对象；原12标签不代表可靠地面真值。'},
      'not_proposed':'灯具已有官方32实例；蓝色容器已有16实例；箱柜已有7/9关系，因此不重复造类。'}

def main():
    manifest=read(OUT/'manifest.json');assert not manifest['partial_pilot'] and len(manifest['images'])==24
    metas={p.parent.name:read(p.parent/'metadata.json') for p in (OUT/'images').glob('*/complete.json')};assert set(metas)==set(NOTES)
    # Refine ambiguous language after seeing the nearer 1-46 endpoint with a blue handle.
    for cid in ['1-29','1-33','1-34']:
        NOTES[cid]=NOTES[cid].replace('黑色圆形开口','黑色圆形端面（开口/阀件待核查）').replace('黑色圆开口','黑色圆形端面').replace('黑色管口','黑色管端').replace('黑管口','黑色管端')
    cfg,refs=prep_taxonomy();write(OUT/'taxonomy/proposed_classes.json',proposals());class_html(refs)
    (OUT/'GPT_TASK.txt').write_text(PROMPT,encoding='utf-8');(OUT/'coordinate_tools.py').write_text(COORD_CODE,encoding='utf-8');write(OUT/'decision_schema.json',decision_schema())
    (OUT/'downloads').mkdir(exist_ok=True);alltasks=[];allrecords=[]
    for cid,meta in sorted(metas.items(),key=lambda kv:int(kv[0].split('-')[1])):
        folder=OUT/'images'/cid;neighbors=[]
        xyz=np.array([meta[k] for k in ['x_m','y_m','z_m']])
        others=sorted([r for k,r in metas.items() if k!=cid and r['has_official_semantics']],key=lambda r:np.linalg.norm(xyz-[r[k] for k in ['x_m','y_m','z_m']]))[:2]
        for r in others:
            dest=folder/'references'/r['camera_id'];dest.mkdir(parents=True,exist_ok=True)
            for n in ['rgb_board.jpg','overlay_board.jpg']:shutil.copy2(OUT/'images'/r['camera_id']/n,dest/n)
            neighbors.append({'camera_id':r['camera_id'],'split':r['split'],'distance_m':float(np.linalg.norm(xyz-[r[k] for k in ['x_m','y_m','z_m']])),
              'rgb_board':f'references/{r["camera_id"]}/rgb_board.jpg','overlay_board':f'references/{r["camera_id"]}/overlay_board.jpg','relation':'nearby_scene_reference; same_instance_unverified'})
        tasks=make_tasks(folder,meta,refs,neighbors);alltasks.append(tasks)
        shutil.copytree(OUT/'taxonomy',folder/'taxonomy',dirs_exist_ok=True)
        for n in ['GPT_TASK.txt','coordinates.json','coordinate_tools.py','decision_schema.json']:shutil.copy2(OUT/n,folder/n)
        upload=folder/'upload_first';upload.mkdir(exist_ok=True)
        save_image(upload/'01_ERP_RGB.jpg',Image.open(folder/'erp/rgb.png'))
        save_image(upload/'02_six_RGB.jpg',Image.open(folder/'rgb_board.jpg').crop((0,940,1800,2220)))
        shutil.copy2(folder/'overlay_board.jpg',upload/'03_six_OFFICIAL_overlay.jpg');shutil.copy2(OUT/'taxonomy/legend.png',upload/'04_class_legend.png');shutil.copy2(folder/'priority_regions_board.jpg',upload/'05_priority_regions.jpg')
        start=f'''IRIS {cid} / {meta['image_id']} / {meta['split']}
本包完全自包含，index.html 可直接本地打开。先看逐面分析与原始官方信息。

直接交给 GPT 的推荐步骤：
1. 解压本包。将 upload_first 中 5 张真实图片作为图像附件上传，同时上传 tasks.json、taxonomy/classes.json、GPT_TASK.txt。
2. 粘贴 GPT_TASK.txt 内容。明确要求 GPT 先列出实际看见的图片；仅知道路径或ZIP文件名不能代替视觉检查。
3. 一次重点处理一两个面，再上传对应 faces/<face>/rgb.jpg、overlay.jpg、priority_crop_rgb.jpg 和 taxonomy/reference_<ID>.jpg。需要核查邻视角时上传 references 内图片。
4. 坐标必须基于1500×1500原始面图，不是拼图坐标；粉框与青点只是候选提示。裁剪坐标需加 crop_box_xyxy 左上角。
5. 返回 decision_schema.json 结构。无依据保持null/ignore；本轮不运行SAM2、不新增最终掩码。后续确认后再用SAM2 Hiera-L。

官方记录：{'有六面COCO标注' if meta['has_official_semantics'] else '没有；所有标签是ignore=255'}。
六面优先对象只是核查入口，不是全部物体的最终清单；必须继续扫描整个面。
原始RLE在official_annotations.json；categories完整748记录在taxonomy/raw_categories.json。
原始label PNG每个像素值就是官方ID或255。semantic/overlay JPG仅用于看图，不能作为训练标签。
provenance PNG：0未知，1原始官方导入（未人工确认），2跨类别冲突。
类掩码在erp/class_masks与faces/*/class_masks；未出现的类别没有PNG，表示全零掩码而非背景。
没有类别合并；没有启动训练；没有生成任何补标后的像素真值。
'''
        (folder/'START_HERE.txt').write_text(start,encoding='utf-8')
        meta['visual_review_status']='AI_inspected_ERP_and_all_six_faces_not_human_confirmed';write(folder/'metadata.json',meta)
        write(folder/'complete.json',{'prepared':True,'visual_review_pending':False,'visual_review':'AI_not_human','final_annotation_pending':True})
        image_html(folder,meta,tasks)
        files=[f for f in folder.rglob('*') if f.is_file() and f.name!='checksums.json']
        write(folder/'checksums.json',{str(f.relative_to(folder)).replace('\\','/'):sha(f) for f in files})
        zip_path=OUT/'downloads'/f'{cid}_GPT_annotation_pack.zip'
        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for f in folder.rglob('*'):
                if f.is_file():z.write(f,str(Path(cid)/f.relative_to(folder)))
        allrecords.append({'camera_id':cid,'image_id':meta['image_id'],'split':meta['split'],'official':meta['has_official_semantics'],
          'tasks':6,'zip':'downloads/'+zip_path.name,'zip_bytes':zip_path.stat().st_size,'zip_sha256':sha(zip_path)})
        print('pack',cid,round(zip_path.stat().st_size/1e6,1),'MB',flush=True)
    write(OUT/'all_annotation_tasks.json',alltasks);write(OUT/'package_index.json',allrecords)
    write(OUT/'pilot_validation.json',{'pilot_ids':['1-43','1-44'],'official_and_unlabelled_checked':True,'ERP_and_all_faces_viewed':True,
      'findings':['1-43 has zero official records, all six faces ignore.','1-44 official labels cover nearly all pixels but miss actual tripod/paper occluders.','Official semantic key is category id, supercategory gives its name; 748 CAD aliases form 16 IDs.'],
      'projection_checks':'projection_validation.json','continued_to_all_24_after_visual_pilot':True})
    write(OUT/'visual_inspection_log.json',{'reviewer':'Codex actual image viewing, not human truth','panoramas_viewed':24,'faces_viewed':144,
      'official_overlay_faces_viewed':108,'category_example_pairs_viewed':32,'notes_source':str(ROOT/'tools/iris_fullclass_visual_notes.py'),
      'scope':'All ERP and six-face RGB boards; all official overlay boards; category example crops; all 144 priority crops checked separately, with 53 regions refined. Not pixel-exhaustive certification.',
      'images':[{'camera_id':r['camera_id'],'faces':[n for n,_ in FACES],'human_confirmed':False} for r in allrecords]})
    body='<p>最新紧凑区域 18 train + 6 test，无验证集。采用服务器 approved_split.json；18 张有官方六面记录，6 张全图未标注。</p><p class="notice">本资料面向阅读与 GPT 辅助标注。官方覆盖率接近100%仍会漏掉实拍遮挡物；9=杂项不是未知，255=ignore。所有分析及提示均未人工确认。未启动任何训练，也未生成新增像素真值。</p>'
    body+='<nav><a href="taxonomy/index.html">16类完整参考库</a><a href="GPT_TASK.txt">标准GPT任务说明</a><a href="all_annotation_tasks.json">全部任务JSON</a><a href="package_index.json">24包清单及校验值</a><a href="source_split.json">当前18+6清单</a><a href="projection_validation.json">投影验证</a></nav>'
    body+='<section><h2>本次视觉检查的主要结论</h2><ul><li>748条CAD名称共享16个语义ID；annotation不含具体CAD名称，不伪造实例对应。</li><li>蓝色容器、黑色泵/电机及部分圆座属于官方16；长条灯有官方32实例。</li><li>相似白设备外壳可能是7或9；transfero设备下部官方9、上部控制盒7，不能仅凭“像柜子”统一改类。</li><li>脚架、纸张、袋装物与散线可能被背景CAD标签吞并。手轮和支撑孔洞也有实心化现象。</li><li>类别参考图已区分可辨认对象与存在遮挡/错位的掩码；全部原始RLE、独立类别掩码保留。</li></ul><p>下载某张ZIP后先解压，把upload_first图片作为图像附件提供给GPT；ZIP中的文件路径不能替代看图。</p></section><div class="grid">'
    for r in allrecords:
        cid=r['camera_id'];body+='<article>'+img(f'images/{cid}/erp/rgb.png',f'{cid} · {r["split"]}')+f'<h3>{cid} <span class="badge {"" if r["official"] else "missing"}">{"官方标注" if r["official"] else "未标注"}</span></h3><p>{r["image_id"]}</p><p><a href="images/{cid}/index.html">逐面分析与资料</a> · <a download href="{r["zip"]}">独立 ZIP（{r["zip_bytes"]/1e6:.1f} MB）</a></p></article>'
    body+='</div><section><h2>原始数据与坐标说明</h2><p>ERP 2048×1024；六面1500×1500、90°。F/R/B/L/U/D沿用仓库已验证投影。与官方参考RGB相关系数0.992–0.998，连续坐标往返误差小于3×10⁻¹²面像素。该检查不证明CAD标签和实物完全一致。</p><p>独立类别掩码保留；跨类别重叠单独记冲突并在合成图设ignore。完整来源、统计及坐标程序均在每个包内。</p></section>'
    page(OUT/'report.html','IRIS 全类别语义分析 · GPT 标注资料',body)
    (OUT/'README.txt').write_text('打开 report.html 查看全部资料；每张独立ZIP在downloads。使用说明在每包START_HERE.txt。原始数据只读导入，未改写，未训练。官方类别16个，原始categories748条。所有视觉分析是AI候选，不是人工真值。\n',encoding='utf-8')
    from add_iris_category_palette import refresh
    refresh(OUT)
    print('DONE',len(allrecords),'packs',sum(r['tasks'] for r in allrecords),'priority tasks')

if __name__=='__main__':main()
