"""Read-only original-COCO import and image assets; no models or training.

Uses the existing validated geometry and RLE functions, never the seven-class map.
Run --ids 1-43 1-44 first, inspect pilot assets, then run without --ids.
"""
import argparse, collections, csv, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, read, write, sha
from spcgs.segmentation import decode_rle, from_faces, inverse_face
from spcgs.clip_dense import FACES, perspective, erp_rays

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(2)
BASE = ROOT.parent
OUT = ROOT/'runs/iris_fullclass_gpt_annotation_20261025'
SOURCE = BASE/'datasets/images_xphase_1_cubemap_semantics_1500/_source'
COCO = SOURCE/'annotations_segmentation_cubemaps_xphase.json'
SPLIT = ROOT/'runs/omnigs_compact18_6_20261025/approved_split.json'
N = 1500
CODES = dict(zip('FRBLUD', [f for f,_ in FACES]))
COLORS = [[214,154,72],[232,99,194],[123,146,253],[255,213,65],
          [189,153,213],[116,131,146],[250,68,68],[62,195,135],
          [202,194,93],[48,217,229],[165,106,67],[255,123,46],
          [212,60,117],[124,186,238],[179,232,82],[78,108,205]]
PROJECT_MAPS = None
COCO_SHA = None

def project_faces(faces):
    global PROJECT_MAPS
    if PROJECT_MAPS is None:
        rays=erp_rays(1024,2048)
        owner=np.stack([(rays@r)[...,2] for _,r in FACES]).argmax(0)
        PROJECT_MAPS=[]
        for i,(name,_) in enumerate(FACES):
            yy,xx,valid,_=inverse_face(name,N,90);pick=(owner==i)&valid
            PROJECT_MAPS.append((name,pick,yy[pick],xx[pick]))
        sample={name:np.full((N,N),i,np.uint8) for i,(name,_) in enumerate(FACES)}
        expected=from_faces(sample)
        assert np.array_equal(project_faces(sample),expected)
    out=np.full((1024,2048),255,np.uint8)
    for name,pick,yy,xx in PROJECT_MAPS:out[pick]=faces[name][yy,xx]
    return out
EXPLAIN = {
2:('格栅','可见的金属网格、格栅踏面或平台。','与 12 连续楼板、21 扶手分开。'),
5:('电缆桥架','电缆承载槽、托盘和桥架结构；不据此把所有散线标成桥架。','与 6 梁、30 风管、27 管道区分；观察槽形边缘和走线。'),
6:('梁与钢结构构件','承重或框架型钢梁、立柱等；具体范围以 CAD 类别示例为准。','与 24 管道局部托架区别；连续主结构与局部支撑是线索。'),
7:('电气箱柜','电箱、接线箱、控制柜的外壳。','与 32 电气设备、16 机械设备易混；柜门、锁、开关面板是线索。'),
9:('杂项','官方保留的杂项 CAD 对象；不是所有未知像素的兜底类。','无对象功能或官方实例依据时保持 ignore。'),
12:('土建结构','地面、墙体、混凝土基础等土建构件。','与 2 格栅、6 钢构、16 设备机座区分；可见结构和材质仅为线索。'),
13:('消防设施','明确属于消防系统的设施。','红色不是消防类别依据；红色工艺容器、管线不能仅按颜色归类。'),
16:('机械设备','泵、机械装置、容器等候选；包含范围须参照官方实例。','电机独立部分可能归 32；大型容器须核查官方示例，不能只凭形状定类。'),
19:('门','建筑门及门扇类对象。','柜门属于箱柜候选，不自动划入 19。'),
21:('护栏','安全栏杆、扶手等防护结构。','与 24 支撑杆件、31 梯子区分；连续扶手和立柱布局是线索。'),
24:('支撑件','管道或设备的托架、吊架、支座等支撑构件。','与 6 主体钢梁、21 护栏区分；需要观察连接关系。'),
27:('管道','工艺管道及其连续弯头、管段；保温外表是否同类依据原标注。','与 29 阀门、30 风管、5 桥架区分；不要将连接设备整体吞入。'),
29:('阀门','具有调节或截断功能的阀体、手轮/执行部件候选。','法兰、接头不必然是阀门；与 27 边界结合官方实例核查。'),
30:('暖通设施','通风、采暖、空调相关设施。','与 27 工艺管、5 桥架区分，不能仅按截面或白色外观定类。'),
31:('梯子','直梯、爬梯等登高结构。','与 21 护栏、2 格栅踏面区别；必须看见梯级或完整结构关系。'),
32:('电气设备','当前实拍官方可靠实例包括长条灯具；散线、仪表等是否纳入仍需确认，不能仅按电气属性扩张。','与 7 箱柜、16 机械设备区分。当前黑色泵电机常整体属于官方16，不自动改为32。')}

def save_image(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    im = Image.fromarray(a) if isinstance(a,np.ndarray) else a
    im.save(p, **({'quality':94,'subsampling':0} if p.suffix=='.jpg' else {}))

def font(n=26):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)

def board(folder,kind,with_erp=False):
    top=940 if with_erp else 0
    canvas=Image.new('RGB',(1800,top+1280),(245,246,248));d=ImageDraw.Draw(canvas)
    if with_erp:
        d.text((12,5),folder.name+' | ERP 2048 x 1024 | '+read(folder/'metadata.json')['split'],fill='black',font=font())
        canvas.paste(Image.open(folder/'erp/rgb.png').resize((1800,900)),(0,40))
    for i,(face,_) in enumerate(FACES):
        x=(i%3)*600;y=top+(i//3)*640
        d.text((x+10,y+5),face.upper()+' | '+kind+' | 90 deg',fill='black',font=font(24))
        canvas.paste(Image.open(folder/f'faces/{face}/{kind}.jpg').resize((600,600)),(x,y+40))
    save_image(folder/f'{kind}_board.jpg',canvas)

def taxonomy(data):
    classes=[]
    for idx,cid in enumerate(sorted({c['id'] for c in data['categories']})):
        raw=[c for c in data['categories'] if c['id']==cid];names=sorted({c['supercategory'] for c in raw})
        assert len(names)==1
        zh,definition,confusion=EXPLAIN[cid]
        classes.append({'id':cid,'official_semantic_name':names[0],'name_zh':zh,
          'definition_zh':definition,'definition_status':'解释性中文释义；源 COCO 没有自然语言定义，不扩写为官方规则',
          'confusion_notes':confusion,'cad_object_names':[c['name'] for c in raw],
          'raw_category_record_count':len(raw),'display_color':COLORS[idx],
          'instance_linkage':'COCO annotation only has category_id; individual CAD name cannot be recovered.',
          'isthing_values':sorted({c.get('isthing') for c in raw})})
    write(OUT/'taxonomy/classes.json',{'ignore_id':255,'classes':classes,'raw_records':'raw_categories.json',
      'semantic_key':'id','official_semantic_name_field':'supercategory','cad_object_name_field':'name',
      'background_class':None,'warning':'Do not use id=9 MISCELLANEOUS as unknown/background.'})
    write(OUT/'taxonomy/raw_categories.json',data['categories'])
    with (OUT/'taxonomy/classes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['id','official_semantic_name','name_zh','cad_object_count','cad_object_names','definition_zh','confusion_notes'])
        for c in classes:w.writerow([c['id'],c['official_semantic_name'],c['name_zh'],c['raw_category_record_count'],';'.join(c['cad_object_names']),c['definition_zh'],c['confusion_notes']])
    palette=np.zeros((256,3),np.uint8);palette[255]=[35,40,50]
    legend=Image.new('RGB',(1150,650),'white');draw=ImageDraw.Draw(legend)
    for i,c in enumerate(classes):
        palette[c['id']]=c['display_color'];x=(i//8)*570;y=(i%8)*75+20
        draw.rectangle((x+10,y,x+65,y+45),fill=tuple(c['display_color']));draw.text((x+80,y+7),f"{c['id']:02d} {c['official_semantic_name']}",fill='black',font=font(25))
    save_image(OUT/'taxonomy/legend.png',legend)
    return classes,palette

def align_check():
    rows=[];name='2023-05-11_10.18.46'
    rgb=np.array(Image.open(BASE/f'datasets/iris_full_A_official_2k/images/{name}.png').convert('RGB'))
    for code,face in CODES.items():
        ref=np.array(Image.open(SOURCE/f'reference_rgb/{name}_{code}.png').convert('RGB').resize((500,500)))
        pred=perspective(rgb,dict(FACES)[face],500,90)
        corr=float(np.corrcoef(ref.mean(-1).ravel(),pred.mean(-1).ravel())[0,1])
        rows.append({'face':face,'source_code':code,'correlation':corr,'mae':float(np.abs(ref.astype(float)-pred).mean())})
        assert corr>.93,(face,corr)
    # Round-trip sampling convention at random pixels, including seam and polar faces.
    rng=np.random.default_rng(4);errors=[]
    for face,R in FACES:
        xy=rng.uniform(0,1499,(1000,2));q=np.c_[(xy+.5)/1500*2-1,np.ones(len(xy))];r=q@R.T;r/=np.linalg.norm(r,axis=1)[:,None]
        erp=np.c_[(np.arctan2(r[:,0],r[:,2])/(2*np.pi)+.5)*2048-.5,(np.arcsin(r[:,1])/np.pi+.5)*1024-.5]
        lon=((erp[:,0]+.5)/2048-.5)*2*np.pi;lat=((erp[:,1]+.5)/1024-.5)*np.pi
        r2=np.c_[np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)]@R
        back=(r2[:,:2]/r2[:,2:]+1)*750-.5;errors.append(float(np.abs(back-xy).max()))
    write(OUT/'projection_validation.json',{'reference_image':name,'official_reference_faces':rows,'roundtrip_max_face_pixel_error':max(errors),
       'note':'Checks projection convention, not CAD-to-photo registration or per-image semantic correctness.'})

def prepare(row,data,imindex,annotations,classes,palette):
    cid=row['camera_id'];folder=OUT/'images'/cid
    if (folder/'complete.json').exists():return read(folder/'metadata.json')
    folder.mkdir(parents=True,exist_ok=True);name=row['image_id']
    low=BASE/f'datasets/iris_full_A_official_2k/images/{name}.png'
    high=Path(row['rgb_source'])
    if not high.exists():high=BASE/f'images_xphase_1/{name}.jpg'
    rgb2=np.array(Image.open(low).convert('RGB'));assert rgb2.shape==(1024,2048,3)
    # Full source is 16K. 8K ERP preserves enough detail for 1500-pixel 90-degree faces.
    src=Image.open(high).convert('RGB');original_size=list(src.size);rgb8=np.array(src.resize((8192,4096),Image.Resampling.LANCZOS));del src
    save_image(folder/'erp/rgb.png',rgb2)
    meta={**row,'rgb_2k_source':str(low),'rgb_2k_sha256':sha(low),'rgb_highres_source':str(high),'rgb_highres_sha256':sha(high),
      'rgb_highres_size':original_size,'face_rgb_input_size':[8192,4096],'face_size':N,'fov_deg':90,'ignore_id':255,
      'annotation_source':'official_COCO_CAD_projection' if row['has_official_semantics'] else 'none',
      'human_confirmed':False,'visual_review_status':'pending','faces':[]}
    write(folder/'metadata.json',meta)
    rawfaces={};conflicts={};perclass={c['id']:{} for c in classes};rawimages=[];rawann=[]
    for code,face in CODES.items():
        d=folder/'faces'/face;d.mkdir(parents=True,exist_ok=True)
        frgb=perspective(rgb8,dict(FACES)[face],N,90);save_image(d/'rgb.jpg',frgb)
        im=imindex.get(name+'_'+code);ann=annotations[im['id']] if im else []
        if im:assert im['width']==im['height']==N;rawimages.append(im)
        rawann.extend(ann);masks={};objects=[]
        for a in ann:
            mask=decode_rle(a['segmentation']);assert mask.shape==(N,N)
            assert int(mask.sum())==int(a['area']),(a['id'],mask.sum(),a['area'])
            cl=a['category_id'];assert cl in perclass
            masks[cl]=masks.get(cl,np.zeros((N,N),bool))|mask
            objects.append({'annotation_id':a['id'],'category_id':cl,'pixels':int(mask.sum()),'bbox_xywh':a['bbox'],'cad_instance_name':None})
        label=np.full((N,N),255,np.uint8);count=np.zeros((N,N),np.uint8)
        stats=[]
        for c in classes:
            cl=c['id'];mask=masks.get(cl,np.zeros((N,N),bool));perclass[cl][face]=mask.astype('u1')
            if mask.any():
                save_image(d/f'class_masks/{cl:02d}.png',mask.astype('u1')*255);label[mask]=cl;count+=mask
                stats.append({'id':cl,'pixels':int(mask.sum()),'fraction':float(mask.mean())})
        conflict=count>1;label[conflict]=255;rawfaces[face]=label;conflicts[face]=conflict.astype('u1')
        save_image(d/'labels.png',label);save_image(d/'conflict.png',conflict.astype('u1')*255)
        save_image(d/'provenance.png',np.where(conflict,2,np.where(label!=255,1,0)).astype('u1'))
        sem=palette[label];save_image(d/'semantic.jpg',sem)
        overlay=frgb.copy();known=label!=255;overlay[known]=(.52*frgb[known]+.48*sem[known]).astype('u1');overlay[conflict]=[255,0,255]
        save_image(d/'overlay.jpg',overlay)
        # Boundary-only audit facilitates spotting CAD silhouettes covering foreground objects.
        edge=np.zeros((N,N),np.uint8)
        for m in masks.values():edge|=cv2.morphologyEx(m.astype('u1'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        boundary=frgb.copy();boundary[edge>0]=[255,255,0];save_image(d/'boundaries.jpg',boundary)
        face_meta={'face':face,'official_code':code,'official_image_id':im['id'] if im else None,'has_official_record':bool(im),
          'objects':objects,'classes':stats,'union_coverage_fraction':float((count>0).mean()),'exclusive_coverage_fraction':float(known.mean()),
          'conflict_pixels':int(conflict.sum()),'ignore_fraction':float((label==255).mean()),'empty_class_masks':'Unlisted IDs are all zero; never background.'}
        meta['faces'].append(face_meta);write(d/'statistics.json',face_meta)
    labels=project_faces(rawfaces);conflict=project_faces(conflicts)>0
    save_image(folder/'erp/labels.png',labels);save_image(folder/'erp/conflict.png',conflict.astype('u1')*255)
    save_image(folder/'erp/provenance.png',np.where(conflict,2,np.where(labels!=255,1,0)).astype('u1'))
    sem=palette[labels];save_image(folder/'erp/semantic.jpg',sem)
    overlay=rgb2.copy();known=labels!=255;overlay[known]=(.52*rgb2[known]+.48*sem[known]).astype('u1');overlay[conflict]=[255,0,255];save_image(folder/'erp/overlay.jpg',overlay)
    stats=[]
    for cl,fs in perclass.items():
        m=project_faces(fs)>0
        if m.any():save_image(folder/f'erp/class_masks/{cl:02d}.png',m.astype('u1')*255)
        stats.append({'id':cl,'pixels':int(m.sum()),'fraction':float(m.mean())})
    meta.update(erp_class_stats=stats,erp_coverage_fraction=float(known.mean()),erp_conflict_pixels=int(conflict.sum()),annotation_count=len(rawann))
    assert sum(f['has_official_record'] for f in meta['faces'])==(6 if row['has_official_semantics'] else 0)
    write(folder/'official_annotations.json',{'images':rawimages,'annotations':rawann,'categories_file':'taxonomy/raw_categories.json','source_sha256':COCO_SHA})
    write(folder/'metadata.json',meta);board(folder,'rgb',True);board(folder,'overlay');board(folder,'semantic')
    write(folder/'complete.json',{'prepared':True,'source_annotation_count':len(rawann),'visual_review_pending':True})
    print(cid,row['split'],'official',bool(rawann),'coverage',round(float(known.mean()),3),flush=True)
    return meta

def main():
    global COCO_SHA
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);data=read(COCO);split=read(SPLIT);COCO_SHA=sha(COCO)
    assert len(split['train_ids'])==18 and len(split['test_ids'])==6 and not split['validation_ids']
    write(OUT/'source_split.json',split)
    classes,palette=taxonomy(data);align_check()
    imindex={Path(x['file_name']).stem:x for x in data['images']};annotations=collections.defaultdict(list)
    for a in data['annotations']:annotations[a['image_id']].append(a)
    manifest={'schema':'iris_original16_gpt_materials_v1','split_path':str(SPLIT),'split_sha256':sha(SPLIT),'split_status':split['status'],
      'official_coco_path':str(COCO),'official_coco_sha256':sha(COCO),'official_semantic_class_count':len(classes),
      'cad_category_record_count':len(data['categories']),'ignore_id':255,'training_started_by_this_task':False,
      'annotation_is_human_confirmed':False,'images':[]}
    for row in sorted(split['selected_cameras'],key=lambda x:int(x['camera_id'].split('-')[1])):
        if args.ids and row['camera_id'] not in args.ids:continue
        meta=prepare(row,data,imindex,annotations,classes,palette)
        manifest['images'].append({k:meta[k] for k in ['camera_id','image_id','split','has_official_semantics','erp_coverage_fraction','erp_conflict_pixels','annotation_count']})
    manifest['partial_pilot']=bool(args.ids);write(OUT/'manifest.json',manifest)
    write(OUT/'coordinates.json',{'ERP':{'width':2048,'height':1024},'face':{'width':N,'height':N,'horizontal_fov_deg':90,'vertical_fov_deg':90},
      'pixel_convention':'zero-based pixel centers; x right, y down; xyxy boxes use exclusive right/bottom',
      'rotation_face_to_erp_ray':{n:r.tolist() for n,r in FACES},
      'face_to_erp':'q=[2*(x+0.5)/1500-1,2*(y+0.5)/1500-1,1]; r=R*q/||q||; u=(atan2(rx,rz)/(2*pi)+0.5)*2048-0.5; v=(asin(ry)/pi+0.5)*1024-0.5; u wraps modulo 2048.',
      'erp_to_face':'lon=((u+0.5)/2048-0.5)*2*pi; lat=((v+0.5)/1024-0.5)*pi; r=[cos(lat)*sin(lon),sin(lat),cos(lat)*cos(lon)]; q=R.T*r; x=(qx/qz+1)*750-0.5; y=(qy/qz+1)*750-0.5; valid qz>0 and x/y in [-0.5,1499.5].',
      'ownership':'face with maximum forward ray depth; ties in Front,Right,Back,Left,Up,Down order; nearest-neighbor ID sampling',
      'seams':'ERP boxes may cross u=0; use sampled perimeter polygons/wrapped coordinates, never one naive rectangle.',
      'provenance_values':{'0':'unlabelled/ignore','1':'official CAD imported, not visually certified truth','2':'cross-category overlap conflict; composite ignore; separate masks retained'},
      'RGB':'ERP supplied at native prepared 2K. Face RGB projected from original 16K resized to 8K. Coordinates share the same angular frame; no 100-degree faces included.'})

if __name__=='__main__':main()
