"""Export compact labels for the current 26 views; no training or source overwrite."""
from pathlib import Path
import collections,csv,json,sys,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,ref,sha
from prepare_iris_fullclass_gpt import EXPLAIN,COLORS

RUN=ROOT/'runs/iris26_semantic_consolidation_20261025'
OUT=RUN/'review'
DATA=ROOT.parent/'datasets/iris_official20_6_no_tripod_2k_20261025'
EXPORT=DATA/'semantics_iris26_compact7_v3'
PREVIOUS=DATA/'semantics_iris26_compact7_v2'
CONFIG=ROOT/'configs/iris26_compact7_classes.json'
CORRECTIONS=ROOT/'configs/iris26_misc_object_corrections.json'
EN=['Piping + valves','Equipment + cabinets','Building','Metal supports + beams','Access structures','Cable trays','HVAC ducts']
OLD_EN={2:'Gratings',5:'Cable trays',6:'Beams',7:'Electrical boxes',9:'Miscellaneous',12:'Civil structures',13:'Fire-fighting',16:'Mechanical equipment',19:'Doors',21:'Guardrails',24:'Supports',27:'Piping',29:'Valves',30:'HVAC',31:'Ladders',32:'Electrical equipment'}

def save(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    im=Image.fromarray(a) if isinstance(a,np.ndarray) else a
    im.save(p,**({'quality':95} if p.suffix=='.jpg' else {}))

def font(n):
    for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,n)

def main():
    if read(CONFIG)['version']=='iris26_compact7_v4':
        from fill_iris26_ignore import main as complete_labels
        return complete_labels()
    cfg=read(CONFIG);audit=read(RUN/'source_audit.json');review=read(CORRECTIONS);classes=cfg['classes'];rows=audit['rows']
    assert sha(audit['source_coco']['path'])==audit['source_coco']['sha256']
    assert len(rows)==26 and len({r['image_id'] for r in rows})==26
    obj={o['annotation_id']:o for o in audit['misc_objects'] if o['official_id']==9}
    decisions={d['annotation_id']:d for d in review['decisions']};assert len(decisions)==len(review['decisions'])
    assert cfg['version']=='iris26_compact7_v3'
    assert set(decisions)<=set(obj) and all(d['target_id'] in [0,1,2,3,255] for d in decisions.values())
    pal=np.zeros((256,3),'u1');pal[255]=[30,36,45]
    oldpal=pal.copy()
    for cl,col in zip(sorted(EXPLAIN),COLORS):oldpal[cl]=col
    lut=np.full(256,255,'u1')
    for c in classes:
        pal[c['id']]=c['color']
        for i in c['official_ids']:lut[i]=c['id']
    weights=np.broadcast_to(np.cos((.5-(np.arange(1024)+.5)/1024)*np.pi)[:,None],(1024,2048))
    EXPORT.mkdir(exist_ok=True)
    perimage=[];manifest=[];misc_dest=collections.Counter();support_changes=[];stats=[]
    for r in rows:
        cid=r['camera_id'];name=r['image_id'];src=RUN/'source'/cid
        raw=np.array(Image.open(src/'labels_original.png'))
        ann=np.load(src/'annotation_ids.npy')
        valid=np.array(Image.open(DATA/'valid_masks'/(name+'.png')))>0
        rgb=np.array(Image.open(DATA/'images'/(name+'.png')).convert('RGB'))
        merged=lut[raw];provenance=np.where(merged!=255,1,0).astype('u1')
        support=np.zeros(raw.shape,bool);curb=np.zeros(raw.shape,bool);miscstats={}
        for aid,d in decisions.items():
            if obj[aid]['camera_id']!=cid:continue
            pick=(ann==aid)&(raw==9)&valid
            merged[pick]=d['target_id'];provenance[pick]=2 if d['target_id']!=255 else 0
            if d['target_id']==3:support|=pick
            if d['target_id']==2:curb|=pick
        # The user explicitly assigns the reconstructed nadir floor to building.
        # Keep RGB observation provenance separate from semantic validity.
        merged[~valid]=2;provenance[~valid]=3
        baseline=lut[raw];baseline[~valid]=255
        assert np.array_equal(merged[(raw!=9)&valid],baseline[(raw!=9)&valid])
        assert not ((merged==3)&(raw==9)&~support).any()
        assert set(np.unique(merged))<=set(range(7))|{255}
        assert np.all(merged[~valid]==2)
        previous=np.array(Image.open(PREVIOUS/'labels'/(name+'.png')))
        changed=merged!=previous
        assert np.array_equal(changed,curb|~valid)
        assert np.all(previous[changed]==255) and np.all(merged[changed]==2)
        assert np.array_equal(merged==3,previous==3)
        corrected=raw.copy();corrected[support]=24;corrected[~valid]=255
        for folder,arr in [('labels',merged),('valid_masks',(merged!=255).astype('u1')*255),('observed_valid_masks',((merged!=255)&valid).astype('u1')*255),('provenance',provenance)]:save(EXPORT/folder/(name+'.png'),arr)
        save(OUT/'before_semantics'/(cid+'.png'),pal[previous])
        changes=np.zeros(raw.shape,'u1');changes[curb]=1;changes[~valid]=2
        save(OUT/'building_changes'/(cid+'.png'),changes)
        bh=rgb.copy()
        for mask,color in [(curb,[35,255,80]),(~valid,[255,70,160])]:bh[mask]=(.35*rgb[mask]+.65*np.array(color)).astype('u1')
        save(OUT/'building_highlights'/(cid+'.jpg'),bh)
        for role,arr in [('final_semantics',pal[merged]),('support_corrected16',oldpal[corrected]),('support_changes',support.astype('u1')*255)]:save(OUT/role/(cid+'.png'),arr)
        overlay=rgb.copy();known=merged!=255;overlay[known]=(.55*rgb[known]+.45*pal[merged[known]]).astype('u1')
        save(OUT/'final_overlays'/(cid+'.jpg'),overlay)
        highlight=rgb.copy();highlight[support]=(.35*rgb[support]+.65*np.array([255,85,40])).astype('u1');save(OUT/'support_highlights'/(cid+'.jpg'),highlight)
        old=np.array(Image.open(src/'labels_effective.png'))
        for k in range(7):
            m=merged==k;stats.append({'camera_id':cid,'split':r['split'],'class_id':k,'pixels':int(m.sum()),'erp_percent':float(m.mean()*100),'sphere_percent':float(weights[m].sum()/weights.sum()*100)})
        for k in [0,1,2,3,255]:
            m=(raw==9)&valid&(merged==k);miscstats[str(k)]={'pixels':int(m.sum()),'sphere_percent':float(weights[m].sum()/weights.sum()*100)};misc_dest[k]+=int(m.sum())
        info={'camera_id':cid,'image_id':name,'split':r['split'],'official_annotation_available':True,
            'source_labels':ref(src/'labels_original.png'),'rgb':ref(DATA/'images'/(name+'.png')),
            'labels':ref(EXPORT/'labels'/(name+'.png')),'valid_mask':ref(EXPORT/'valid_masks'/(name+'.png')),'provenance':ref(EXPORT/'provenance'/(name+'.png')),
            'support_reclassified_pixels':int(support.sum()),'misc_destinations':miscstats,
            'building_curb_pixels':int(curb.sum()),'bottom_building_pixels':int((~valid).sum()),
            'building_curb_annotation_ids':[aid for aid,d in decisions.items() if obj[aid]['camera_id']==cid and d['target_id']==2],
            'observed_valid_mask':ref(EXPORT/'observed_valid_masks'/(name+'.png')),
            'rgb_observation_mask':ref(DATA/'valid_masks'/(name+'.png')),
            'previous_labels':ref(PREVIOUS/'labels'/(name+'.png')),
            'ignored_erp_percent':float((~known).mean()*100),'ignored_sphere_percent':float(weights[~known].sum()/weights.sum()*100),
            'reason_for_ignore_counts':{'rare_fire_class':int(((raw==13)&valid).sum()),'unresolved_miscellaneous':int(((raw==9)&valid&(merged==255)).sum()),'source_unlabelled_or_conflict':int(((raw==255)&valid).sum())}}
        manifest.append(info)
        support_changes.append({'camera_id':cid,'pixels':int(support.sum()),'annotation_ids':[aid for aid,d in decisions.items() if obj[aid]['camera_id']==cid and d['target_id']==3]})
        print('EXPORTED',cid,'supports',int(support.sum()),flush=True)
    summary=[]
    for c in classes:
        ss=[x for x in stats if x['class_id']==c['id']]
        summary.append({'id':c['id'],'name':c['name'],'zh':c['zh'],'erp_percent':sum(x['erp_percent'] for x in ss)/26,'sphere_percent':sum(x['sphere_percent'] for x in ss)/26,'pixels':sum(x['pixels'] for x in ss),'views':sum(x['pixels']>0 for x in ss),'train_views':sum(x['pixels']>0 and x['split']=='train' for x in ss),'test_views':sum(x['pixels']>0 and x['split']=='test' for x in ss)})
    totals={'classes_before':audit['classes'],'classes_after':summary,'per_image':stats,
        'mean_ignore_sphere_percent':sum(r['ignored_sphere_percent'] for r in manifest)/26,
        'mean_ignore_erp_percent':sum(r['ignored_erp_percent'] for r in manifest)/26,
        'miscellaneous_final_pixels':dict(misc_dest),'support_changes':support_changes,
        'reviewed_misc_annotation_records':len(decisions),'all_misc_annotation_records':len(obj),
        'reviewed_misc_face_area_fraction':sum(obj[a]['pixels_face'] for a in decisions)/sum(o['pixels_face'] for o in obj.values()),
        'reviewed_support_annotation_records':sum(d['target_id']==3 for d in decisions.values()),
        'support_corrected_camera_count':sum(r['support_reclassified_pixels']>0 for r in manifest),
        'reviewed_curb_annotation_records':sum(d['target_id']==2 for d in decisions.values()),
        'building_curb_pixels':sum(r['building_curb_pixels'] for r in manifest),
        'bottom_building_pixels':sum(r['bottom_building_pixels'] for r in manifest),
        'building_corrected_camera_count':sum(r['building_curb_pixels']>0 and r['bottom_building_pixels']>0 for r in manifest),
        'note':'Instance records repeat objects across faces/views. Counts are not physical object counts. Percentages use cosine-latitude ERP weighting and the entire sphere denominator. Reviewed curb/platform skirts and reconstructed floor are building; synthetic floor provenance is retained separately.'}
    write(RUN/'statistics.json',totals);write(OUT/'statistics.json',totals)
    write(EXPORT/'classes.json',cfg);write(OUT/'classes.json',cfg);write(OUT/'object_corrections.json',review)
    write(EXPORT/'manifest.json',{'schema':'iris26_compact7_label_export_v3','version':cfg['version'],'source_audit':ref(RUN/'source_audit.json'),'taxonomy':ref(CONFIG),'object_corrections':ref(CORRECTIONS),
        'train_count':20,'test_count':6,'validation_count':0,'ignore_id':255,'classes_count':7,'resolution':[2048,1024],'records':manifest,
        'original_official_COCO_preserved':True,'label_kind':'derived class consolidation and selected RGB-inspected object corrections; inherited CAD boundaries are not a complete pixel-level manual correction',
        'tripod_edit_regions_excluded':False,'reconstructed_floor_class_id':2,'human_approved':False,'training_started':False,
        'previous_export':ref(PREVIOUS/'manifest.json'),
        'mask_integration':'valid_masks = labels != 255, including reconstructed floor. observed_valid_masks additionally excludes RGB synthesis for optional observed-only analysis. RGB observation masks remain independent and unchanged; no training configuration is modified.'})
    shutil.copy2(EXPORT/'manifest.json',OUT/'manifest.json')
    fields=['id','zh','erp_percent','sphere_percent','pixels','views','train_views','test_views']
    with (OUT/'class_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['stage']+fields);w.writeheader()
        for stage,ss in [('original16',audit['classes']),('compact7',summary)]:
            for row in ss:w.writerow({'stage':stage,**{k:row[k] for k in fields}})
    # Static comparison: same area denominator, no arbitrary minimum-presence filter.
    fig,axes=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1.1,1]})
    for ax,ss,names,title in [(axes[0],[s for s in audit['classes'] if s['id']!=255],[f'{i} {OLD_EN[i]}' for i in sorted(EXPLAIN)],'Original 16 classes'),(axes[1],summary,EN,'Consolidated 7 classes')]:
        vals=[s['sphere_percent'] for s in ss];cs=[oldpal[s['id']]/255 if ax==axes[0] else pal[s['id']]/255 for s in ss]
        bars=ax.barh(np.arange(len(ss)),vals,color=cs);ax.set_yticks(np.arange(len(ss)),names);ax.invert_yaxis();ax.set_xlim(0,40);ax.set_xlabel('Mean sphere area (%)');ax.set_title(title,fontweight='bold');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for b,v in zip(bars,vals):ax.text(v+.35,b.get_y()+b.get_height()/2,f'{v:.3f}%',va='center',fontsize=9)
    fig.suptitle('IRIS 20 train + 6 test | v3: curbs and reconstructed floor = building',fontsize=17);fig.tight_layout();fig.savefig(OUT/'class_area_comparison.png',dpi=160);plt.close(fig)
    legend=Image.new('RGB',(1280,300),'white');dr=ImageDraw.Draw(legend)
    for i,c in enumerate(classes):
        x=20+(i%4)*315;y=15+(i//4)*135;dr.rectangle((x,y,x+44,y+44),fill=tuple(c['color']));dr.text((x+56,y+3),f"{c['id']} {c['zh']}",font=font(20),fill='#183848');dr.text((x,y+56),f"{summary[i]['sphere_percent']:.2f}% | 20 train / 6 test",font=font(17),fill='#587180')
    dr.rectangle((965,150,1009,194),fill=tuple(pal[255]));dr.text((1021,154),'255 忽略区域',font=font(20),fill='#183848');save(OUT/'legend.png',legend)
    # Evidence of the user-requested selective correction, using identical old colors.
    for cid in ['1-24','1-26','1-36','1-64']:
        panel=Image.new('RGB',(1536,3*800),'#f1f4f7');dr=ImageDraw.Draw(panel)
        for j,(folder,label) in enumerate([('original_semantics','Original 16 classes'),('support_corrected16','Only selected misc -> supports (brown)'),('final_semantics','Final 7 classes')]):
            dr.text((12,j*800+8),cid+' | '+label,font=font(24),fill='#193446');panel.paste(Image.open(OUT/folder/(cid+'.png')).resize((1536,768),Image.Resampling.NEAREST),(0,j*800+32))
        save(OUT/'comparisons'/(cid+'.jpg'),panel)
    sheet=Image.new('RGB',(1400,4*410),'#f1f4f7');dr=ImageDraw.Draw(sheet)
    for j,aid in enumerate([1459,13835,41729,43881,42484,10487,37357,3340]):
        o=obj[aid];d=decisions[aid];cid=o['camera_id'];folder=RUN/'source'/cid/'faces'/o['face'];rgb=np.array(Image.open(folder/'rgb.jpg'));m=np.load(folder/'annotation_ids.npy')==aid
        ov=rgb.copy();ov[m]=(.5*rgb[m]+.5*pal[d['target_id']]).astype('u1');box=o['crop_box'];x=(j%2)*700;y=(j//2)*410
        for i,arr in enumerate([rgb,ov]):
            im=Image.fromarray(arr).crop(box);im.thumbnail((340,350));sheet.paste(im,(x+i*350+(350-im.width)//2,y+52+(350-im.height)//2))
        target=classes[d['target_id']]['zh'] if d['target_id']!=255 else '保留原始9；精简标签ignore'
        dr.text((x+8,y+8),f"{cid} / ann {aid}: 杂项 → {target}",font=font(20),fill='#183848')
    save(OUT/'object_decision_examples.jpg',sheet)
    qa={'checks_passed':True,'images':26,'train':20,'test':6,'val':0,'original_COCO_sha256_unchanged':True,'all_ids_valid':True,'all_reconstructed_floor_is_building':True,'only_white_listed_misc_can_become_support':True,
        'v2_to_v3_only_selected_ignore_to_building':True,'all_support_labels_unchanged_from_v2':True,'semantic_valid_includes_reconstructed_floor':True,
        'non_misc_changes_exactly_follow_declared_class_map':True,'no_morphology_or_component_area_filter_applied':True,'source_rgb_modified':False,'training_started':False,'final_visual_review_pending':True}
    write(OUT/'qa.json',qa)
    write(EXPORT/'qa.json',qa)
    from report_iris26_building_revision import report
    report(cfg,audit,totals,manifest,review)
    with zipfile.ZipFile(OUT/'iris26_compact7_v3_labels.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in EXPORT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(EXPORT))
        z.write(CORRECTIONS,'object_corrections.json');z.write(OUT/'class_statistics.csv','class_statistics.csv')
    shutil.copy2(OUT/'iris26_compact7_v3_labels.zip',OUT/'iris26_compact7_labels.zip')
    # Advertise the additional version, without changing RGB training defaults.
    dm=read(DATA/'dataset_manifest.json');dm['semantic_annotation']={'version':cfg['version'],'manifest':ref(EXPORT/'manifest.json'),'classes':ref(EXPORT/'classes.json'),'status':'derived_labels_ready_for_review','training_started':False};write(DATA/'dataset_manifest.json',dm)
    print(json.dumps({'after':summary,'support_records':totals['reviewed_support_annotation_records'],'support_views':totals['support_corrected_camera_count'],'misc_dest_pixels':dict(misc_dest),'ignore_sphere_percent':totals['mean_ignore_sphere_percent'],'export':str(EXPORT)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
