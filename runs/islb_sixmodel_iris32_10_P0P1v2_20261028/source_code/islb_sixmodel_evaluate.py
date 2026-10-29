"""Fixed 10-view RGB and eight-class evaluation; no training/model selection."""
from pathlib import Path
import os,json,csv,time,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import ROOT,read,write,ref,signature,native
from .islb_sixmodel_data import RUN,MODELS

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def confusion(gt,pred,mask=None,K=8):
    valid=(gt>=0)&(gt<K)
    if mask is not None:valid&=mask
    assert np.all((pred>=0)&(pred<=K))
    return np.bincount(gt[valid].astype('i8')*(K+1)+pred[valid].astype('i8'),minlength=K*(K+1)).reshape(K,K+1)

def semantic_metrics(cm):
    cm=np.asarray(cm,dtype='i8');K=len(cm);tp=np.diag(cm[:,:K]);support=cm.sum(1);union=support+cm[:,:K].sum(0)-tp
    iou=np.divide(tp,union,out=np.zeros(K,float),where=union>0);recall=np.divide(tp,support,out=np.zeros(K,float),where=support>0)
    return {'mIoU':float(iou[union>0].mean()) if (union>0).any() else None,'mAcc':float(recall[support>0].mean()) if (support>0).any() else None,
         'pixel_accuracy':float(tp.sum()/max(cm.sum(),1)),'IoU':[float(v) if union[i]>0 else None for i,v in enumerate(iou)],
         'recall':[float(v) if support[i]>0 else None for i,v in enumerate(recall)],'support':support.tolist(),'confusion_8x9':cm.tolist(),
         'uncovered_errors':int(cm[:,K].sum()),'coverage':float(1-cm[:,K].sum()/max(cm.sum(),1))}

def csvsave(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def colorize(labels,palette):
    colors=np.vstack([np.asarray(palette,'u1'),[[25,25,25]]]).astype('u1');return Image.fromarray(colors[labels])

def crop_periodic(im,box):
    x,y,w,h=box;W,H=im.size
    assert 0<=y and y+h<=H
    tiled=Image.new(im.mode,(W*2,H));tiled.paste(im,(0,0));tiled.paste(im,(W,0));return tiled.crop((x%W,y,x%W+w,y+h))

def freeze_rois():
    import cv2
    folder=RUN/'07_figures';target=folder/'fixed_rois.json'
    if target.exists():return read(target)
    m=read(RUN/'00_protocol/dataset_manifest.json');palette=[c['color'] for c in m['classes']['classes']];samples=[];candidates=[]
    # Select industrial content using GT only, before first model evaluation.
    for v in m['test']:
        rgb=Image.open(v['image']).convert('RGB');lab=np.array(Image.open(v['label']));valid=np.array(Image.open(v['supervision_mask']))>0
        samples.append((v,rgb,lab,valid))
        for kind,cid in [('valves_and_pipes',7),('equipment_boundary',1),('crossing_pipes_supports',0)]:
            mask=((lab==cid)&valid).astype('u1');n,components,stats,centers=cv2.connectedComponentsWithStats(mask,8)
            for i in np.argsort(stats[1:,cv2.CC_STAT_AREA])[-20:]+1:
                if stats[i,cv2.CC_STAT_AREA]<20:continue
                x,y=centers[i];cw,ch=640,360;box=[int(round(x-cw/2))%2048,int(np.clip(round(y-ch/2),0,1024-ch)),cw,ch]
                a=np.array(crop_periodic(Image.fromarray(lab),box));vm=np.array(crop_periodic(Image.fromarray(valid.astype('u1')),box))>0
                count=np.bincount(a[vm],minlength=8);area=cw*ch;frac=float(vm.mean())
                if frac<.90 or count[0]<.02*area:continue
                if kind=='valves_and_pipes':score=float(count[7]+.06*count[0]+.02*count[1])
                elif kind=='equipment_boundary':score=float(min(count[1],.45*area)+.25*count[0]+.2*count[3])
                else:score=float(min(count[0],.45*area)+.4*count[3]+.2*count[5]+.1*count[7])
                candidates.append({'kind':kind,'camera_id':v['camera_id'],'view':v['name'],'box_ERP_2k':box,'GT_valid_fraction':frac,'GT_class_pixels':count.tolist(),'GT_score':score})
    chosen=[];used=set()
    for kind in ['valves_and_pipes','equipment_boundary','crossing_pipes_supports']:
        eligible=sorted([a for a in candidates if a['kind']==kind and a['view'] not in used],key=lambda a:a['GT_score'],reverse=True)
        assert eligible,kind;r=eligible[0];r['roi_id']='ROI'+str(len(chosen)+1);chosen.append(r);used.add(r['view'])
    initial_chosen=[dict(r) for r in chosen]
    # GT-only visual review replaces a dark pole-adjacent ceiling crop with a
    # well-lit crossing-pipe / cable-tray / support-frame region. No prediction
    # or quantitative model result is used in this deterministic final choice.
    v,rgb,lab,valid=next(s for s in samples if s[0]['camera_id']=='1-60');box=[520,160,640,360]
    a=np.array(crop_periodic(Image.fromarray(lab),box));vm=np.array(crop_periodic(Image.fromarray(valid.astype('u1')),box))>0
    chosen[2]={'kind':'crossing_pipes_supports','camera_id':v['camera_id'],'view':v['name'],'box_ERP_2k':box,'GT_valid_fraction':float(vm.mean()),
        'GT_class_pixels':np.bincount(a[vm],minlength=8).tolist(),'GT_score':None,'roi_id':'ROI3','selection_note':'GT-only visual review: better illumination and crossing industrial structures than automatic ceiling crop.'}
    record={'frozen_before_first_evaluation':True,'used_predictions':False,'selection':'Three distinct held-out views; fixed640x360 ERP crops selected on valid GT industrial content only, then visually inspected. Same ROIs for F05(a,b,c).',
       'ROIs':chosen,'initial_automatic_selection':initial_chosen,'all_candidates':candidates,'created':time.time(),'classes':m['classes'],'primary_semantic_GT_mask':'provenance1/2 and original RGB observations'}
    write(target,record)
    # Full test contact sheet and GT-only ROI evidence for human review.
    font=ImageFont.truetype(FONT,20);sheet=Image.new('RGB',(1536,5*426),'white');d=ImageDraw.Draw(sheet)
    for i,(v,rgb,lab,valid) in enumerate(samples):
        x=(i%2)*768;y=(i//2)*426;d.text((x+5,y+3),v['camera_id'],font=font,fill='black');sheet.paste(rgb.resize((768,384)),(x,y+32))
    sheet.save(folder/'GT_test_contact.jpg',quality=95)
    sheet=Image.new('RGB',(1280,len(chosen)*402),'white');d=ImageDraw.Draw(sheet);by={v['name']:(v,rgb,lab,valid) for v,rgb,lab,valid in samples}
    for i,r in enumerate(chosen):
        v,rgb,lab,valid=by[r['view']];gt=lab.copy();gt[~valid]=8
        d.text((7,i*402+5),f'{r["roi_id"]} / {r["camera_id"]} / {r["kind"]}',font=font,fill='black')
        sheet.paste(crop_periodic(rgb,r['box_ERP_2k']),(0,i*402+37));sheet.paste(crop_periodic(colorize(gt,palette),r['box_ERP_2k']),(640,i*402+37))
    sheet.save(folder/'GT_ROI_evidence.png');print('ROIS_FROZEN',chosen,flush=True);return record

def evaluate(name,gpu):
    import torch,lpips
    from skimage.metrics import structural_similarity
    torch.set_num_threads(4);m=read(RUN/'00_protocol/dataset_manifest.json');cfg=read(RUN/'00_protocol/common_config.json');model=RUN/'05_models'/name
    assert read(model/'outputs.json')['status']=='completed' and read(model/'checkpoint_validation.json')['status']=='passed'
    archive=model/'final/model.pt';assert read(archive.parent/'model.json')['step']==10000
    rois=read(RUN/'07_figures/fixed_rois.json');assert rois['frozen_before_first_evaluation'] and not rois['used_predictions']
    out=RUN/'06_evaluation'/name;out.mkdir(parents=True,exist_ok=True);has_sem=name!='B-RGB';K=8;H,W=1024,2048;h,w=256,512
    inputs={'archive':ref(archive),'protocol':ref(RUN/'00_protocol/dataset_manifest.json'),'native':ref(cfg['native_binary']),'semantic_resolution':[w,h],'ROIs':ref(RUN/'07_figures/fixed_rois.json')};key=signature(inputs)
    if (out/'metrics.json').exists():assert read(out/'metrics.json')['signature']==key;return read(out/'metrics.json')
    write(out/'inputs.json',inputs)
    req={**cfg,'action':'render','archive':str(archive),'views':m['test'],'output':str(out/'native'),'gpu':gpu,'render_semantics':has_sem,'semantic_width':w,'semantic_height':h,'resume':False,'stage_reset':False}
    expected=['rgb.f32','rgb.png']+(['logits.f16','ids.i32','features.f16'] if has_sem else [])
    if not all((out/'native'/(v['name']+'_'+s)).is_file() for v in m['test'] for s in expected):native(req,out/f'render_job_{time.time_ns()}')
    device=torch.device(f'cuda:{gpu}');network=lpips.LPIPS(net='alex',version='0.1').eval().to(device)
    palette=[c['color'] for c in m['classes']['classes']];total=np.zeros((K,K+1),'i8');fulltotal=total.copy();querytotal=total.copy();rows=[]
    if has_sem:
        state=torch.jit.load(str(archive),map_location='cpu');decoder=state.decoder.numpy();bias=state.bias.numpy();text=state.text.numpy()
    for v in m['test']:
        prefix=out/'native'/v['name'];pred=np.fromfile(str(prefix)+'_rgb.f32','f4').reshape(3,H,W).transpose(1,2,0)
        gt=np.array(Image.open(v['image']).convert('RGB'),dtype='f4')/255.;assert pred.shape==gt.shape and np.isfinite(pred).all()
        with torch.inference_mode():
            lp=float(network(torch.from_numpy(pred.transpose(2,0,1).copy())[None].to(device)*2-1,torch.from_numpy(gt.transpose(2,0,1).copy())[None].to(device)*2-1).item())
        row={'view':v['name'],'camera_id':v['camera_id'],'PSNR':float(-10*np.log10(max(float(np.mean((pred-gt)**2)),1e-12))),
             'SSIM':float(structural_similarity(gt,pred,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),'LPIPS':lp}
        if has_sem:
            logits=np.fromfile(str(prefix)+'_logits.f16','f2').astype('f4').reshape(K,h,w);ids=np.fromfile(str(prefix)+'_ids.i32','i4').reshape(h,w)
            assert np.isfinite(logits).all();prediction=logits.argmax(0).astype('u1');prediction[ids<0]=K
            target=np.array(Image.open(v['label']).resize((w,h),Image.Resampling.NEAREST));mask=np.array(Image.open(v['supervision_mask']).resize((w,h),Image.Resampling.NEAREST))>0
            cm=confusion(target,prediction,mask);full=confusion(target,prediction);total+=cm;fulltotal+=full
            row.update(semantic=semantic_metrics(cm),secondary_full_consolidated_labels=semantic_metrics(full),valid_GT_fraction=float(mask.mean()))
            target[~mask]=K;colorize(prediction,palette).save(out/(v['name']+'_semantic.png'));colorize(target,palette).save(out/(v['name']+'_GT_semantic.png'));Image.fromarray(prediction).save(out/(v['name']+'_labels.png'))
            features=np.fromfile(str(prefix)+'_features.f16','f2').astype('f4').reshape(32,-1);mapped=decoder@features+bias[:,None];mapped/=np.maximum(np.linalg.norm(mapped,axis=0,keepdims=True),1e-6)
            direct=(text@mapped).reshape(K,h,w).argmax(0).astype('u1');direct[ids<0]=K;qcm=confusion(target,direct,mask);querytotal+=qcm;row['frozen_text_query']=semantic_metrics(qcm)
        rows.append(row);write(out/'partial_metrics.json',rows);print('EVALUATED',name,v['camera_id'],row['PSNR'],row.get('semantic',{}).get('mIoU'),flush=True)
    result={'status':'completed','signature':key,'model':name,'display_model':'+ISLB' if name=='ISLB-Only' else name,'level':'layout' if name in ['L0','L1','L2'] else 'final',
       'test_count':10,'rows':rows,'RGB_mean':{k:float(np.mean([r[k] for r in rows])) for k in ['PSNR','SSIM','LPIPS']},
       'semantic_pooled':semantic_metrics(total) if has_sem else None,'semantic_image_mean':{k:float(np.mean([r['semantic'][k] for r in rows])) for k in ['mIoU','mAcc']} if has_sem else None,
       'secondary_full_consolidated_labels':semantic_metrics(fulltotal) if has_sem else None,'frozen_text_query_pooled':semantic_metrics(querytotal) if has_sem else None,
       'gaussians':read(model/'final/model.json')['gaussians'],'classes':[c['name'] for c in m['classes']['classes']],
       'RGB_resolution':[W,H],'semantic_resolution':[w,h],'semantic_GT':'Only provenance1/2 + real RGB; full8-label secondary reported separately; uncovered predictions are errors',
       'aggregation':'RGB arithmetic mean over10 fixed test ERP images; primary semantic pooled8x9 confusion; no test-based checkpoint selection; uniform ERP pixel weighting',
       'LPIPS_protocol':'AlexNet v0.1, full2048x1024 float RGB scaled to[-1,1], no crop or image downsampling',
       'checkpoint':ref(archive),'inputs':inputs,'render_timings':read(out/'native/render_timings.json'),'completed':time.time()}
    assert len(rows)==10 and all(np.isfinite(r[k]) for r in rows for k in ['PSNR','SSIM','LPIPS'])
    flat=[]
    for r in rows:
        row={k:r[k] for k in ['view','camera_id','PSNR','SSIM','LPIPS']};row.update(mIoU=r['semantic']['mIoU'] if has_sem else None,mAcc=r['semantic']['mAcc'] if has_sem else None)
        row.update({c+'_IoU':r['semantic']['IoU'][i] if has_sem else None for i,c in enumerate(result['classes'])});flat.append(row)
    csvsave(out/'per_view_metrics.csv',flat);write(out/'metrics.json',result);print('METRICS_COMPLETE',name,result['RGB_mean'],result['semantic_pooled'],flush=True);return result

def metric_checks():
    a=np.array([[0,1,7]]);b=np.array([[0,8,7]]);r=semantic_metrics(confusion(a,b));assert abs(r['mIoU']-2/3)<1e-12 and abs(r['mAcc']-2/3)<1e-12 and r['uncovered_errors']==1
    r=semantic_metrics(confusion(a,b,np.array([[True,False,True]])));assert r['mIoU']==r['mAcc']==1
    write(RUN/'04_preflight/metric_checks.json',{'status':'passed','uncovered_counts_as_false_negative':True,'invalid_GT_excluded':True,'macro_absent_classes_excluded':True})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-rois','evaluate','checks']);p.add_argument('--model',choices=MODELS);p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    if a.action=='freeze-rois':freeze_rois()
    elif a.action=='checks':metric_checks()
    else:evaluate(a.model,a.gpu)
