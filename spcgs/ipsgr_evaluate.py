"""Held-out six-view RGB and seven-class evaluation; uncovered pixels count as errors."""
import os,time,json
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read,write,ref,native,signature
from .pipeline import base_request
from .ipsgr_protocol import check_inputs
from .ipsgr_training import model_folder

def confusion(gt,pred,mask=None):
    valid=(gt>=0)&(gt<7)
    if mask is not None:valid &= mask
    return np.bincount(gt[valid].astype('i8')*8+pred[valid].astype('i8'),minlength=56).reshape(7,8)

def metrics(matrix):
    m=np.asarray(matrix,dtype='i8');tp=np.diag(m[:,:7]);support=m.sum(1);union=support+m[:,:7].sum(0)-tp
    iou=np.divide(tp,union,out=np.zeros(7,float),where=union>0);rec=np.divide(tp,support,out=np.zeros(7,float),where=support>0)
    return {'miou':float(iou[union>0].mean()) if (union>0).any() else None,'macc':float(rec[support>0].mean()) if (support>0).any() else None,'pixel_accuracy':float(tp.sum()/max(m.sum(),1)),'iou':[float(iou[k]) if union[k]>0 else None for k in range(7)],'recall':[float(rec[k]) if support[k]>0 else None for k in range(7)],'support':support.tolist(),'confusion_7x8':m.tolist(),'uncovered_errors':int(m[:,7].sum()),'coverage':float(1-m[:,7].sum()/max(m.sum(),1))}

def softmax(x,axis=0):
    x=np.asarray(x,dtype='f4');z=np.exp(x-x.max(axis=axis,keepdims=True));return z/z.sum(axis=axis,keepdims=True)

def colors(labels,palette):
    p=np.concatenate([palette,np.array([[25,25,25]],dtype='u1')]);return Image.fromarray(p[labels])

def evaluate(run,cfg,name):
    import cv2,torch,lpips
    from skimage.metrics import structural_similarity
    from .ipsgr_analysis import region_evaluation
    m=check_inputs(run,'test');folder=model_folder(run,name);assert read(folder/'outputs.json')['status']=='completed'
    archive=folder/'final/model.pt';assert read(archive.parent/'model.json')['step']==10000
    regions=read(run/'07_industrial_analysis/fixed_regions.json')
    assert regions['frozen_before_first_evaluation'] and not regions['used_predictions']
    dest=run/'06_evaluation'/name;dest.mkdir(parents=True,exist_ok=True);has_sem=name!='B-RGB';H,W=cfg['height'],cfg['width'];h,w=128,256
    inputs={'checkpoint':ref(archive),'dataset':ref(run/'00_dataset/dataset_manifest.json'),'regions':ref(run/'07_industrial_analysis/fixed_regions.json'),'implementation':ref(__file__),'semantic_resolution':[w,h]};key=signature(inputs)
    if (dest/'metrics.json').exists():
        result=read(dest/'metrics.json');assert result['signature']==key;return result
    write(dest/'inputs.json',inputs)
    request={**base_request(cfg),'action':'render','archive':str(archive),'views':m['test'],'output':str(dest/'native'),'render_semantics':has_sem}
    if not all((dest/'native'/(v['name']+'_rgb.f32')).exists() for v in m['test']):native(request,dest/'render_job')
    device=torch.device(f"cuda:{cfg['gpu']}");torch.set_num_threads(4);network=lpips.LPIPS(net='alex',version='0.1').eval().to(device)
    palette=np.array([c['color'] for c in m['classes']['classes']],dtype='u1');total=np.zeros((7,8),dtype='i8');query_total=total.copy();rows=[]
    state=torch.jit.load(str(archive),map_location='cpu') if has_sem else None
    if has_sem:
        decoder=state.decoder.numpy();bias=state.bias.numpy();text=state.text.numpy()
    for v in m['test']:
        prefix=dest/'native'/v['name'];pred=np.fromfile(str(prefix)+'_rgb.f32',dtype='f4').reshape(3,H,W).transpose(1,2,0);gt=np.array(Image.open(v['image']).convert('RGB'),dtype='f4')/255.;assert np.isfinite(pred).all()
        with torch.inference_mode():lp=float(network(torch.from_numpy(pred.transpose(2,0,1).copy())[None].to(device)*2-1,torch.from_numpy(gt.transpose(2,0,1).copy())[None].to(device)*2-1).item())
        row={'id':v['name'],'camera_id':v['camera_id'],'split':'test','psnr':float(-10*np.log10(max(float(np.mean((pred-gt)**2)),1e-12))),'ssim':float(structural_similarity(gt,pred,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),'lpips':lp}
        if has_sem:
            logits=np.fromfile(str(prefix)+'_logits.f16',dtype='f2').astype('f4').reshape(7,h,w);ids=np.fromfile(str(prefix)+'_ids.i32',dtype='i4').reshape(h,w);feature=np.fromfile(str(prefix)+'_features.f16',dtype='f2').astype('f4').reshape(32,h,w)
            assert np.isfinite(logits).all() and np.isfinite(feature).all();prediction=logits.argmax(0).astype('u1');prediction[ids<0]=7;probs=softmax(logits)
            target=np.array(Image.open(v['label']).resize((w,h),Image.Resampling.NEAREST));matrix=confusion(target,prediction);total+=matrix;row['semantic']=metrics(matrix);row['valid_gt_fraction']=float((target<7).mean());row['render_coverage']=float((ids>=0).mean())
            np.save(dest/(v['name']+'_probabilities.npy'),probs);Image.fromarray(prediction).save(dest/(v['name']+'_labels.png'));colors(prediction,palette).save(dest/(v['name']+'_semantic.png'));colors(target,palette).save(dest/(v['name']+'_gt.png'))
            mapped=decoder@feature.reshape(32,-1)+bias[:,None];mapped/=np.maximum(np.linalg.norm(mapped,axis=0,keepdims=True),1e-6);response=(10*text@mapped).reshape(7,h,w);query=response.argmax(0).astype('u1');query[ids<0]=7
            row['direct_text_query']=metrics(confusion(target,query));query_total+=confusion(target,query);np.save(dest/(v['name']+'_query_response.npy'),response);np.save(dest/(v['name']+'_query_probabilities.npy'),softmax(response));Image.fromarray(query).save(dest/(v['name']+'_query_labels.png'));colors(query,palette).save(dest/(v['name']+'_query.png'))
            row['industrial']=region_evaluation(run,name,v,target,prediction,probs,feature,ids,state,regions)
        rows.append(row);write(dest/'partial_metrics.json',rows);print('TEST',name,v['camera_id'],row['psnr'],row.get('semantic',{}).get('miou'),flush=True)
    result={'signature':key,'inputs':inputs,'model':name,'views':rows,'test_count':6,'rgb_mean':{k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim','lpips']},'semantic_pooled':metrics(total) if has_sem else None,'semantic_image_mean':{k:float(np.mean([r['semantic'][k] for r in rows])) for k in ['miou','macc']} if has_sem else None,'direct_text_query_pooled':metrics(query_total) if has_sem else None,'semantic_resolution':[256,128],'classes':[c['name'] for c in m['classes']['classes']],'uncovered_prediction_index':7,'note':'Index7 is an evaluation error bucket, not an eighth model class. Full repaired ERP RGB; valid held-out GT only; uncovered GT errors retained. Classifier and frozen-prototype query reported separately.','lpips':'v0.1 AlexNet, full2048x1024 ERP, float RGB [-1,1], no crop.','render_timings':read(dest/'native/render_timings.json'),'completed':time.time()}
    write(dest/'metrics.json',result);return result

def high_resolution_check(run,cfg,name='IPSGR-Full'):
    m=check_inputs(run,'test');dest=run/'06_evaluation'/name/'semantic_512x256';folder=model_folder(run,name);req={**base_request(cfg),'action':'render','archive':str(folder/'final/model.pt'),'views':m['test'],'output':str(dest),'semantic_width':512,'semantic_height':256,'render_semantics':True};native(req,dest/'job')
    total=np.zeros((7,8),dtype='i8');rows=[];palette=np.array([c['color'] for c in m['classes']['classes']],dtype='u1')
    for v in m['test']:
        scores=np.fromfile(dest/(v['name']+'_logits.f16'),dtype='f2').reshape(7,256,512);ids=np.fromfile(dest/(v['name']+'_ids.i32'),dtype='i4').reshape(256,512);pred=scores.argmax(0).astype('u1');pred[ids<0]=7;gt=np.array(Image.open(v['label']).resize((512,256),Image.Resampling.NEAREST));cm=confusion(gt,pred);total+=cm;rows.append({'id':v['name'],**metrics(cm)});colors(pred,palette).save(dest/(v['name']+'_semantic.png'))
    write(dest/'metrics.json',{'native_render_resolution':[512,256],'not_upsampled':True,'views':rows,'pooled':metrics(total),'primary_ranking_unchanged':True})
