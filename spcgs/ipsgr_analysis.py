"""Predeclared industrial ROIs and visually identified cross-view objects."""
import itertools,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .common import read,write,ref
from .ipsgr_protocol import protocol,MODEL_NAMES

def mask_box(box,h,w):
    x0,y0,x1,y1=box;x0,x1=int(round(x0*w)),int(round(x1*w));y0,y1=int(round(y0*h)),int(round(y1*h));mask=np.zeros((h,w),bool);mask[y0:y1,x0:x1]=True;return mask

def freeze_regions(run):
    target=run/'07_industrial_analysis/fixed_regions.json'
    if target.exists():return read(target)
    assert not list((run/'06_evaluation').glob('*/metrics.json'))
    m=protocol(run);by_id={v['camera_id']:v for v in m['test']}
    definitions={
      '1-26':[('dense_piping',[.08,.02,.56,.34]),('valve_and_equipment',[.29,.27,.53,.65]),('tray_support_occlusion',[.58,.02,.91,.65])],
      '1-27':[('dense_piping',[.06,.02,.55,.42]),('valve_and_equipment',[.30,.44,.50,.79]),('tray_support_occlusion',[.56,.02,.89,.81])],
      '1-60':[('dense_piping',[.36,.01,.72,.47]),('valve_tower_occlusion',[.68,.25,.91,.66]),('tray_support',[.37,.05,.80,.40])],
      '1-61':[('dense_piping',[.34,.02,.79,.48]),('valve_tower_occlusion',[.72,.30,.94,.76]),('equipment_and_piping',[.52,.44,.73,.80])],
      '1-68':[('dense_piping',[.28,.01,.73,.44]),('valve_tower_occlusion',[.72,.23,.98,.68]),('equipment_and_piping',[.44,.36,.70,.77])],
      '1-69':[('dense_piping',[.26,.01,.73,.44]),('valve_tower_occlusion',[.69,.28,.96,.83]),('equipment_and_piping',[.40,.50,.66,.89])]}
    rois=[{'id':cid+'_'+name,'camera_id':cid,'image_id':by_id[cid]['name'],'category':name,'normalized_box':box} for cid,items in definitions.items() for name,box in items]
    objects=[
      {'id':'pump_left_26_27','class_id':1,'identity_basis':'Left black vertical motor on the left insulated pump housing; matching flanges, neighboring pipe bend and red foundation plate.','observations':[('1-26',[.165,.36,.230,.61]),('1-27',[.132,.55,.202,.78])]},
      {'id':'pump_right_26_27','class_id':1,'identity_basis':'Right pump motor beside the vertical insulated riser and red valve line; same paired layout and foundation.','observations':[('1-26',[.350,.32,.458,.59]),('1-27',[.322,.50,.430,.78])]},
      {'id':'control_cabinet_26_27','class_id':1,'identity_basis':'Tall white control cabinet at the right of the tower, with vertically aligned buttons and same flanking metal posts.','observations':[('1-26',[.800,.28,.892,.63]),('1-27',[.765,.43,.845,.75])]},
      {'id':'blue_vessel_60_61_68_69','class_id':1,'identity_basis':'Distinctive blue vertical vessel against the same red wall, behind the low gray cabinet.','observations':[('1-60',[.207,.399,.263,.645]),('1-61',[.225,.530,.280,.742]),('1-68',[.201,.430,.238,.606]),('1-69',[.183,.523,.221,.691])]},
      {'id':'front_cabinet_60_61_68_69','class_id':1,'identity_basis':'Same low light-gray cabinet with blue oval badge and small display, in front of the blue vessel and red wall.','observations':[('1-60',[.291,.460,.418,.777]),('1-61',[.296,.641,.425,.860]),('1-68',[.235,.557,.336,.790]),('1-69',[.216,.624,.300,.820])]},
      {'id':'foreground_pump_68_69','class_id':1,'identity_basis':'Foreground black vertical motor on the curved concrete foundation, beside the same downward insulated elbow.','observations':[('1-68',[.476,.370,.585,.657]),('1-69',[.460,.590,.560,.820])]},
      {'id':'tower_riser_blue_valve','class_id':0,'identity_basis':'Same large insulated vertical tower riser with blue valve and yellow marker, immediately behind the gray safety railing; confirmed by neighboring orange actuator and red column.','observations':[('1-60',[.745,.280,.785,.605]),('1-61',[.762,.400,.795,.688]),('1-68',[.805,.226,.857,.675]),('1-69',[.783,.367,.837,.762])]},
      {'id':'corrugated_hvac','class_id':6,'identity_basis':'Same horizontal corrugated metal ventilation duct across the red wall and blue vessel, with the same elbow and overhead pipe crossings.','observations':[('1-60',[.045,.195,.331,.367]),('1-61',[.032,.345,.350,.483]),('1-68',[.085,.365,.277,.457]),('1-69',[.079,.419,.309,.503])]},
      {'id':'perforated_support_26_27','class_id':3,'identity_basis':'Perforated vertical steel post immediately right of the red column and paired pumps; matching bolted foot and overhead connection.','observations':[('1-26',[.598,.220,.626,.723]),('1-27',[.568,.320,.598,.839])]},
    ]
    for obj in objects:obj['observations']=[{'camera_id':cid,'image_id':by_id[cid]['name'],'normalized_box':box} for cid,box in obj['observations']]
    result={'created':time.time(),'frozen_before_first_evaluation':True,'used_predictions':False,'selection':'Codex visual inspection of captured test RGB and fixed GT before computing any new model test metrics; no label changes.','roi_policy':'Rectangle pixels, all seven existing classes; no invented valve class.','identity_policy':'Small set of visually identified physical objects based on shape and local layout; GT-class intersection within tight rectangles isolates visible object pixels. These are reviewed region correspondences, not dense point matches or SAM-derived identities.','rois':rois,'objects':objects,'test_RGB_and_GT':[{'id':v['name'],'rgb':ref(v['image']),'labels':ref(v['label'])} for v in m['test']]}
    write(target,result)
    for v in m['test']:
        im=Image.open(v['image']).convert('RGB').resize((1024,512));draw=ImageDraw.Draw(im)
        for i,r in enumerate([r for r in rois if r['image_id']==v['name']]):
            b=r['normalized_box'];box=(int(b[0]*1024),int(b[1]*512),int(b[2]*1024),int(b[3]*512));draw.rectangle(box,outline=['#ee6622','#22cc88','#eecc22'][i],width=3);draw.text((box[0]+3,box[1]+3),r['category'],fill='white',stroke_width=1,stroke_fill='black')
        im.save(run/'07_industrial_analysis'/(v['camera_id']+'_fixed_rois.jpg'),quality=95)
    return result

def region_evaluation(run,name,v,gt,pred,probs,feature,ids,state,regions):
    from .ipsgr_evaluate import metrics,confusion,softmax
    h,w=gt.shape;roi_rows=[];objects=[]
    for r in regions['rois']:
        if r['image_id']!=v['name']:continue
        mask=mask_box(r['normalized_box'],h,w);roi_rows.append({'id':r['id'],'category':r['category'],'pixels':int(mask.sum()),**metrics(confusion(gt,pred,mask))})
    for obj in regions['objects']:
        for obs in obj['observations']:
            if obs['image_id']!=v['name']:continue
            mask=mask_box(obs['normalized_box'],h,w)&(gt==obj['class_id']);covered=mask&(ids>=0);n=int(mask.sum());assert n>0,(obj['id'],v['name'])
            dominant=int(np.bincount(pred[mask],minlength=8).argmax());row={'object':obj['id'],'camera_id':v['camera_id'],'target_class':obj['class_id'],'dominant_prediction':dominant,'correct':dominant==obj['class_id'],'pixels':n,'coverage':float(covered.sum()/n),'pixel_accuracy':float((pred[mask]==obj['class_id']).mean())}
            if covered.any():
                chosen=ids[covered].astype('i8');gauss=state.semantic.detach().numpy()[chosen];row.update(mean_feature=feature[:,covered].mean(1).tolist(),mean_probability=probs[:,covered].mean(1).tolist(),mean_gaussian_probability=softmax(gauss,axis=1).mean(0).tolist(),distinct_gaussians=int(len(np.unique(chosen))))
            objects.append(row)
    return {'rois':roi_rows,'objects':objects}

def cross_view_report(run):
    from .ipsgr_evaluate import metrics
    fixed=read(run/'07_industrial_analysis/fixed_regions.json');results={}
    def symkl(a,b):
        a=np.maximum(np.array(a),1e-8);b=np.maximum(np.array(b),1e-8);a/=a.sum();b/=b.sum();return float(.5*((a*np.log(a/b)).sum()+(b*np.log(b/a)).sum()))
    for name in MODEL_NAMES[1:]:
        data=read(run/'06_evaluation'/name/'metrics.json');rows=[o for r in data['views'] for o in r['industrial']['objects']];pairs=[];objects=[]
        for obj in fixed['objects']:
            observed=[o for o in rows if o['object']==obj['id']];assert len(observed)==len(obj['observations'])
            labels=[o['dominant_prediction'] for o in observed];status='correct_consistent' if all(x==obj['class_id'] for x in labels) else 'wrong_consistent' if len(set(labels))==1 and labels[0]<7 else 'inconsistent';objects.append({'object':obj['id'],'status':status,'predictions':labels,'ground_truth_class':obj['class_id'],'observations':observed})
            for a,b in itertools.combinations(observed,2):
                pair={'object':obj['id'],'a':a['camera_id'],'b':b['camera_id'],'correct_consistent':a['correct'] and b['correct'],'wrong_consistent':a['dominant_prediction']==b['dominant_prediction'] and not a['correct'] and a['dominant_prediction']<7,'dominant_consistent':a['dominant_prediction']==b['dominant_prediction'] and a['dominant_prediction']<7,'coverage_a':a['coverage'],'coverage_b':b['coverage']}
                pair['inconsistent']=not(pair['correct_consistent'] or pair['wrong_consistent'])
                if 'mean_feature' in a and 'mean_feature' in b:
                    x=np.array(a['mean_feature']);y=np.array(b['mean_feature']);pair.update(feature_cosine=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-8)),probability_symmetric_KL=symkl(a['mean_probability'],b['mean_probability']),gaussian_symmetric_KL=symkl(a['mean_gaussian_probability'],b['mean_gaussian_probability']))
                pairs.append(pair)
        keys=['correct_consistent','wrong_consistent','inconsistent','dominant_consistent','feature_cosine','probability_symmetric_KL','gaussian_symmetric_KL'];summary={k:float(np.mean([r[k] for r in pairs if k in r])) for k in keys};summary.update(pair_count=len(pairs),object_count=len(objects),continuous_pair_count=sum('feature_cosine' in r for r in pairs));results[name]={'summary':summary,'pairs':pairs,'objects':objects,'identities':ref(run/'07_industrial_analysis/fixed_regions.json')}
    write(run/'07_industrial_analysis/cross_view_results.json',results);return results

def resolution_audit(run):
    """GT sampling/erosion diagnostic; not a change to evaluation or training masks."""
    import cv2
    m=protocol(run);rows=[]
    for v in m['test']:
        im=Image.open(v['label']);full=np.array(im);base=np.bincount(full.ravel(),minlength=7)/full.size
        for w,h in [(256,128),(512,256)]:
            gt=np.array(im.resize((w,h),Image.Resampling.NEAREST))
            for k in range(7):
                mask=gt==k;padded=np.pad(np.pad(mask.astype('u1'),((0,0),(2,2)),mode='wrap'),((2,2),(0,0)),mode='constant');eroded=cv2.erode(padded,np.ones((5,5),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)[2:-2,2:-2]>0
                rows.append({'camera_id':v['camera_id'],'width':w,'height':h,'class_id':k,'class_name':m['classes']['classes'][k]['name'],'full_resolution_fraction':float(base[k]),'sampled_fraction':float(mask.mean()),'sampled_pixels':int(mask.sum()),'GT_class_pixels_after_5x5_erosion':int(eroded.sum()),'retained_fraction':float(eroded.sum()/max(mask.sum(),1))})
    write(run/'07_industrial_analysis/resolution_audit.json',{'rows':rows,'interpretation':'GT-class erosion is a thin-structure diagnostic, not SAM correspondence ground truth. Primary training/evaluation remains256x128;512x256 is an independent native-render check. A256x128 pixel spans8x8 original2048x1024 pixels;5x5 erosion footprint spans40x40 original pixels.'})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5));x=np.arange(7)
    for width,offset,col in [(256,-.18,'#256b9c'),(512,.18,'#d98037')]:
        fractions=[];retention=[]
        for k in range(7):
            rr=[r for r in rows if r['width']==width and r['class_id']==k];fractions.append(np.mean([r['sampled_fraction'] for r in rr])*100);retention.append(sum(r['GT_class_pixels_after_5x5_erosion'] for r in rr)/max(sum(r['sampled_pixels'] for r in rr),1)*100)
        axes[0].bar(x+offset,fractions,.36,color=col,label=f'{width}x{width//2} GT');axes[1].bar(x+offset,retention,.36,color=col,label=f'{width}x{width//2} GT')
    for a in axes:a.set_xticks(x,[c['name'] for c in m['classes']['classes']],rotation=25,ha='right');a.legend();a.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='Class pixels (%)',title='Nearest-neighbor GT sampling');axes[1].set(ylabel='Pixels retained after5x5 erosion (%)',title='Thin-structure diagnostic / GT only');fig.tight_layout();fig.savefig(run/'08_paper_figures/semantic_resolution_diagnostic.png',dpi=220);plt.close(fig)
    return rows
