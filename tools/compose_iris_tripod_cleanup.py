"""Reproject imagegen-edited nadirs into the original ERP, restricted to rig masks.

This performs geometric resampling/compositing only. The actual object removal
is done by the built-in image generation editor, with its outputs kept on disk.
"""
from pathlib import Path
import argparse
import collections
import copy
import csv
import sys
import shutil
import zipfile
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,ref,sha
from spcgs.clip_dense import perspective,erp_rays,FACES

RUN=ROOT/'runs/iris_tripod_cleanup_20261025'
DATA=ROOT.parent/'datasets/iris_official20_6_no_tripod_2k_20261025'
SPLIT=ROOT/'runs/iris_camera_analysis_20261025/official_overlap_split_20261025'


def composite(row, mx, my, valid):
    cid=row['camera_id'];name=row['image_id']+'.png'
    original=np.array(Image.open(row['source_rgb']['path']).convert('RGB'))
    down=np.array(Image.open(RUN/'inputs'/(cid+'.png')).convert('RGB'))
    generated_file=RUN/'generated'/(cid+'.png')
    im=Image.open(generated_file).convert('RGB'); generated_shape=list(im.size)
    assert abs(im.width/im.height-1)<.01
    generated=np.array(im.resize((1024,1024),Image.Resampling.LANCZOS))
    rig=np.array(Image.open(RUN/'masks'/(cid+'_sam.png')))>0
    # Include the space between the tripod legs and their immediate cast shadows.
    contour=np.column_stack(np.nonzero(rig)[::-1]).astype('i4')
    hull=np.zeros(rig.shape,'u1');cv2.fillConvexPoly(hull,cv2.convexHull(contour),1)
    dilation=5; feather=7
    core=cv2.dilate(hull,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))>0
    dist=cv2.distanceTransform((~core).astype('u1'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    alpha=np.clip(1-dist/feather,0,1).astype('f4')
    # The original bolted column foot in 1-27 lies immediately below the rig.
    # Visual inspection found that the generator altered it: keep source pixels.
    protected=np.zeros(alpha.shape,'u1')
    if cid=='1-27':
        cv2.fillPoly(protected,[np.array([[498,549],[544,549],[552,610],[490,610]],dtype='i4')],1)
        alpha[protected>0]=0
    # Match only a local RGB offset using unedited neighboring floor. No alteration
    # of the source panorama outside the finite-support blend footprint is allowed.
    ring=(dist>=12)&(dist<=28)
    offset=np.median(down[ring].astype('f4')-generated[ring].astype('f4'),axis=0)
    corrected=np.clip(generated.astype('f4')+offset,0,255)
    patch=cv2.remap(corrected,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    erp_alpha=cv2.remap(alpha,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    erp_alpha[~valid]=0
    edit=erp_alpha>0
    output=original.copy()
    output[edit]=np.clip(np.rint(original[edit]*(1-erp_alpha[edit,None])+patch[edit]*erp_alpha[edit,None]),0,255).astype('u1')
    assert np.array_equal(output[~edit],original[~edit])
    assert sha(row['source_rgb']['path'])==row['source_rgb']['sha256']
    # Mask semantics: 255 means reliable original observation; 0 means reconstructed.
    valid_mask=(~edit).astype('u1')*255
    for folder,arr in [('images',output),('valid_masks',valid_mask),('edit_masks',edit.astype('u1')*255)]:
        (DATA/folder).mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(DATA/folder/name)
    split_images=DATA/(row['split']+'_images');split_images.mkdir(exist_ok=True)
    link=split_images/name
    if not link.exists():link.symlink_to(DATA/'images'/name)
    for folder in ['nadir_final','nadir_composites','alpha_down','review/compare','review/original','review/clean','review/valid_masks']:
        (RUN/folder).mkdir(parents=True,exist_ok=True)
    nd=perspective(output,dict(FACES)['down'],1024,100)
    Image.fromarray(nd).save(RUN/'nadir_final'/(cid+'.png'))
    local=np.clip(np.rint(down*(1-alpha[...,None])+corrected*alpha[...,None]),0,255).astype('u1')
    Image.fromarray(local).save(RUN/'nadir_composites'/(cid+'.png'))
    Image.fromarray(np.rint(alpha*255).astype('u1')).save(RUN/'alpha_down'/(cid+'.png'))
    overlay=down.copy();mask=alpha>0;overlay[mask]=(.4*overlay[mask]+.6*np.array([255,0,120])).astype('u1')
    panel=Image.new('RGB',(3*512,544),'#f3f5f7');draw=ImageDraw.Draw(panel)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for i,(arr,label) in enumerate([(down,'Before'),(nd,'After: reprojected from final ERP'),(overlay,'Edited / excluded region')]):
        panel.paste(Image.fromarray(arr).resize((512,512)),(i*512,32));draw.text((i*512+8,5),cid+' | '+label,font=font,fill='#173343')
    panel.save(RUN/'review/compare'/(cid+'.jpg'),quality=95)
    shutil.copy2(row['source_rgb']['path'],RUN/'review/original'/name)
    shutil.copy2(DATA/'images'/name,RUN/'review/clean'/name)
    shutil.copy2(DATA/'valid_masks'/name,RUN/'review/valid_masks'/name)
    changed=np.any(output!=original,axis=2)
    yy=np.where(edit)[0]
    # Check that opaque tripod pixels are fully replaced, including the pole.
    erp_rig=cv2.remap(rig.astype('f4'),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)>.1
    assert np.all(erp_alpha[erp_rig]>.999)
    return {'camera_id':cid,'image_id':row['image_id'],'split':row['split'],'has_official_semantics':True,
        'source_rgb':row['source_rgb'],'clean_rgb':ref(DATA/'images'/name),'valid_mask':ref(DATA/'valid_masks'/name),
        'edit_mask':ref(DATA/'edit_masks'/name),'generated_nadir':ref(generated_file),'generated_shape':generated_shape,
        'rig_mask':ref(RUN/'masks'/(cid+'_sam.png')),'edit_fraction_erp':float(edit.mean()),'changed_pixels':int(changed.sum()),
        'protected_factory_base_polygon':[[498,549],[544,549],[552,610],[490,610]] if cid=='1-27' else None,
        'earliest_edited_erp_row':int(yy.min()),'outside_mask_changed_pixels':int((changed&~edit).sum()),
        'original_unchanged_hash_verified':True,'local_rgb_offset':offset.tolist(),'source_shape':[2048,1024],
        'rig_fully_replaced':True,'semantic_validity_policy':'effective_official_valid = official_valid AND valid_mask; never relabel generated pixels as official floor'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    manifest=read(RUN/'manifest.json');rows=manifest['images'];available=[r for r in rows if (RUN/'generated'/(r['camera_id']+'.png')).exists()]
    if not a.partial:assert len(available)==26,(len(available),'of 26 generated edits present')
    DATA.mkdir(exist_ok=True)
    rays=erp_rays(1024,2048);local=rays@dict(FACES)['down'];z=local[...,2]
    uv=local[...,:2]/np.maximum(z[...,None],1e-8)/np.tan(np.deg2rad(50))
    valid=(z>0)&(np.abs(uv)<=1).all(-1)
    mx=((uv[...,0]+1)*512-.5).astype('f4');my=((uv[...,1]+1)*512-.5).astype('f4')
    records=[]
    for row in available:
        record=composite(row,mx,my,valid);records.append(record)
        print('COMPOSITED',row['camera_id'],f"{record['edit_fraction_erp']:.4%}",'first row',record['earliest_edited_erp_row'],flush=True)
    write(RUN/'cleanup_results.json',{'status':'partial' if len(records)<26 else 'composited_pending_final_visual_QA',
        'method':'100 degree down perspective, imagegen object removal, SAM2 rig convex hull, local offset matching, masked inverse projection',
        'projection':manifest['projection'],'rig_dilation_down_px':5,'feather_down_px':7,'records':records,
        'training_started':False,'official_annotations_modified':False,'official_valid_mask_combination_required':True,
        'training_loss_masks_applied_by_existing_trainer':False,'test_metrics_must_exclude_generated_regions':True,
        'all_originals_preserved':all(r['original_unchanged_hash_verified'] for r in records)})
    if a.partial:return
    split=read(SPLIT/'selected_split.json');(DATA/'openMVG').mkdir(exist_ok=True)
    for role in ['train','test']:
        scene=read(SPLIT/'selected_dataset/openMVG'/f'data_openmvg_{role}.json');scene['root_path']=str(DATA/(role+'_images'))
        write(DATA/'openMVG'/f'data_openmvg_{role}.json',scene)
        expected={r['image_id']+'.png' for r in records if r['split']==role}
        assert {p.name for p in (DATA/(role+'_images')).glob('*.png')}==expected
    # Audit existing initial tracks against the edited pixels before reuse.
    view_names={v['key']:v['value']['ptr_wrapper']['data']['filename'] for v in read(SPLIT/'selected_sfm/sfm_data_train_fixed.json')['views']}
    masks={n:np.array(Image.open(DATA/'edit_masks'/n))>0 for n in view_names.values()}
    touched=[];structure=read(SPLIT/'selected_sfm/structure_filtered.json')['structure']
    for lm in structure:
        for obs in lm['value']['observations']:
            x,y=obs['value']['x'];xx=int(np.clip(round(x),0,2047));yy=int(np.clip(round(y),0,1023))
            if masks[view_names[obs['key']]][yy,xx]:touched.append({'point':lm['key'],'view':obs['key'],'xy':[x,y]})
    if not touched:shutil.copy2(SPLIT/'selected_dataset/openMVG/scene.ply',DATA/'openMVG/scene.ply')
    write(DATA/'dataset_manifest.json',{'status':'cleaned_RGB_with_explicit_validity_masks','source_split':ref(SPLIT/'selected_split.json'),
        'train_count':20,'test_count':6,'validation_count':0,'records':records,'resolution':[2048,1024],
        'coordinate_transform':split['coordinate_transform'],'cameras_unchanged':True,
        'initial_cloud_reused':not touched,'initial_cloud_observations_on_edited_pixels':touched,
        'valid_mask_values':{'0':'edited/generated region: exclude from RGB/semantic losses and test metrics','255':'original observed RGB'},
        'trainer_mask_integration_pending':True,'no_training_started':True,'no_test_to_train_content_transfer':True,
        'official_labels_unchanged':True,'official_semantic_ignore':'combine the official valid mask with this RGB valid mask; do not overwrite official source labels'})
    shutil.copy2(DATA/'dataset_manifest.json',RUN/'review/dataset_manifest.json')
    shutil.copy2(RUN/'cleanup_results.json',RUN/'review/cleanup_results.json')
    shutil.copy2(SPLIT/'selected_cameras.csv',RUN/'review/selected_cameras.csv')
    print('COMPLETE',len(records),'initial tracks touching edits',len(touched),DATA,flush=True)

if __name__=='__main__':main()
