"""Extract real official-mask examples; visual review is recorded separately."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from prepare_iris_fullclass_gpt import OUT,read,write,save_image,font,decode_rle

def main():
    classes=read(OUT/'taxonomy/classes.json')['classes'];items=[];appearances={c['id']:[] for c in classes}
    for p in sorted((OUT/'images').glob('*/complete.json')):
        folder=p.parent;meta=read(folder/'metadata.json');raw=read(folder/'official_annotations.json');im={x['id']:x for x in raw['images']}
        codemap=dict(zip('FRBLUD',['front','right','back','left','up','down']))
        for fm in meta['faces']:
            for c in fm['classes']:appearances[c['id']].append({'camera_id':folder.name,'split':meta['split'],'face':fm['face'],**c})
        for a in raw['annotations']:
            if a['area']<100:continue
            face=codemap[Path(im[a['image_id']]['file_name']).stem[-1]]
            items.append((folder,face,a))
    refs=[]
    for c in classes:
        cid=c['id'];pool=[r for r in items if r[2]['category_id']==cid]
        # Preserve an explicit easy-to-read pump/blue-vessel example and an electrical cabinet.
        pool.sort(key=lambda r:r[2]['area'],reverse=True)
        selected=[];used=set()
        for r in pool:
            if r[0].name in used:continue
            selected.append(r);used.add(r[0].name)
            if len(selected)==2:break
        examples=[]
        for i,(folder,face,a) in enumerate(selected):
            d=OUT/f'taxonomy/examples/{cid:02d}_{i+1}';d.mkdir(parents=True,exist_ok=True)
            x,y,w,h=a['bbox'];box=[max(0,int(x)-35),max(0,int(y)-35),min(1500,int(x+w)+35),min(1500,int(y+h)+35)]
            if cid==12:
                box=([1000,70,1450,450] if i==0 else [150,150,550,550])
            rgb=Image.open(folder/f'faces/{face}/rgb.jpg');mask=decode_rle(a['segmentation'])
            crop=np.array(rgb.crop(box));m=mask[box[1]:box[3],box[0]:box[2]];over=crop.copy();over[m]=(.5*crop[m]+.5*np.array(c['display_color'])).astype('u1')
            save_image(d/'rgb.jpg',crop);save_image(d/'mask.png',m.astype('u1')*255);save_image(d/'overlay.jpg',over)
            examples.append({'path':str(d.relative_to(OUT)),'camera_id':folder.name,'face':face,'category_id':cid,'annotation_id':a['id'],
              'bbox_face_xyxy':box,'original_coco_bbox_xywh':a['bbox'],'source_pixels':a['area'],'cad_instance_name':None,
              'source':'official COCO RLE','visual_review':'pending','mask_reliability':'official projected silhouette, not human verified'})
        rec={**c,'target_appearances':appearances[cid],'examples':examples,'no_reliable_example':not bool(examples)};refs.append(rec)
        sheet=Image.new('RGB',(1200,1040),'white');draw=ImageDraw.Draw(sheet)
        draw.text((15,5),f'{cid:02d} {c["official_semantic_name"]} | official mask examples',fill='black',font=font(27))
        for i,e in enumerate(examples):
            d=OUT/e['path'];y=i*500+40
            draw.text((15,y),f'{e["camera_id"]} {e["face"]} ann {e["annotation_id"]} | RGB / overlay',fill='black',font=font(21))
            for j,n in enumerate(['rgb.jpg','overlay.jpg']):
                im=Image.open(d/n);scale=min(590/im.width,460/im.height,3);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS);sheet.paste(im,(j*600,y+30))
        save_image(OUT/f'taxonomy/reference_{cid:02d}.jpg',sheet)
    write(OUT/'taxonomy/class_references.json',refs)
    print([(r['id'],len(r['target_appearances']),[(e['camera_id'],e['face']) for e in r['examples']]) for r in refs])

if __name__=='__main__':main()
