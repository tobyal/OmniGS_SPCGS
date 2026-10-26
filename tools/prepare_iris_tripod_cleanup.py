"""Prepare nadir projections for removal of the camera rig; never start training."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.clip_dense import perspective,FACES
from spcgs.common import ROOT,read,write,ref

RUN=ROOT/'runs/iris_tripod_cleanup_20261025'
SPLIT=ROOT/'runs/iris_camera_analysis_20261025/official_overlap_split_20261025/selected_split.json'
SIZE=1024
FOV=100.

def main():
    RUN.mkdir(exist_ok=False); (RUN/'inputs').mkdir(); rows=[]
    split=read(SPLIT)
    for row in split['selected_cameras']:
        cid=row['camera_id']; path=ROOT.parent/'datasets/iris_full_A_official_2k/images'/(row['image_id']+'.png')
        source=np.asarray(Image.open(path).convert('RGB')); assert source.shape==(1024,2048,3)
        down=perspective(source,dict(FACES)['down'],SIZE,FOV)
        out=RUN/'inputs'/(cid+'.png');Image.fromarray(down).save(out)
        rows.append({**row,'source_rgb':ref(path),'down_projection':ref(out)})
    write(RUN/'manifest.json',{'status':'projected_for_local_edit','split':ref(SPLIT),'images':rows,
       'projection':{'face':'down','size':SIZE,'fov_degrees':FOV,'rotation':dict(FACES)['down'].tolist(),
                     'pixel_centers':True,'erp_horizontal_periodic':True},'training_started':False})
    for start in range(0,len(rows),13):
        subset=rows[start:start+13]; w=5*320;h=((len(subset)+4)//5)*346
        sheet=Image.new('RGB',(w,h),'#f3f5f7');draw=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        for i,row in enumerate(subset):
            x=i%5*320;y=i//5*346
            im=Image.open(row['down_projection']['path']).resize((320,320))
            sheet.paste(im,(x,y+25));draw.text((x+5,y+3),f"{row['camera_id']}  {row['split']}",font=font,fill='#173343')
        sheet.save(RUN/f'nadir_inputs_{start+1}_{start+len(subset)}.jpg',quality=94)
    print('NADIR_PROJECTIONS_READY',len(rows),RUN,flush=True)

if __name__=='__main__':main()
