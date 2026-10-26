"""Measured IPSGR paper artifacts. No placeholder numbers or assumed improvements."""
import csv,html,json,re,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import ROOT,read,write,ref
from .ipsgr_protocol import protocol,MODEL_NAMES
from .ipsgr_training import model_folder

def table(run,key,rows,extra=None):
    dest=run/'09_paper_tables';write(dest/(key+'.json'),{'rows':rows,**(extra or {})})
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (dest/(key+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    registry=read(dest/'registry.json');registry[key]={'status':'completed','script':ref(__file__),'raw_data':[str(dest/(key+'.json'))]};write(dest/'registry.json',registry)

def register(run,key,paths,sources,note=''):
    dest=run/'08_paper_figures';registry=read(dest/'registry.json');registry[key]={'status':'completed','figures':[str(p) for p in paths],'raw_data':[str(p) for p in sources],'script':ref(__file__),'note':note};write(dest/'registry.json',registry)

def panel(path,items,columns=3,tw=640,th=320):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);canvas=Image.new('RGB',(columns*tw,((len(items)+columns-1)//columns)*(th+30)),'white');d=ImageDraw.Draw(canvas)
    for i,(title,im,semantic) in enumerate(items):
        if isinstance(im,(str,Path)):im=Image.open(im).convert('RGB')
        x=i%columns*tw;y=i//columns*(th+30);label_font=font
        while d.textlength(title,font=label_font)>tw-10 and label_font.size>9:label_font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',label_font.size-1)
        d.text((x+5,y+4),title,fill='black',font=label_font)
        scale=min(tw/im.width,th/im.height);size=(max(1,round(im.width*scale)),max(1,round(im.height*scale)))
        resized=im.resize(size,Image.Resampling.NEAREST if semantic else Image.Resampling.LANCZOS)
        canvas.paste(resized,(x+(tw-size[0])//2,y+30+(th-size[1])//2))
    canvas.save(path);return path

def semantic_legend(path,m):
    im=Image.open(path).convert('RGB');canvas=Image.new('RGB',(im.width,im.height+46),'white');canvas.paste(im,(0,0));d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for k,c in enumerate(m['classes']['classes']):
        x=round(k*im.width/7)+8;y=im.height+12;d.rectangle((x,y,x+18,y+18),fill=tuple(c['color']));d.text((x+24,y+1),c['name'],fill='black',font=font)
    canvas.save(path)

def prose_html(lines):
    """Small renderer for this report's generated headings, tables and code."""
    def inline(text):
        text=html.escape(text)
        text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
        return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',text)
    out=[];in_table=False;in_code=False;in_list=False
    for line in lines:
        if line.startswith('```'):
            out.append('</code></pre>' if in_code else '<pre><code>');in_code=not in_code;continue
        if in_code:out.append(html.escape(line)+'\n');continue
        if in_table and not line.startswith('|'):out.append('</tbody></table></div>');in_table=False
        if in_list and not line.startswith('- '):out.append('</ul>');in_list=False
        if not line:continue
        if line.startswith('|'):
            cells=line.strip('|').split('|')
            if all(set(c.strip())<=set('-:') for c in cells):continue
            if not in_table:out.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cells)+'</tr></thead><tbody>');in_table=True
            else:out.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
        elif line.startswith('# '):out.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):out.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('- '):
            if not in_list:out.append('<ul>');in_list=True
            out.append('<li>'+inline(line[2:])+'</li>')
        else:out.append('<p>'+inline(line)+'</p>')
    if in_table:out.append('</tbody></table></div>')
    if in_list:out.append('</ul>')
    return ''.join(out)

def diagrams(run):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    dest=run/'08_paper_figures';fig,ax=plt.subplots(figsize=(16,6));ax.set(xlim=(0,16),ylim=(0,6));ax.axis('off')
    boxes=[(.2,3.5,2.5,1.4,'20 repaired ERP RGB\n+ fixed cameras\n+ seven-class labels','#e6f0f8'),(3.2,3.5,3.0,1.4,'ISLB geometry\nFresh P0 + Wan360\nregistered additions -> P1','#e6f0f8'),(6.7,3.5,2.7,1.4,'ISLB / 10k steps\nRGB + semantic layout\nDensify / prune / OGR','#e6f0f8'),(9.9,3.5,2.7,1.4,'Stage reset\nKeep xyz / DC / semantics\nReset shape / opacity / SH','#f5eddc'),(9.9,.8,2.7,1.7,'ICSC / 10k steps\n2 online pseudo views\nSAM2 Hiera-L\n2D inter/intra + 3D KL','#e5f3e8'),(13.1,.8,2.7,1.7,'6 held-out test views\nRGB + semantic classes\nText-prototype queries\nROI and object analysis','#f5eddc'),(3.2,.8,6.2,1.7,'Fresh CLIP class-text prototypes T\nLabel-derived F / softmax S / valid W\n32D Gaussian semantics -> 512D mapping\nTrain labels only; no generated semantic/RGB targets','#e5f3e8')]
    for x,y,w,h,t,col in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=col,edgecolor='#32475b'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
    for a,b in [((2.7,4.2),(3.15,4.2)),((6.2,4.2),(6.65,4.2)),((9.4,4.2),(9.85,4.2)),((11.25,3.5),(11.25,2.55)),((12.6,1.65),(13.05,1.65)),((6.1,2.5),(7.9,3.45)),((9.4,1.65),(9.85,1.65))]:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=18,color='#32475b',lw=1.7))
    ax.text(.2,5.6,'IPSGR: Industrial Panoramic Semantic Gaussian Reconstruction',fontsize=18,weight='bold');fig.tight_layout();paths=[]
    for ext in ['png','pdf','svg']:
        p=dest/f'F02_method.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight');paths.append(p)
    plt.close(fig);register(run,'F02',paths,[run/'experiment_config.json',run/'00_dataset/dataset_manifest.json'],'Diagram describes the implemented industrial adaptation, not an identical reproduction of every SPC-GS setting.')

def comparisons(run,m,results):
    from .ipsgr_evaluate import colors
    dest=run/'08_paper_figures';palette=np.array([c['color'] for c in m['classes']['classes']],dtype='u1');paths=[]
    for v in m['test']:
        gt=Image.open(v['label']);gt=colors(np.array(gt),palette);items=[('Captured repaired RGB / '+v['camera_id'],v['image'],False)]
        items += [(name,run/'06_evaluation'/name/'native'/(v['name']+'_rgb.png'),False) for name in MODEL_NAMES]
        items += [('Fixed seven-class GT',gt,True),('B-RGB / semantic not trained',Image.new('RGB',(1024,512),'#eeeeee'),True)]
        items += [(name+' / classifier',run/'06_evaluation'/name/(v['name']+'_semantic.png'),True) for name in MODEL_NAMES[1:]]
        path=dest/('F05_'+v['camera_id']+'.png');panel(path,items,columns=7,tw=512,th=256);semantic_legend(path,m);paths.append(path)
    register(run,'F05',paths,[run/'06_evaluation'/n/'metrics.json' for n in MODEL_NAMES],'Every held-out view; nearest resizing for display only; primary native semantics256x128.')
    v=next(v for v in m['test'] if v['camera_id']=='1-68');items=[('Held-out RGB / 1-68',v['image'],False),('B-RGB: fresh P0 / 10k',run/'06_evaluation/B-RGB/native'/(v['name']+'_rgb.png'),False),('B-SEM: fresh P0 / 10k',run/'06_evaluation/B-SEM'/(v['name']+'_semantic.png'),True),('Seven-class GT',colors(np.array(Image.open(v['label'])),palette),True)]
    p=panel(dest/'F01_sparse_problem.png',items,2);register(run,'F01',[p],[run/'06_evaluation/B-RGB/metrics.json',run/'06_evaluation/B-SEM/metrics.json'],'Fixed view1-68 chosen before results; visually inspect actual strengths and errors without assuming baseline failure.')
    fixed=read(run/'07_industrial_analysis/fixed_regions.json');roi_paths=[]
    for r in fixed['rois']:
        v=next(v for v in m['test'] if v['name']==r['image_id']);b=r['normalized_box'];rgb=Image.open(v['image']);gt=colors(np.array(Image.open(v['label'])),palette)
        def crop(im):return im.crop(tuple(int(round(b[i]*im.size[i%2])) for i in range(4)))
        items=[(r['id']+' / RGB',crop(rgb),False),('GT / seven existing classes',crop(gt),True)]
        for name in MODEL_NAMES[1:]:items.append((name,crop(Image.open(run/'06_evaluation'/name/(v['name']+'_semantic.png'))),True))
        path=panel(dest/('F07_'+r['id']+'.png'),items,7,tw=360,th=280);semantic_legend(path,m);roi_paths.append(path)
        # Separate RGB comparisons preserve the same fixed crop.
        panel(dest/('F07_RGB_'+r['id']+'.png'),[(n,crop(Image.open(run/'06_evaluation'/n/'native'/(v['name']+'_rgb.png'))),False) for n in MODEL_NAMES],6,tw=360,th=280)
    register(run,'F07',roi_paths,[run/'07_industrial_analysis/fixed_regions.json',run/'09_paper_tables/T04.json'],'ROI location is fixed from RGB/GT; displayed coarse semantic pixels are not claimed as new detail.')

def geometry(run,m):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .common import ply
    p0=np.fromfile(run/'01_geometry/P0/points.f32',dtype='f4').reshape(-1,6);p1=np.fromfile(run/'01_geometry/P1/enhanced_points.f32',dtype='f4').reshape(-1,6);xyz,rgb,_=ply(run/'03_layout_bootstrap/final/point_cloud.ply');new=p1[len(p0):,:3]
    fig,axes=plt.subplots(2,3,figsize=(16,10));cams=np.array([v['center'] for v in m['train']]);rng=np.random.default_rng(42)
    for col,(name,pts,cols) in enumerate([('P0',p0[:,:3],p0[:,3:]),('P1',p1[:,:3],p1[:,3:]),('ISLB10k',xyz,rgb)]):
        ids=rng.choice(len(pts),min(len(pts),60000),replace=False)
        for row,(a,b) in enumerate([(0,1),(0,2)]):
            ax=axes[row,col];ax.scatter(pts[ids,a],pts[ids,b],c=cols[ids],s=.4,rasterized=True);ax.scatter(cams[:,a],cams[:,b],c='black',s=12,marker='^');ax.set_aspect('equal');ax.set_title(f'{name}: {len(pts):,} points / Gaussians');ax.set(xlabel=['X','Y','Z'][a],ylabel=['X','Y','Z'][b]);lo=np.quantile(p0[:,:3],[.005,.995],axis=0);ax.set_xlim(lo[0,a]-.3,lo[1,a]+.3);ax.set_ylim(lo[0,b]-.3,lo[1,b]+.3)
            if col==1 and len(new):ax.scatter(new[:,a],new[:,b],s=4,c='red',label=f'new: {len(new)}');ax.legend()
    fig.tight_layout();dest=run/'08_paper_figures';paths=[]
    for ext in ['png','pdf']:
        path=dest/f'F06_geometry.{ext}';fig.savefig(path,dpi=240);paths.append(path)
    plt.close(fig);register(run,'F06',paths,[run/'01_geometry/P0/metrics.json',run/'01_geometry/P1/metrics.json',run/'03_layout_bootstrap/final/model.pt'],'All original data retained; viewport uses fixed P0 .5-99.5 percentiles, ISLB plotted with a fixed random sample up to60000. Projection previews are not visibility guarantees.')

def isp_figures(run):
    from .spc27_evaluate import contact_sheet,contributor_plot,saved_isp_check
    folder=run/'05_models/IPSGR-Full';loss=[json.loads(x) for x in (folder/'losses.jsonl').read_text().splitlines()];events=[r for r in loss if r.get('isp_event')];assert events
    success=max(events,key=lambda r:(r['isp_valid_pairs'],r['spc3d_semantic_grad_norm']>0,-r['step']));failure=min(events,key=lambda r:(r['isp_valid_pairs'],r['spc3d_semantic_grad_norm']>0,r['step']));dest=run/'08_paper_figures';paths=[];audits=[]
    for label,row in [('success',success),('lowest_validity',failure)]:
        event=folder/'isp'/str(row['step']);p=dest/f'F09_{label}_{row["step"]}.png';contact_sheet(event,p,f'IPSGR-Full step{row["step"]}');paths.append(p);audits.append({'step':row['step'],'valid_pairs':row['isp_valid_pairs'],**saved_isp_check(event,row['inter'])})
    event=folder/'isp'/str(failure['step']);status=read(event/'outputs.json');masks=np.fromfile(event/'masks.u8',dtype='u1').reshape(3,128,256);cx,cy=status['prompt_xy'];items=[]
    for i,label in enumerate(['Captured training RGB','Online pseudo 0','Online pseudo 1']):
        im=Image.open(event/f'{i}.png').convert('RGB');w,h=im.size;left=int(np.clip(cx-256,0,max(w-512,0)));top=int(np.clip(cy-128,0,max(h-256,0)));box=(left,top,min(left+512,w),min(top+256,h))
        items.append((label,im.crop(box),False));mask=np.array(Image.fromarray(masks[i]*255).resize(im.size,Image.Resampling.NEAREST))>0;arr=np.array(im);arr[mask]=(arr[mask]*.4+np.array([240,40,40])*.6).astype('u1');overlay=Image.fromarray(arr)
        if i==0:
            d=ImageDraw.Draw(overlay);d.ellipse((cx-5,cy-5,cx+5,cy+5),fill='yellow')
        items.append((f'{int((masks[i]>0).sum())} native mask pixels / '+('source' if i==0 else str(status['valid'][i-1])),overlay.crop(box),False))
    zoom=dest/f'F09_lowest_validity_zoom_{failure["step"]}.png';panel(zoom,[items[i] for i in [0,2,4,1,3,5]],3,512,256);paths.append(zoom)
    p=dest/'F03_dual_domain_actual.png';contributor_plot(folder/'isp'/str(success['step']),p);write(run/'04_cross_view/saved_event_validation.json',audits)
    register(run,'F03',[p],[folder/'isp'/str(success['step']),folder/'losses.jsonl'],'Actual maximum alpha*T IDs/weights; IDs belong to this event state, not the final densified Gaussian ordering.')
    register(run,'F09',paths,[folder/'isp',run/'04_cross_view/saved_event_validation.json'],'Mask success is a geometric/area proxy, not verified correspondence accuracy. Lowest-validity event is not called a failure if no rejected pairs occurred.')

def cross_figures(run,m,cross):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fixed=read(run/'07_industrial_analysis/fixed_regions.json');dest=run/'08_paper_figures';names=MODEL_NAMES[1:];objects=[o['id'] for o in fixed['objects']];code={'correct_consistent':2,'wrong_consistent':1,'inconsistent':0};data=np.array([[code[o['status']] for o in cross[n]['objects']] for n in names]);fig,ax=plt.subplots(figsize=(14,5));from matplotlib.colors import ListedColormap
    ax.imshow(data,cmap=ListedColormap(['#b7bec9','#e5a34a','#64ad8b']),vmin=0,vmax=2,aspect='auto');ax.set_yticks(range(5),names);ax.set_xticks(range(len(objects)),objects,rotation=28,ha='right');ax.set_title('Verified-region outcomes: C=correct consistent; W=wrong consistent; I=inconsistent')
    for y in range(5):
        for x in range(len(objects)):ax.text(x,y,['I','W','C'][data[y,x]],ha='center',va='center',weight='bold')
    fig.tight_layout();paths=[]
    for ext in ['png','pdf']:p=dest/f'F08_consistency.{ext}';fig.savefig(p,dpi=220);paths.append(p)
    plt.close(fig)
    for obj in fixed['objects']:
        items=[]
        for obs in obj['observations']:
            v=next(v for v in m['test'] if v['name']==obs['image_id']);box=obs['normalized_box']
            for label,path,sem in [('RGB',v['image'],False),('GT',run/'06_evaluation/IPSGR-Full'/(v['name']+'_gt.png'),True),*[(n,run/'06_evaluation'/n/(v['name']+'_semantic.png'),True) for n in names]]:
                im=Image.open(path);crop=im.crop(tuple(int(round(box[i]*im.size[i%2])) for i in range(4)));items.append((obs['camera_id']+' / '+label,crop,sem))
        p=panel(dest/('F08_object_'+obj['id']+'.png'),items,7,tw=250,th=250);paths.append(p)
    register(run,'F08',paths,[run/'07_industrial_analysis/cross_view_results.json',run/'07_industrial_analysis/fixed_regions.json'],'Small visually identified object/region set; no dense instance correspondence ground truth. Missing correctness categories are not fabricated.')

def query_figures(run,m):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=[];v=next(v for v in m['test'] if v['camera_id']=='1-68');base=run/'06_evaluation/IPSGR-Full'
    scores=np.load(base/(v['name']+'_query_response.npy'));relative=np.load(base/(v['name']+'_query_probabilities.npy'))
    ids=np.fromfile(base/'native'/(v['name']+'_ids.i32'),dtype='i4').reshape(128,256)
    for values,low,high,filename,label in [(scores,float(scores.min()),float(scores.max()),'F11_class_queries.png','Frozen text correlation / shared scale'),(relative,0.,1.,'F11_relative_queries.png','Fixed-seven Softmax response / not calibrated confidence')]:
        fig,axes=plt.subplots(4,2,figsize=(16,14));axes[0,0].imshow(Image.open(v['image']));axes[0,0].set_title('Held-out captured RGB /1-68');palette=plt.get_cmap('viridis').copy();palette.set_bad('#eeeeee')
        for k,ax in enumerate(axes.flat[1:]):
            im=ax.imshow(np.ma.array(values[k],mask=ids<0),vmin=low,vmax=high,cmap=palette);ax.set_title(m['classes']['classes'][k]['name'])
        for ax in axes.flat:ax.axis('off')
        fig.colorbar(im,ax=list(axes.flat),shrink=.5,label=label);p=run/'08_paper_figures'/filename;fig.savefig(p,dpi=200,bbox_inches='tight');plt.close(fig);paths.append(p)
    register(run,'F11',paths,[base/(v['name']+'_query_response.npy'),base/(v['name']+'_query_probabilities.npy'),base/'metrics.json'],'Seven fixed classes queried through the trained mapping. Raw correlations and fixed-seven Softmax responses are separate displays; neither is unrestricted open vocabulary or calibrated confidence.')

def point_support(run,m):
    """Analysis only: class votes at measured real observations; never a geometry filter."""
    cameras={v['name']:v for v in m['train']};labels={n:np.array(Image.open(v['label'])) for n,v in cameras.items()};observed={n:np.array(Image.open(v['rgb_observation_mask']['path']))>0 for n,v in cameras.items()};summaries={}
    for name,path,obskey in [('P0',run/'01_geometry/P0/observations.json','observations'),('P1_added',run/'01_geometry/P1/observations.json','real_observations')]:
        rows=[]
        for i,p in enumerate(read(path)):
            votes=[];original=[]
            for o in p[obskey]:
                u,y=o['uv'];u=int(round(u))%2048;y=int(np.clip(round(y),0,1023));votes.append(int(labels[o['view']][y,u]));original.append(bool(observed[o['view']][y,u]))
            hist=np.bincount(votes,minlength=7);rows.append({'point_index':i,'x':p['xyz'][0],'y':p['xyz'][1],'z':p['xyz'][2],'majority_class':int(hist.argmax()),'class_agreement':float(hist.max()/len(votes)),'real_observations':len(votes),'original_RGB_observations':sum(original),'synthetic_RGB_observations':len(original)-sum(original)})
        out=run/'07_industrial_analysis'/(name+'_point_support.json');write(out,rows);summaries[name]={'points':len(rows),'class_counts':np.bincount([r['majority_class'] for r in rows],minlength=7).tolist(),'any_synthetic_RGB_observation':sum(r['synthetic_RGB_observations']>0 for r in rows),'all_synthetic_RGB_observations':sum(r['original_RGB_observations']==0 for r in rows),'mean_semantic_vote_agreement':float(np.mean([r['class_agreement'] for r in rows])),'evidence':ref(out)}
    write(run/'07_industrial_analysis/geometry_semantic_support.json',{'data':summaries,'policy':'Only measured real feature observations are sampled. Derived class votes are analysis proxies, not new labels, visibility guarantees or geometric acceptance rules.'})

def training_figures(run):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=[];cfg=read(run/'experiment_config.json');reset=int(cfg['opacity_reset_every']);until=int(cfg['densify_until'])
    for name in ['ISLB',*MODEL_NAMES]:
        rows=[json.loads(x) for x in (model_folder(run,name)/'losses.jsonl').read_text().splitlines()];fig,ax=plt.subplots(2,3,figsize=(16,8));step=np.array([r['step'] for r in rows]);groups=[(['rgb','clip','local3d'],'Base losses'),(['inter','intra','spc3d'],'ICSC losses at ISP events'),(['semantic_grad_norm','xyz_grad_norm'],'Actual gradient norms'),(['gaussians'],'Gaussian count'),(['added_cumulative','pruned_cumulative'],'Densification and pruning'),(['inter_semantic_grad_norm','intra_semantic_grad_norm','spc3d_semantic_grad_norm'],'ICSC gradients at ISP events')]
        for a,(keys,title) in zip(ax.flat,groups):
            for k in keys:
                rr=[r for r in rows if k in r and (r.get('isp_event') if 'ICSC' in title else True)];x=[r['step'] for r in rr];y=[r[k] for r in rr]
                if len(rr)>500:
                    ix=np.arange(0,len(rr),50);x=[x[i] for i in ix];y=[float(np.mean(y[i:i+50])) for i in ix]
                a.plot(x,y,label=k,lw=1)
            if reset>0:
                for at in range(reset,min(until,int(step[-1])+1),reset):a.axvline(at,color='#768595',ls='--',lw=.7,alpha=.5)
            a.set_title(title);a.set_xlabel('Step');a.grid(alpha=.2);a.legend(fontsize=7)
        fig.suptitle(name+' / dashed lines: configured opacity resets');fig.tight_layout();p=run/'08_paper_figures'/('training_'+name+'.png');fig.savefig(p,dpi=180);plt.close(fig);paths.append(p)
    write(run/'10_report/training_figure_evidence.json',{'plots':[str(p) for p in paths],'data':[str(model_folder(run,n)/'losses.jsonl') for n in ['ISLB',*MODEL_NAMES]],'script':ref(__file__)})
    register(run,'S_training_history',paths,[model_folder(run,n)/'losses.jsonl' for n in ['ISLB',*MODEL_NAMES]],'Actual training losses, gradients and Gaussian counts. Dashed lines mark configured opacity resets; these are not held-out evaluation curves.')

def report(run):
    from .ipsgr_analysis import cross_view_report
    from .ipsgr_evaluate import metrics
    from .ipsgr_supplement import layout_previews,report_details,measured_geometry_projection,resource_figure
    from .ipsgr_audit import fresh_generation,final_models,source_snapshot
    from .ipsgr_viewer import build_viewer
    fresh_generation(run);final_models(run);source_snapshot(run)
    m=protocol(run);results={n:read(run/'06_evaluation'/n/'metrics.json') for n in MODEL_NAMES};cross=cross_view_report(run);p0=read(run/'01_geometry/P0/metrics.json');p1=read(run/'01_geometry/P1/metrics.json');summary=[];cost=[];roi=[];classes=[c['name'] for c in m['classes']['classes']]
    for name,r in results.items():
        row={'model':name,**r['rgb_mean'],'mIoU':r['semantic_pooled']['miou'] if r['semantic_pooled'] else None,'mAcc':r['semantic_pooled']['macc'] if r['semantic_pooled'] else None,'coverage':r['semantic_pooled']['coverage'] if r['semantic_pooled'] else None,'direct_text_query_mIoU':r['direct_text_query_pooled']['miou'] if r['direct_text_query_pooled'] else None}
        if r['semantic_pooled']:row.update({c+'_IoU':r['semantic_pooled']['iou'][k] for k,c in enumerate(classes)})
        summary.append(row);folder=model_folder(run,name);out=read(folder/'outputs.json');times=r['render_timings'];native_archive=folder/'final/model.pt';inference_files=[folder/'final'/f for f in ['point_cloud.ply','semantic.f32','decoder.f32','decoder_bias.f32','text.f32']]
        cost.append({'model':name,'gaussians':out['gaussians'],'training_seconds_last_invocation':out['seconds_this_invocation'],'ISLB_extra_seconds':read(run/'03_layout_bootstrap/outputs.json')['seconds_this_invocation'] if name in ['L-BOOT','CV-2D','IPSGR-Full'] else 0,'archive_bytes_with_optimizer':native_archive.stat().st_size,'PLY_and_exported_sidecar_bytes':sum(p.stat().st_size for p in inference_files),'mean_RGB_render_seconds':float(np.mean([t['rgb_seconds'] for t in times])),'mean_semantic_feature_render_seconds':float(np.mean([t['semantic_seconds'] for t in times])),'resumed':out['resumed']})
        if name!='B-RGB':
            for v in r['views']:
                for x in v['industrial']['rois']:
                    roi.append({'model':name,'camera_id':v['camera_id'],'ROI':x['id'],'category':x['category'],'mIoU':x['miou'],'mAcc':x['macc'],'coverage':x['coverage'],**{c+'_IoU':x['iou'][k] for k,c in enumerate(classes)}})
    table(run,'T02',summary,{'definition':'Pooled held-out7-class mIoU; mean RGB across6 views. No semantic score for B-RGB.'})
    deltas=[];by={r['model']:r for r in summary}
    for a,b,module in [('B-SEM','G-AUG','Wan360 geometry'),('G-AUG','L-BOOT','ISLB preoptimization'),('L-BOOT','CV-2D','ICSC image domain'),('CV-2D','IPSGR-Full','ICSC Gaussian domain')]:deltas.append({'before':a,'after':b,'module':module,**{k:by[b][k]-by[a][k] for k in ['psnr','ssim','lpips','mIoU','mAcc']}})
    table(run,'T03',deltas,{'P0':p0,'P1':p1,'training_budget':'All core final stages10k; ISLB10k extra shared initialization, separately costed.'});table(run,'T04',roi);table(run,'T05',[{'model':name,**c['summary']} for name,c in cross.items()],{'identities':ref(run/'07_industrial_analysis/fixed_regions.json')});table(run,'T06',cost,{'timing_environment':'Shared A6000 GPUs with concurrent external jobs; operational timings, not isolated throughput benchmark. Semantic timing is feature rasterization, excludes classification head/export.','ISLB':read(run/'03_layout_bootstrap/outputs.json'),'Wan360':read(run/'01_geometry/Wan360/parallel_completed.json')})
    diagrams(run);comparisons(run,m,results);geometry(run,m);isp_figures(run);cross_figures(run,m,cross);query_figures(run,m);point_support(run,m);training_figures(run)
    layout_figure=layout_previews(run);measured_figure=measured_geometry_projection(run)
    reg=read(run/'08_paper_figures/registry.json');reg['F06']['figures'] += [str(layout_figure),str(measured_figure)];reg['F06']['raw_data'].append(str(run/'03_layout_bootstrap/representative_train_renders/evidence.json'));reg['F07']['figures'] += [str(p) for p in sorted((run/'08_paper_figures').glob('F07_RGB_*.png'))];write(run/'08_paper_figures/registry.json',reg)
    detailed_lines=report_details(run,results,p1,roi,cost);resource_figure(run);viewer=build_viewer(run)
    register(run,'S_semantic_resolution',[run/'08_paper_figures/semantic_resolution_diagnostic.png'],[run/'07_industrial_analysis/resolution_audit.json'],'Fixed-GT sampling and erosion diagnostic; not SAM correspondence ground truth and not a change to the evaluation resolution.')
    f=run/'08_paper_figures/registry.json';reg=read(f);reg['F11']['interactive_viewer']=str(viewer);reg['F11']['raw_data'].append(str(run/'10_report/viewer_manifest.json'));reg['F10']={'status':'not_run_optional','reason':'Core six-model20-view experiment prioritized; no12-view experiment has been claimed.'};write(f,reg)
    evidence={'dataset':ref(run/'00_dataset/dataset_manifest.json'),'P0':ref(run/'01_geometry/P0/outputs.json'),'P1':ref(run/'01_geometry/P1/outputs.json'),'teacher':ref(run/'02_semantic_teacher/cache/manifest.json'),'models':{n:read(model_folder(run,n)/'outputs.json') for n in MODEL_NAMES},'model_inputs':{n:{'inputs':ref(model_folder(run,n)/'inputs.json'),'configuration':ref(model_folder(run,n)/'fixed_request.json')} for n in MODEL_NAMES},'initialization':{n:read(model_folder(run,n)/'initialization_validation.json') for n in ['L-BOOT','CV-2D','IPSGR-Full']},'training':{n:read(model_folder(run,n)/'training_audit.json') for n in MODEL_NAMES},'figures':read(f),'tables':read(run/'09_paper_tables/registry.json')};write(run/'10_report/completion_audit.json',evidence)
    full=by['IPSGR-Full'];base=by['B-RGB'];lines=['# IPSGR 修复后 IRIS20/6 七类实验报告','', '## 数据与协议',f'本轮冻结20张训练、6张测试、不设验证集；全部使用修复后的2048×1024 RGB与v4七类标签。数据签名：`{m["dataset_signature"]}`。本轮未重新修图或改标签。26张标签均为0–6，无ignore类别。', '标签包含官方投影、人工指令下的类别合并与底部修复衍生标签，不能称为纯官方逐像素真值。原始观测与合成区域来源单独保存。先前选择插值测试视角时参考过图像重叠；本轮训练开始前固定划分，属于可行性实验，不是盲选困难外推benchmark。','', '## 实际执行',f'重新完成20张训练图的RootSIFT、190对匹配和固定已知位姿球面三角化。P0={p0["points"]}点，观测球面重投影中位数{p0["reprojection_spherical_px_2k"]["median"]:.4f}像素。80个Wan360视频全部为本轮修复RGB重新生成，四方向各20个；旧视频、旧SfM、旧教师、旧高斯模型均未复用。',f'P1={p1["enhanced_count"]}点，新增{p1["added"]}点；有效选定视角{p1["selected_views"]}，新增点最大观测重投影误差中位数{p1["reprojection_px"]["median"]:.4f}像素。方向仅用于生成，不用于方向一致性硬筛选。','本轮CLIP监督是七类标签驱动的冻结文本原型F/S/T/W，不是完整开放词汇图像CLIP蒸馏。仅20张训练标签进入教师。Wan360帧仅提供几何，不参与RGB或伪语义监督。','ISLB实际训练10k步。L-BOOT、CV-2D、IPSGR-Full独立从同一ISLB开始；逐张量验证保留xyz、DC、语义及映射参数，重置尺度、旋转、透明度、高阶SH、优化器和增密统计。核心六组最终训练均为10k。','ICSC使用原始SAM2 Hiera-L，当前高斯在线渲染两个邻近伪视角，每100步一次ISP，权重前4000步递增。保留5×5 ERP循环腐蚀、真实alpha×T最大贡献ID及高斯语义KL。ISP频率和工业标签教师是明确的工程适配。','', '## 六张独立测试视角的结果','', '|模型|PSNR↑|SSIM↑|LPIPS↓|七类mIoU↑|mAcc↑|直接文本查询mIoU↑|','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda x:'—' if x is None else f'{x:.4f}'
        lines.append('|'+r['model']+'|'+'|'.join(fmt(r[k]) for k in ['psnr','ssim','lpips','mIoU','mAcc','direct_text_query_mIoU'])+'|')
    lines += ['', 'RGB指标为六张测试图均值，主语义mIoU为六图累计混淆矩阵；逐图mIoU和图均值同时保留。有效GT而无高斯贡献的像素进入7×8矩阵的未覆盖错误列，没有被剔除。该错误列不是第八语义类。','', '## 消融的实测变化','']
    for r in deltas:lines.append(f'- {r["before"]} → {r["after"]}（{r["module"]}）：mIoU变化{100*r["mIoU"]:+.2f}个百分点，PSNR变化{r["psnr"]:+.3f}dB，LPIPS变化{r["lpips"]:+.4f}。')
    lines += ['', '上述差值按实际结果报告；新增点数增加本身不是重建收益证据。ISLB多训练10k步，其差异不能归结为零额外预算下的单纯结构优势。','', '## 工业局部和跨视角分析','18个局部ROI在查看模型结果前固定，覆盖密集管路、阀件区域、设备、桥架支撑与遮挡。阀门合并在piping内，没有独立“阀门IoU”。T04保留每个ROI及七类结果；F07保留同区域预测。','9组实际可见对象/区域由RGB形状及邻近布局进行视觉身份检查，保存框与依据。它们不是完整实例级稠密对应真值，也不是SAM自动身份真值。T05将正确一致、错误一致、不一致分别计数；特征余弦与KL只在有贡献的对象像素上计算，覆盖率同时报告。','']
    for n,c in cross.items():s=c['summary'];lines.append(f'- {n}：正确一致{100*s["correct_consistent"]:.2f}%，错误一致{100*s["wrong_consistent"]:.2f}%，不一致{100*s["inconsistent"]:.2f}%；区域概率对称KL={s["probability_symmetric_KL"]:.5f}，高斯语义对称KL={s["gaussian_symmetric_KL"]:.5f}。')
    lines += ['',f'T05一致性比例按{len(cross["B-SEM"]["pairs"])}个对象视角对等权计算；F08矩阵按9组对象的全部观察共同判定。视角对来自同一小批对象，不是独立统计样本。']
    if cross['B-SEM']['summary']['correct_consistent']==1:
        lines += ['B-SEM在预先固定对象集上的主导类别已经全部正确且一致，这项指标存在上限效应，不能据此证明ICSC提升了正确一致率。未根据模型结果更换对象；局部像素错误、分布KL和总体测试mIoU分别报告。']
    lines += detailed_lines
    lines += ['', '## 论文资料、结论边界与复现','F01–F09、F11和T01–T06由本轮实际数据生成，注册表逐项关联脚本、原始数据和模型。F10为可选输入数量实验，本轮未执行。256×128语义为主评价；独立512×256检查调用真实光栅渲染，不以插值放大冒充细节。','本轮可以支持固定IRIS工业区域、20个稀疏全景输入下的重建与七类语义可行性，以及各模块在此协议下的实测消融结论。不能据此声称跨场景泛化、开放词汇识别、阀门独立类别准确率、SAM对应真值准确率或统计显著优于全部方法。单场景、单种子、合成底部、投影标签误差、稀疏ROI身份样本与低语义分辨率是局限。','与OmniGS相比，加入七类32维语义高斯及工业标签原型监督；参照SPC-GS两阶段框架，用四方向Wan360替代原生成路径、采用工业固定标签与本轮球面几何注册，并以SAM2 Hiera-L实现图像/高斯域协同。报告不声称逐项复现原论文的所有数据和运行设置。','GPU为共享A6000，多任务并行运行，T06耗时受同机其他作业影响。模型archive包含优化器，不能把其大小直接当作精简推理模型大小。','', '```bash',f'cd {ROOT}',f'.venv/bin/python tools/run_ipsgr.py --run {run} --stage report','```','', '[SPC-GS论文](https://arxiv.org/html/2503.12535)','']
    reportdir=run/'10_report';(reportdir/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    body=prose_html(lines);gallery=''.join(f'<section><h2>{key}</h2><p>'+html.escape(v.get('note',v.get('reason','')))+'</p>'+''.join(f'<figure><a href="../08_paper_figures/{Path(p).name}"><img loading="lazy" src="../08_paper_figures/{Path(p).name}"></a><figcaption>'+html.escape(Path(p).stem.replace('_',' '))+'</figcaption></figure>' for p in v.get('figures',[]) if str(p).endswith('.png'))+'</section>' for key,v in sorted(read(run/'08_paper_figures/registry.json').items()));page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>IPSGR20/6 实验报告</title><style>body{max-width:1600px;margin:30px auto;padding:0 24px;font:16px/1.7 sans-serif;color:#203347}pre{white-space:pre-wrap;background:#f1f5f8;padding:18px}img{width:100%;border:1px solid #dce3e9;margin:10px 0}section{margin:50px 0}figure{margin:24px 0}figcaption{font-size:14px;color:#4f6576}a{color:#14649b}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px 14px;border-bottom:1px solid #dce3e9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f8}h2{margin-top:40px}</style><nav><a href="viewer.html">六图预测与类别查询浏览</a> · <a href="report.md">中文报告源文件</a> · <a href="completion_audit.json">完成核验</a> · <a href="#gallery">论文图表</a> · <a href="../09_paper_tables/T02.csv">核心指标CSV</a></nav>'+body+'<h2>原始表格下载</h2><p>'+ ' · '.join(f'<a href="../09_paper_tables/{p.name}">{p.stem}</a>' for p in sorted((run/'09_paper_tables').glob('*.csv')))+'</p><h1 id="gallery">论文图表</h1>'+gallery
    (reportdir/'report.html').write_text(page,encoding='utf-8');manifest=read(run/'experiment_manifest.json');manifest.update(status='completed',completed=time.time(),models=evidence['models'],completion_audit=ref(reportdir/'completion_audit.json'));write(run/'experiment_manifest.json',manifest);print('REPORT_COMPLETE',reportdir,flush=True)
