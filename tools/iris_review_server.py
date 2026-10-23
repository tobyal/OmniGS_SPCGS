"""Local-only IRIS review API. Explicit revision-bound human approval."""
import argparse,base64,io,sys,secrets,threading
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask,request,jsonify,send_file,send_from_directory,abort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.segmentation import *

def create_app(root=DEFAULT_RUN,enable_sam=True):
    store=Store(root);app=Flask(__name__,static_folder=None);app.config['MAX_CONTENT_LENGTH']=12*1024*1024
    token=secrets.token_hex(24);sam=[None];sam_lock=threading.Lock()
    web=ROOT/'web/iris_review'
    @app.before_request
    def protect():
        if request.method=='POST' and request.headers.get('X-Review-Token')!=token:abort(403)
    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response
    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    def bad(e):return jsonify(error=str(e)),400
    @app.errorhandler(RuntimeError)
    def conflict(e):return jsonify(error=str(e)),409
    @app.get('/')
    def index():return send_from_directory(web,'index.html')
    @app.get('/static/<path:name>')
    def static(name):return send_from_directory(web,name)
    @app.get('/api/bootstrap')
    def bootstrap():return jsonify(config=store.config,token=token,sam_enabled=enable_sam,images=[{**i,**store.state(n),
        'official_faces':sum(f['has_record'] for f in read(store.folder(n)/'coverage.json'))} for n,i in store.items.items()])
    @app.get('/api/image/<name>')
    def detail(name):
        f=store.folder(name);return jsonify(item=store.items[name],state=store.state(name),coverage=read(f/'coverage.json'),
            sam=read(f/'sam_candidates.json') if (f/'sam_candidates.json').exists() else None,
            revisions=[read(p) for p in sorted((f/'revisions').glob('*/metadata.json'),key=lambda p:p.stat().st_mtime)])
    @app.get('/api/image/<name>/<kind>')
    def asset(name,kind):
        folder=store.folder(name)
        if kind=='rgb':path=Path(store.items[name]['rgb'])
        elif kind in ['thumb','official_raw','official_mapped']:
            path=folder/(kind+('.jpg' if kind=='thumb' else '.png'))
        elif kind in ['labels','source']:
            state=store.state(name);path=folder/'revisions'/state['revision']/(kind+'.png')
        elif kind=='preview':
            labels,_,_=store.arrays(name);rgb=np.array(Image.open(store.items[name]['rgb']).convert('RGB'));p=palette(store.config);mask=labels!=255
            rgb[mask]=(.55*rgb[mask]+.45*p[labels[mask]]).astype('u1');return send_file(io.BytesIO(png_bytes(rgb)),mimetype='image/png')
        else:abort(404)
        return send_file(path)
    @app.get('/api/image/<name>/perspective/<face>')
    def perspective_view(name,face):
        store.folder(name)
        if face not in dict(FACES):abort(404)
        labels,_,_=store.arrays(name);rgb=np.array(Image.open(store.items[name]['rgb']).convert('RGB'))
        p=palette(store.config);known=labels!=255;rgb[known]=(.55*rgb[known]+.45*p[labels[known]]).astype('u1')
        result=perspective(rgb,dict(FACES)[face],1024,100)
        return send_file(io.BytesIO(png_bytes(result)),mimetype='image/png')
    @app.post('/api/image/<name>/save')
    def save(name):
        d=request.get_json();old,source,state=store.arrays(name)
        b=base64.b64decode(d['labels_u8'],validate=True)
        if len(b)!=2048*1024:raise ValueError('Expected one byte per ERP pixel')
        new=np.frombuffer(b,dtype='u1').reshape(1024,2048);validate(new,store.config);source[new!=old]=3
        return jsonify(store.save(name,new,source,d['revision'],actor='human:'+d.get('reviewer','anonymous'),details={'note':d.get('note','')}))
    @app.post('/api/image/<name>/review')
    def review(name):
        d=request.get_json()
        if d.get('approve') and d.get('confirmation')!='human_reviewed':raise ValueError('Explicit human confirmation required')
        return jsonify(store.review(name,d['revision'],bool(d.get('approve')),d.get('reviewer',''),d.get('note',''),d.get('issues')))
    @app.post('/api/image/<name>/restore')
    def restore(name):
        d=request.get_json();folder=store.folder(name)
        rev=d['restore_revision']
        if len(rev)!=16 or any(c not in '0123456789abcdef' for c in rev):raise ValueError('Invalid revision')
        path=folder/'revisions'/rev
        labels=np.array(Image.open(path/'labels.png'));source=np.array(Image.open(path/'source.png'))
        return jsonify(store.save(name,labels,source,d['revision'],actor='human:'+d.get('reviewer','anonymous'),action='restore',details={'restored':rev}))
    @app.post('/api/image/<name>/sam')
    def predict(name):
        if not enable_sam:abort(503)
        store.folder(name);d=request.get_json();points=d['points'];labels=d.get('point_labels')
        if not points or len(points)>50 or any(len(p)!=2 or not (0<=p[0]<2048 and 0<=p[1]<1024) for p in points):raise ValueError('Invalid points')
        if labels is not None and (len(labels)!=len(points) or not set(labels).issubset({0,1}) or 1 not in labels):raise ValueError('Invalid point labels')
        with sam_lock:
            if sam[0] is None:sam[0]=SamSegmenter()
            mask,info=sam[0].predict(store.items[name]['rgb'],points,labels)
        return jsonify(mask=base64.b64encode(png_bytes(mask.astype('u1')*255)).decode(),info=info)
    @app.get('/api/export')
    def export():
        approved=request.args.get('scope','approved')=='approved';data,count=store.export(approved)
        return send_file(io.BytesIO(data),mimetype='application/zip',as_attachment=True,download_name=f'iris_{"approved" if approved else "UNREVIEWED_candidates"}_{count}.zip')
    return app

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(DEFAULT_RUN));p.add_argument('--port',type=int,default=8876);p.add_argument('--no-sam',action='store_true');a=p.parse_args()
    create_app(a.run,not a.no_sam).run(host='127.0.0.1',port=a.port,threaded=True,use_reloader=False)
