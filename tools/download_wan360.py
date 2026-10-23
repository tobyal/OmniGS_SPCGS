"""Download the pinned official Wan360 release, including VAE/T5/tokenizer."""
import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download, get_hf_file_metadata, hf_hub_url

REVISION = '914e60491baca94bf7c3346551bdb168b2445559'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def download_http(root, siblings, workers, direct_cdn=False, modelscope_base=False):
    """Bounded, proxy-compatible Range downloads, verified against official LFS SHA.

    Some compute servers route HTTPS through a proxy incompatible with Xet or
    hf_transfer. Complete chunks survive interruption; never trust sparse partial
    files left by a different downloader.
    """
    import requests
    parts=root/'.download_parts';parts.mkdir(parents=True,exist_ok=True)
    jobs=[];large=[]
    for item in siblings:
        target=root/item.rfilename
        if item.size < 64<<20:
            if target.is_file() and target.stat().st_size==item.size:
                if item.lfs:
                    verified=file_sha(target)==item.lfs.sha256
                else:
                    blob=target.read_bytes()
                    verified=hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()==item.blob_id
                if verified:continue
            hf_hub_download('AlayaLab/Wan360',item.rfilename,revision=REVISION,local_dir=root)
            continue
        expected=item.lfs.sha256
        if target.exists() and target.stat().st_size==item.size and file_sha(target)==expected:
            continue
        url=hf_hub_url('AlayaLab/Wan360',item.rfilename,revision=REVISION)
        base_component=modelscope_base and item.rfilename in {
            'models_t5_umt5-xxl-enc-bf16.pth','Wan2.2_VAE.pth'}
        if base_component:
            location='https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B/resolve/master/'+item.rfilename
        else:
            location=get_hf_file_metadata(url,timeout=60).location
        chunks=[]
        for start in range(0,item.size,32<<20):
            end=min(start+(32<<20),item.size)-1
            chunk=parts/(expected+f'.{start}-{end}')
            chunks.append(chunk)
            if not chunk.exists() or chunk.stat().st_size!=end-start+1:
                jobs.append((url,location,start,end,item.size,chunk,base_component))
        large.append((item,target,expected,chunks))
    def fetch(job):
        url,location,start,end,total,chunk,base_component=job
        if chunk.is_file() and chunk.stat().st_size==end-start+1:
            return 0
        session=requests.Session()
        session.trust_env=not (direct_cdn or base_component)
        for attempt in range(5):
            try:
                with session.get(location,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90)) as response:
                    if response.status_code in (401,403) and not base_component:
                        location=get_hf_file_metadata(url,timeout=60).location
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}':
                        raise RuntimeError(f'Unexpected range response: HTTP {response.status_code}')
                    temporary=chunk.with_name(chunk.name+'.tmp')
                    with temporary.open('wb') as f:
                        for data in response.iter_content(1<<20):f.write(data)
                if temporary.stat().st_size!=end-start+1:raise RuntimeError('Truncated range response')
                temporary.replace(chunk)
                return end-start+1
            except Exception as error:
                print('Range retry',start,attempt+1,type(error).__name__,flush=True)
                if attempt==4:raise RuntimeError(f'Failed range at byte {start}') from None
                time.sleep(2*(attempt+1))
    done=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch,job) for job in jobs]):
            done+=future.result()
            print(f'Downloaded new chunks: {done/(1<<30):.2f} GiB',flush=True)
    for item,target,expected,chunks in large:
        temporary=target.with_name(target.name+'.assembled')
        with temporary.open('wb') as out:
            for chunk in chunks:
                with chunk.open('rb') as inp:shutil.copyfileobj(inp,out,8<<20)
        if file_sha(temporary)!=expected:raise RuntimeError(f'Official LFS checksum mismatch: {item.rfilename}')
        temporary.replace(target)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'weights/Wan360'))
    p.add_argument('--http-workers',type=int,default=int(os.environ.get('WAN360_HTTP_WORKERS','0')))
    p.add_argument('--direct-cdn',action='store_true',help='Bypass proxy only for large-file CDN requests')
    p.add_argument('--modelscope-base',action='store_true',help='Fetch shared T5/VAE from official Wan-AI ModelScope; still require identical Wan360 LFS hashes')
    a = p.parse_args()
    info=HfApi().model_info('AlayaLab/Wan360',revision=REVISION,files_metadata=True)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if a.http_workers:
        download_http(root,info.siblings,a.http_workers,a.direct_cdn,a.modelscope_base)
    else:
        snapshot_download('AlayaLab/Wan360',revision=REVISION,local_dir=root,max_workers=3)
    records = {}
    for item in info.siblings:
        path=root/item.rfilename;checksum=file_sha(path)
        if item.lfs and checksum!=item.lfs.sha256:raise RuntimeError(f'Official LFS checksum mismatch: {item.rfilename}')
        records[item.rfilename]={'bytes':path.stat().st_size,'sha256':checksum}
    (root / 'provenance.json').write_text(json.dumps({
        'repository': 'AlayaLab/Wan360', 'revision': REVISION, 'files': records,
        'transport': {'http_workers':a.http_workers,'direct_cdn':a.direct_cdn,
                      'modelscope_base':a.modelscope_base,
                      'integrity':'Every LFS file must match the pinned official Wan360 SHA256'}}, indent=2))
    print('Wan360 weights ready:', root, flush=True)

if __name__ == '__main__':
    main()
