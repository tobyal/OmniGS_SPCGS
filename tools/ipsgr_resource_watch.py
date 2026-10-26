"""Read-only GPU telemetry while the current IPSGR experiment is active."""
import csv,io,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import read,write
from spcgs.ipsgr_protocol import DEFAULT_RUN,MODEL_NAMES


def query(fields,kind='gpu'):
    out=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+','.join(fields),'--format=csv,noheader,nounits'],text=True)
    return [dict(zip(fields,[v.strip() for v in row])) for row in csv.reader(io.StringIO(out))]


def main():
    run=DEFAULT_RUN;dest=run/'10_report';peaks={};started=time.time()
    with (dest/'resource_samples.jsonl').open('a') as f:
        while time.time()-started<12*3600:
            gpus=query(['index','uuid','memory.used','memory.total','utilization.gpu']);index={r['uuid']:int(r['index']) for r in gpus};processes=[]
            for p in query(['pid','gpu_uuid','used_gpu_memory'],'compute-apps'):
                try:cmd=(Path('/proc')/p['pid']/'cmdline').read_text().replace('\0',' ')
                except FileNotFoundError:continue
                if str(run) not in cmd:continue
                model='ISLB' if str(run/'03_layout_bootstrap') in cmd else next((n for n in ['smoke-IPSGR-Full',*MODEL_NAMES] if '/'+n+'/' in cmd),'other')
                try:memory=int(p['used_gpu_memory'])
                except ValueError:continue
                processes.append({'pid':int(p['pid']),'gpu':index[p['gpu_uuid']],'model':model,'memory_MiB':memory})
            row={'time':time.time(),'gpus':gpus,'experiment_processes':processes};f.write(json.dumps(row)+'\n');f.flush()
            for name in {p['model'] for p in processes}:
                memory=sum(p['memory_MiB'] for p in processes if p['model']==name);peaks[name]=max(peaks.get(name,0),memory)
            write(dest/'resource_peaks.json',{'started':started,'updated':time.time(),'sample_interval_seconds':3,'per_model_observed_sum_MiB':peaks,'note':'Sampled process-resident memory, including persistent SAM for that model; not a CUDA allocator maximum. Observation begins during the running experiment.'})
            if read(run/'experiment_manifest.json').get('status')=='completed' or (dest/'runner_failure.json').exists():break
            time.sleep(3)


if __name__=='__main__':main()
