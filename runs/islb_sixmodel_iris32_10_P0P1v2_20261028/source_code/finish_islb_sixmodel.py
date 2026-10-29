"""One-shot postprocessing for this explicitly requested experiment (not a timer job)."""
from pathlib import Path
import os, sys, time, traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import read, write
from spcgs.islb_sixmodel_data import RUN
from spcgs.islb_sixmodel_train import gpu_inventory
from spcgs.islb_sixmodel_report import qualitative, build

if __name__ == '__main__':
    import fcntl
    lock = (RUN/'scheduler/postprocess.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    status = RUN/'scheduler/postprocess_status.json'
    write(status, {'status':'waiting_for_all_six_evaluations', 'pid':os.getpid()})
    try:
        while read(RUN/'status.json')['status'] != 'completed':
            time.sleep(30)
        candidates = sorted([g for g in gpu_inventory() if g['free_MB'] >= 16000], key=lambda g:-g['free_MB'])
        assert candidates, 'No GPU with enough free memory for qualitative renders'
        gpu = candidates[0]['id']
        for name in ['B-SEM','L0','L1','L2','ISLB-Only']:
            write(status, {'status':'rendering_fixed_ROIs', 'model':name, 'GPU':gpu, 'pid':os.getpid()})
            qualitative(name,gpu)
        write(status, {'status':'building_report', 'pid':os.getpid()})
        build()
        write(status, {'status':'completed', 'report':str(RUN/'09_report/portable/report.html'), 'finished':time.time()})
    except Exception:
        p=RUN/'scheduler'/f'postprocess_failure_{time.time_ns()}.log';p.write_text(traceback.format_exc())
        write(status, {'status':'failed', 'failure_log':str(p)})
        raise
