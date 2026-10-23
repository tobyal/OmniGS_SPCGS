"""Validate and export revision-bound approved labels or explicit candidates."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.segmentation import Store,DEFAULT_RUN

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(DEFAULT_RUN))
    p.add_argument('--scope',choices=['approved','candidates'],default='approved');p.add_argument('--output',required=True);a=p.parse_args()
    data,count=Store(a.run).export(a.scope=='approved');dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    print(f'{count} images -> {dest} ({len(data)} bytes); scope={a.scope}')
    if a.scope=='candidates':print('Includes UNREVIEWED candidates; not a human-approved training set.')
