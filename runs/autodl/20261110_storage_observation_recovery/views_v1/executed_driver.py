"""Serial native observation recovery for explicitly frozen states.

Stops on the first failure, preserves logs, never retries or edits a target.
Only the completed per-state BLEND intermediate is removed after annotation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['out','source','bundle','ledger','units','python']:
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    units=json.loads(a.units.read_text())
    ids=[r['unit_id'] for r in units]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate units')
    for uid in ids:
        if Path(uid).name!=uid:raise ValueError('Invalid unit name')
    before={str(p):sha(p) for p in a.bundle.rglob('*') if p.is_file()}
    before[str(a.units)]=sha(a.units);before[str(a.ledger)]=sha(a.ledger)
    rows=[];start=time.monotonic();commands=[]
    def save():
        d={'status':'running','completed':len(rows),'planned':len(ids),'rows':rows,
           'commands':commands,'elapsed_seconds':time.monotonic()-start,'inputs_sha256':before,
           'model_calls':0,'training_updates':0,'new_scene_generations':0}
        (a.out/'progress.json').write_text(json.dumps(d,indent=2)+'\n')
    def run(args, log, timeout):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        tick=time.monotonic();rec={'argv':args,'log':str(log),'status':'started'};commands.append(rec);save()
        try:
            with log.open('w') as f:
                cp=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
            rec.update(returncode=cp.returncode,status='completed' if cp.returncode==0 else 'failed')
            if cp.returncode:raise RuntimeError('Stage failed; see '+str(log))
        finally:
            rec['seconds']=time.monotonic()-tick;save()
    error=None
    try:
        for uid in ids:
            if shutil.disk_usage(a.out).free<5*1024**3+48*1024**2:
                raise RuntimeError('Insufficient temporary export/view headroom')
            d=a.out/uid;d.mkdir();export=d/'export';views=d/'views'
            common=[str(a.python),'-B']
            print(uid,'native export',flush=True)
            run(common+['scripts/autodl/export_paper_native_blend.py','--source',str(a.source),'--imported',str(a.bundle),
                '--out',str(export),'--state-ledger',str(a.ledger),'--unit',uid,'--checkpoint-labels'],d/'export.log',180)
            ex=json.loads((export/'result.json').read_text())
            if ex['source_room_sha256']!=ex['runtime_room_sha256']:raise ValueError('Export changed state')
            print(uid,'five-view render',flush=True)
            run(common+['scripts/autodl/render_paper_native_views.py','--source',str(a.source),'--export',str(export),
                '--out',str(views),'--seconds','600'],d/'render.log',660)
            run(common+['scripts/autodl/finish_paper_native_observation.py','--source',str(a.source),'--imported',str(a.bundle),
                '--export',str(export),'--views',str(views),'--checkpoint-labels'],d/'annotation.log',120)
            ann=json.loads((views/'annotation_result.json').read_text())
            if ann['status']!='native_five_view_observation_completed' or ann['runtime_room_sha256']!=ex['source_room_sha256']:
                raise ValueError('Annotated scene differs')
            for im in ann['images']:
                if sha(im['path'])!=im['sha256']:raise ValueError('Image changed')
            blend=Path(ex['blend_path'])
            if sha(blend)!=ex['blend_sha256']:raise ValueError('Intermediate changed')
            cleanup={'path':str(blend),'sha256':sha(blend),'bytes':blend.stat().st_size,
                     'reason':'Completed render/annotation; native resources and all views retained.'}
            blend.unlink();(d/'cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
            rows.append({'unit_id':uid,'export':str(export/'result.json'),'annotation':str(views/'annotation_result.json'),
                         'state_sha256':ex['source_room_sha256'],'images':len(ann['images'])})
            print(uid,'completed',flush=True);save()
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);raise
    finally:
        save();d=json.loads((a.out/'progress.json').read_text())
        d.update(status='failed' if error else 'completed',error=error,
                 inputs_unchanged=all(sha(p)==h for p,h in before.items()),
                 training_ready=False,historical_pixels_equivalent=False)
        (a.out/'result.json').write_text(json.dumps(d,indent=2)+'\n')


if __name__=='__main__':main()
