"""Encode a completed fixed observation batch and its ordinary joint control.

Uses the local processor only; never loads model weights. An optional bounded
wait observes a known producer PID and a terminal receipt, never restarts it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['run','base','ledger','processor','python','scene-observation']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--wait-for-pid',type=int)
    ap.add_argument('--wait-seconds',type=int,default=1500)
    a=ap.parse_args();receipt=a.run/'material_pipeline.json'
    if receipt.exists():raise FileExistsError('Existing pipeline receipt; inspect it before any continuation')
    result={'status':'waiting_for_observations','commands':[],'model_calls':0,'training_updates':0,
            'new_scene_generations':0,'paid_api_calls':0,'producer_pid':a.wait_for_pid}
    start=time.monotonic()
    def save():
        result['seconds']=time.monotonic()-start
        receipt.write_text(json.dumps(result,indent=2)+'\n')
    def run(argv,name,timeout):
        log=a.run/(name+'.log');row={'argv':argv,'log':str(log),'status':'started'}
        result['commands'].append(row);save();tick=time.monotonic()
        try:
            with log.open('w') as f:
                p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
            row.update(returncode=p.returncode,status='completed' if p.returncode==0 else 'failed')
            if p.returncode:raise RuntimeError('Stage failed: '+name)
        finally:
            row['seconds']=time.monotonic()-tick;save()
    save()
    try:
        terminal=a.run/'views_v2/result.json'
        while not terminal.exists():
            if not a.wait_for_pid:raise FileNotFoundError('No completed observation receipt')
            if time.monotonic()-start>a.wait_seconds:raise TimeoutError('Producer observation limit, not a renderer restart')
            try:os.kill(a.wait_for_pid,0)
            except ProcessLookupError:
                if terminal.exists():break
                raise RuntimeError('Producer absent without terminal receipt')
            time.sleep(5)
        batch=json.loads(terminal.read_text())
        if batch['status']!='completed' or len(batch['rows'])!=5:
            raise ValueError('Producer did not complete all five fixed states')
        registry=json.loads((a.base/'registry.json').read_text())
        for r in batch['rows']:
            if r['unit_id'] in registry:raise ValueError('Do not overwrite an earlier observation')
            registry[r['unit_id']]={k:r[k] for k in ['export','annotation']}
        reg=a.run/'registry.json';reg.write_text(json.dumps(registry,indent=2)+'\n')
        result.update(status='encoding',producer_result_sha256=sha(terminal),
                      input_material_sha256=sha(a.base/'material_v3/components.jsonl'))
        save();py=[str(a.python),'-B'];material=a.run/'material_v4';joint=a.run/'joint_v4'
        run(py+['scripts/autodl/build_rebuilt_observation_material.py',
            '--material',str(a.base/'material_v3/components.jsonl'),'--registry',str(reg),
            '--state-ledger',str(a.ledger),'--processor',str(a.processor),
            '--scene-observation',str(a.scene_observation),'--out',str(material)],'encode_split',300)
        run(py+['scripts/autodl/build_joint_serialization_control.py','--material',str(material/'components.jsonl'),
            '--processor',str(a.processor),'--out',str(joint)],'build_joint',120)
        run(py+['scripts/autodl/check_joint_multimodal_control.py','--material',str(joint/'components.jsonl'),
            '--equivalence','runs/autodl/20261110_exact_view_recovery/pixel_v1/pixel_equivalence.json',
            '--bindings',str(material/'observation_bindings.json'),'--reference-encoding',str(material/'encoding_records.json'),
            '--processor',str(a.processor),'--scene-observation',str(a.scene_observation),
            '--out',str(a.run/'joint_multimodal_v4')],'encode_joint',300)
        run(py+['scripts/autodl/verify_storage_observation_recovery.py','--run',str(a.run)],'verify_final',120)
        if sha(a.base/'material_v3/components.jsonl')!=result['input_material_sha256']:
            raise ValueError('Old material changed')
        result.update(status='completed',training_ready=False,full_runtime_ready=False,
                      scope='Actual processor encoding and artifact checks; no free model output, optimization or generated scene')
    except BaseException as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        save();print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
