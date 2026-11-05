"""Execute a prospective input-only batch once, preserving actual stage exits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from native_project import sha256


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args();protocol=read(a.protocol)
    repo=Path(__file__).resolve().parents[2];public=a.protocol.resolve().parent;t=repo.parent/'runtime'/protocol['run_id']
    for name,digest in protocol['code_sha256'].items():assert sha256(repo/'scripts/920'/name)==digest,name
    now=lambda:datetime.now(timezone.utc).isoformat();state=dict(status='running',started_at=now(),variants={},model_calls=0,training_updates=0)
    assert not (t/'BATCH.json').exists()
    def save():
        state['updated_at']=now();write(t/'BATCH.json',state)
        write(public/'STATUS.json',dict(run_id=protocol['run_id'],question=protocol['question'],status=state['status'],
            started_at=state['started_at'],ended_at=state.get('ended_at'),updated_at=state['updated_at'],
            completed=len(state['variants']),planned=3,configuration_sha256=sha256(a.protocol),runtime_directory=str(t),
            failure_reason=state.get('error'),latest_verifiable_artifact=str(t/'BATCH.json'),next_step='Review nominal observation and retain source/task isolation; no model inference.'))
        write(public.parent/'latest.json',dict(run_id=protocol['run_id'],stage=state['status'],updated_at=now(),run_directory=str(t)))
    def stage(label,seconds,python,script,*args):
        subprocess.run([sys.executable,str(repo/'scripts/920/run_stage.py'),'--receipt-dir',str(t/(label+'_execution')),
            '--timeout',str(seconds),'--',str(python),str(repo/'scripts/920'/script),*map(str,args)],check=True,cwd=repo)
    print(now(),'begin three prospective input variants, no model',flush=True);save()
    try:
        for label,config in protocol['variants'].items():
            pp=Path(config['plan']['path']);assert sha256(pp)==config['plan']['sha256'];plan=read(pp);out=Path(config['out'])
            stage(label+'_build',180,sys.executable,'build_moabit_input.py','--plan',pp,'--out',out)
            stage(label+'_geometry',60,repo.parent/'envs/geometry311/bin/python','check_moabit_input.py',
                '--task',plan['inputs']['task']['path'],'--bundle',out/'bundle','--out',t/label/'GEOMETRY.json')
            state['variants'][label]=dict(build=read(out/'RESULT.json'),geometry=read(t/label/'GEOMETRY.json'));save()
        nominal=state['variants']['nominal']['build'];old=read(repo/'runs/920/P920-CASA-OBS-v1_20261105T123507Z/PLAN.json')
        observe={k:old[k] for k in ['designer','scenesmith']}
        observe.update(run_id=protocol['run_id'],question=protocol['question'],bundle=nominal['bundle'],manifest_sha256=nominal['manifest_sha256'],
            public_directory=str(public/'nominal'),runtime_directory=str(t/'nominal'),budget=dict(views=5,seconds=300,model_calls=0,training_updates=0),
            code_sha256={n:sha256(repo/'scripts/920'/n) for n in ['observe_heldout_input.py','catalog_project_session.py','native_project.py','native_checkpoint.py']})
        write(public/'nominal/OBSERVE_PLAN.json',observe)
        stage('nominal_observation',300,sys.executable,'observe_heldout_input.py','--plan',public/'nominal/OBSERVE_PLAN.json','--out',t/'nominal/observation')
        state.update(status='native_inputs_and_nominal_views_complete_pending_review',observation=read(t/'nominal/observation/RESULT.json'))
    except Exception as exc:
        state.update(status='failed',error=str(exc));raise
    finally:
        state['ended_at']=now();save();print(now(),state['status'],flush=True)


if __name__=='__main__':main()
