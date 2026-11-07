"""One explicit input/reference repair after V1, before any policy exposure.

Retains V1 intact. This is task development, not a preregistered model result.
"""
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    a = ap.parse_args()
    p,t = a.run.resolve(),a.runtime.resolve()
    q=p/'input_v2'
    q.mkdir(exist_ok=False)
    def read(f): return json.loads(f.read_text())
    def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
    def write(n,x): (q/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    failure=read(t/'feasibility/RESULT.json')
    assert failure['status']=='measured_with_rejected_candidates'
    ref=read(p/'INITIAL_REFERENCE.json')
    shower=next(o for o in ref['furniture'] if o['name']=='bath_shower')
    before=copy.deepcopy(shower)
    shower['location'][2]-=.05
    write('INITIAL_REFERENCE.json',ref)
    task=read(p/'TASK_INPUTS.json')
    write('TASK_INPUTS.json',task)
    assert sha(q/'TASK_INPUTS.json')==sha(p/'TASK_INPUTS.json')
    build=read(p/'BUILD_PLAN.json')
    build['initial_reference']={'path':str(q/'INITIAL_REFERENCE.json'),'sha256':sha(q/'INITIAL_REFERENCE.json')}
    write('BUILD_PLAN.json',build)
    feasibility=read(p/'FEASIBILITY_PLAN.json')
    def move(oid,x,y):return {'name':'move_furniture_tool','arguments':{'object_id':'recon_'+oid+'_0','x':x,'y':y,'yaw':0}}
    feasibility['references']['desk_work']=[move('mobile_table',.1,0),move('work_stool',-.9,-.85),
        move('work_stool',-.9,0),move('mobile_table',-.9,-.85)]
    feasibility['created_at']=datetime.now(timezone.utc).isoformat()
    feasibility['rules']='One explicit V2 repair, no search or subsequent retries. Reference endpoints unchanged; only obstacle-avoiding move order changes.'
    write('FEASIBILITY_PLAN.json',feasibility)
    write('REVISION.json',dict(created_at=feasibility['created_at'],reason='V1 had overlapping bath proxies and a blocked stool translation. V1 closed as rejected, not erased.',
        original_result=str(t/'feasibility/RESULT.json'),original_result_sha256=sha(t/'feasibility/RESULT.json'),
        changes=dict(shower_before=before,shower_after=shower,desk_reference='Temporarily move low table, move stool in two clear translations, return low table.'),
        unchanged=['Requests','Thresholds','Room size','All other initial objects','Reference endpoints','0.6m bath-access failure retained'],
        budget=dict(new_native_builds=1,build_seconds=240,candidates=2,max_native_reference_moves=6,reference_seconds_per_condition=300,model_calls=0,training_updates=0),
        restriction='This task is under developer construction, not a blind test. No model output has been consulted. No further layout repair in this batch.'))
    execution=read(p/'EXECUTION_PLAN.json')
    execution['build_command']=[str(q/'BUILD_PLAN.json') if x==str(p/'BUILD_PLAN.json') else str(t/'build_v2') if x==str(t/'build') else x for x in execution['build_command']]
    execution.update(frozen_at=feasibility['created_at'],revision=2,hashes={str(f):sha(f) for f in [q/'BUILD_PLAN.json',q/'INITIAL_REFERENCE.json',q/'TASK_INPUTS.json',q/'FEASIBILITY_PLAN.json',Path(__file__)]})
    write('EXECUTION_PLAN.json',execution)
    print(json.dumps({'revision':2,'request_bytes_unchanged':True,'new_model_calls':0}))


if __name__=='__main__':main()
