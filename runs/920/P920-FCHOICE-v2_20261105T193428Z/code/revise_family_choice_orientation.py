"""Create a new version after visual rejection; never mutate the executed input."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',required=True);a=ap.parse_args()
    r=Path(__file__).resolve().parents[2];old=r/'runs/920'/a.previous
    now=datetime.now(timezone.utc);run='P920-FCHOICE-v2_'+now.strftime('%Y%m%dT%H%M%SZ')
    p=r/'runs/920'/run;t=r.parent/'runtime'/run;p.mkdir();t.mkdir()
    for name in ['SOURCE_RECORD.json','SOURCE_RECHECK.json','TASK_VARIANTS.json','RAY_ANALYTIC_CHECKS.json']:
        shutil.copy2(old/name,p/name)
    before=read(old/'INITIAL_REFERENCE.json');after=copy.deepcopy(before)
    for obj in after['furniture']:
        if obj['name']=='sofa':obj['facing']=270
        if obj['name']=='tall_storage':obj['facing']=180
    write(p/'INITIAL_REFERENCE.json',after)
    build=read(old/'REFERENCE_BUILD_PLAN.json');build['run_id']=run;build['frozen_at']=now.isoformat()
    build['project']['project_id']=run
    build['initial_reference']={'path':str(p/'INITIAL_REFERENCE.json'),'sha256':sha(p/'INITIAL_REFERENCE.json')}
    write(p/'REFERENCE_BUILD_PLAN.json',build)
    choice=read(old/'CHOICE_PLAN.json');choice['run_id']=run;choice['frozen_at']=now.isoformat()
    choice['candidates']['perimeter']['yaw']=180
    write(p/'CHOICE_PLAN.json',choice)
    write(p/'CORRECTION.json',dict(previous_run=a.previous,created_at=now.isoformat(),
        evidence=dict(visual_review=str(old/'VISUAL_REVIEW.json'),sha256=sha(old/'VISUAL_REVIEW.json')),
        changes=['Initial sofa facing90→270 to face the shared room.',
                 'Initial storage facing0→180; perimeter target yaw0→180 to match the declared accessible face.'],
        unchanged=['Room dimensions/openings','All furniture IDs/dimensions/asset hashes',
                   'Both candidate positions','Divider target yaw90','Both task conditions',
                   'All view-ray points/heights and access targets','One action/two observations per candidate'],
        new_source_family=False,independent_test=False,training_updates=0))
    write(p/'MEASUREMENT_LOCK.json',dict(code_sha256=sha(r/'scripts/920/measure_family_choice.py'),
        choice_plan_sha256=sha(p/'CHOICE_PLAN.json')))
    write(p/'STATUS.json',dict(run_id=run,status='prepared_orientation_revision',completed=0,planned=2,
        question='Does the two-condition layout tradeoff remain after correcting the visually rejected input orientations?',
        updated_at=now.isoformat(),runtime_directory=str(t),configuration_sha256=sha(p/'CHOICE_PLAN.json'),
        failure_reason=None,next_step='Build corrected input, preflight and execute each candidate once.'))
    write(r/'runs/920/latest.json',dict(run_id=run,stage='prepared_orientation_revision',
        updated_at=now.isoformat(),run_directory=str(t)))
    Path('/tmp/fchoice_current').write_text(run);print(run)


if __name__=='__main__':main()
