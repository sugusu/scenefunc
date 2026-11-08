"""Freeze a prospective two-adapter room-learning proposal; never train."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from native_project import sha256
from shared_action_contract import context_from_request, schemas_from_observation


def read(p): return json.loads(Path(p).read_text())
def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-id',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];scripts=repo/'scripts/920';d=repo.parent
    root=repo/'runs/920'/a.run_id;root.mkdir(exist_ok=False)
    runtime=d/'runtime'/a.run_id
    eq=repo/'runs/920/P920-EQUALMATERIAL-v1_20261108T073050Z'
    source=repo/'runs/920/P920-MATCHEDROLL-v1_20261108T062940Z'
    train=read(eq/'ENTRY_CPU_PLAN.json')
    train.update(run_id=a.run_id,public_directory=str(root),runtime_directory=str(runtime),
        training_output_root=str(runtime/'training'),training_admission='review_ready',
        arm_order=['action_only','paired'],updates_per_arm=8,seconds_per_arm=7200,
        total_training_seconds=14400,learning_rate=0.0001,
        note='Prospective two-arm proposal; exact user training approval required. No automatic resource extension.')
    train['components']={k:v for k,v in train['components'].items() if k in train['arm_order']}
    write(root/'TRAIN_PLAN.json',train)
    parser=read(eq/'DEPLOYMENT_CHECK_PLAN.json')['upstream_parser']
    bases={c['condition']:read(c['plan']) for c in read(source/'BATCH_PLAN.json')['cells'] if c['arm']=='frozen_base'}
    ref=next(iter(bases.values()))
    stages={'gate':[], 'development':[]}
    provenance={}
    for mode in ('desk_work','clear_floor'):
        old=repo/f'runs/920/P920-EVALFIX-v1_20261107T130916Z/evaluation/{mode}_29_frozen_base_knowledge_feedback/PLAN.json'
        p=read(old);prepared=Path(p['prepared_directory']);req=read(prepared/'request.json');obs=read(prepared/'observation.json')
        context=context_from_request(req);audit=read(prepared/'initial_audit.json')
        context.update(current_requested_mode=mode,available_measurement=audit['measurements'])
        text=next(x['text'] for x in req[1]['content'] if x['type']=='text')
        scope=text.split('## Original task and operation constraints\n',1)[1].split('\n## Current room (not an action-history snapshot)',1)[0]
        scope+='\nThe explicit current_activity_request supersedes the historical original activity. All original preservation constraints remain.'
        supplement=Path(p['prompt_supplement']['path'])
        inputs=dict(context=context,tools=schemas_from_observation(obs),scope=scope,
            images=obs['image_paths'],history=[],next_attempt=1,policy=p['input_contract'],
            additional_text=supplement.read_text()+'\nStopping never certifies whole-room usability. Plan with the available source knowledge, execute changes only when needed, then report remaining unknowns.')
        ip=root/'inputs'/mode/'inputs.json';write(ip,inputs)
        provenance[mode]={str(f):sha256(f) for f in (old,prepared/'request.json',prepared/'observation.json',prepared/'initial_audit.json',supplement)}
        p['matched_input']=dict(path=str(ip),sha256=sha256(ip));bases[mode]=p
    write(root/'INPUT_PROVENANCE.json',provenance)
    for stage,conditions,seeds,calls,gpu,wall in (
        ('gate',list(bases)[:4],[17],4,200,600),
        ('development',['desk_work','clear_floor'],[17,29],6,300,900)):
        for condition in conditions:
            for seed in seeds:
                for arm in ('frozen_base','action_only','paired'):
                    p=copy.deepcopy(bases[condition]);cid=f'{condition}__{seed}__{arm}'
                    public=root/'evaluation'/stage/cid;out=runtime/'evaluation'/stage/cid/'rollout'
                    p.update(run_id=a.run_id,cell_id=cid,public_directory=str(public),runtime_directory=str(out.parent),
                        completion_protocol='upstream-stop-v1',upstream_parser=parser,adapter=None)
                    p['inference']['seed']=seed
                    names=set(ref['code_sha256'])|set(p['code_sha256'])|{'run_learning_room_policy.py','upstream_termination_policy.py'}
                    p['code_sha256']={name:sha256(scripts/name) for name in sorted(names)}
                    p['budget'].update(model_calls=calls,tool_calls=calls,gpu_seconds_per_cell=gpu)
                    template=root/'templates'/stage/cid/'PLAN.json';write(template,p)
                    frozen={str(template):sha256(template),p['matched_input']['path']:p['matched_input']['sha256'],
                        p['evaluation']['plan']:p['evaluation']['sha256']}
                    for name in ('RESULT.json','checkpoint/manifest.json','checkpoint/house_state.json'):
                        f=Path(p['prepared_directory'])/name;frozen[str(f)]=sha256(f)
                    stages[stage].append(dict(cell_id=cid,condition=condition,seed=seed,arm=arm,
                        template=str(template),output=str(out),receipt=str(public/'rollout_receipt'),wall_seconds=wall,frozen_inputs=frozen))
    trial=dict(schema='bounded-room-learning-v1',run_id=a.run_id,created_at=datetime.now(timezone.utc).isoformat(),
        training_plan=dict(path=str(root/'TRAIN_PLAN.json'),sha256=sha256(root/'TRAIN_PLAN.json')),
        native_python=str(d/'envs/scene_designer311_native/bin/python'),stages=stages,
        gate_rule='any_trained_arm_all_four_without_new_failure',
        budget=dict(training_seconds=14400,gate_model_calls=48,gate_gpu_seconds=2400,
            development_model_calls=72,development_gpu_seconds=3600,total_gpu_subprocess_seconds=20400),
        inputs={str(scripts/n):sha256(scripts/n) for n in ('run_learning_room_policy.py','run_learning_trial_eval.py','run_matched_trajectory_batch.py','run_stage.py','prepare_learning_trial.py')},
        limitations=['Only 5 sourced principles and 8 reason positions; six training families.',
            'Only Waterloo has balanced same-state counterfactual input coverage; gate is training-exposed, one seed.',
            'Development is the existing exposed nesting family, excluded from training/retrieval; not blind or unseen-pretraining.',
            'Action-only and paired have different decision-token supervision. No equal-material mechanism claim.',
            'All room/use metrics are declared proxies; inherited bath-access failure remains a failure.'])
    write(root/'TRIAL_PLAN.json',trial)
    write(root/'STATUS.json',dict(run_id=a.run_id,status='proposal_preflight',state='proposal_preflight',
        question='Can task-matched conditional decision supervision improve complete editing trajectories beyond action training and the same-knowledge base?',
        completed=0,planned=16,updated_at=trial['created_at'],configuration_sha256=sha256(root/'TRIAL_PLAN.json'),
        input_hashes=str(root/'INPUT_PROVENANCE.json'),latest_verifiable_artifact=str(root/'TRIAL_PLAN.json'),
        failure_reason=None,next_step='Native development input restore and CPU encoding; then one exact training review.'))
    print(root,flush=True)

if __name__=='__main__':main()
