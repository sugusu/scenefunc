"""Prepare a distinct evaluation-only revision; does not authorize or launch it."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json
R=Path(__file__).resolve().parents[3]
P=Path(__file__).resolve().parent
OLD=R/'runs/920/P920-STOPDATA-v1_20261107T102628Z'
T=R.parent/'runtime'/P.name
S=R/'scripts/920'
read=lambda p:json.loads(Path(p).read_text())
hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
study=read(OLD/'STUDY_PLAN.json')
oldruntime=Path(study['runtime_directory'])
study.update(run_id=P.name,public_directory=str(P),runtime_directory=str(T),
    frozen_at=datetime.now(timezone.utc).isoformat(),training_authorized=False,
    training_result=str(OLD/'TRAINING_RESULT.json'),new_training_updates=0,
    prior_training_gpu_seconds=5512.523018927081,carried_evaluation_gpu_seconds=24.04812056897208,
    additional_evaluation_gpu_seconds_upper_bound=7175.951879431028,
    protocol='Evaluation-only continuation requires new exact approval after stopped infrastructure error. Preserve first output as rejected; at most71 new calls, unchanged72 total. First infrastructure error stops, no auto retry.',
    parent_study=dict(path=str(OLD/'STUDY_PLAN.json'),sha256=hashfile(OLD/'STUDY_PLAN.json')))
# Retain all original frozen files and add the revised executable dependency chain.
study['frozen_files'].update({str(S/n):hashfile(S/n) for n in
 ('shared_action_contract_v2.py','run_edit_policy_executed_v2.py','resume_interrupted_edit.py',
  'run_terminal_edit_evaluation_v2.py','check_existing_contract_v2.py')})
study['frozen_files'][str(OLD/'TRAINING_RESULT.json')]=hashfile(OLD/'TRAINING_RESULT.json')
for i,cell in enumerate(study['cells']):
    plan=read(cell['template']);plan.update(run_id=P.name,public_directory=str(P/'evaluation'/cell['id']))
    for n in ('shared_action_contract_v2.py','run_edit_policy_executed_v2.py','resume_interrupted_edit.py'):
        plan['code_sha256'][n]=hashfile(S/n)
    if cell['arm']!='frozen_base_knowledge_feedback':
        adapter=Path(plan.pop('adapter_pending_binding'))
        plan['adapter']=dict(directory=str(adapter),sha256={n:hashfile(adapter/n) for n in ('adapter_config.json','adapter_model.safetensors')})
        for n in plan['adapter']['sha256']:
            study['frozen_files'][str(adapter/n)]=hashfile(adapter/n)
    if i==0:
        source=oldruntime/'evaluation'/cell['id']/'rollout'
        names=['RESULT.json','final_state.json','final_observation.json','round_000/request.json',
               'round_000/model/raw_output.txt','round_000/model/RESULT.json','final_checkpoint/manifest.json']
        # Bind checkpoint contents too, not just its inventory file.
        names+= [str(f.relative_to(source)) for f in (source/'final_checkpoint').rglob('*') if f.is_file()]
        plan['interrupted_rollout']=dict(directory=str(source),sha256={n:hashfile(source/n) for n in sorted(set(names))})
        study['frozen_files'].update({str(source/n):h for n,h in plan['interrupted_rollout']['sha256'].items()})
    cell.update(template=str(P/'evaluation'/cell['id']/'TEMPLATE.json'),runtime=str(T/'evaluation'/cell['id']))
    save(Path(cell['template']),plan);study['frozen_files'][cell['template']]=hashfile(cell['template'])
save(P/'STUDY_PLAN.json',study)
save(P/'STATUS.json',dict(run_id=P.name,state='repair_preflight',updated_at=study['frozen_at'],
    runtime_directory=str(T),completed=0,planned=12,training_updates=0,
    question=study['question'],config_sha256=hashfile(P/'STUDY_PLAN.json'),
    next_step='CPU/native checks, then exact evaluation-only authorization; no training.'))
print(P)
