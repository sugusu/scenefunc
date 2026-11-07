"""Freeze one inference-only, training-exposed three-arm execution gate."""
import copy, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
R=Path(__file__).resolve().parents[2]; D=R.parent
now=datetime.now(timezone.utc); rid='P920-STORAGEGATE-v1_'+now.strftime('%Y%m%dT%H%M%SZ')
P=R/'runs/920'/rid; T=D/'runtime'/rid
P.mkdir(); T.mkdir()
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
old=R/'runs/920/P920-EDITSTUDY-v1_20261106T074752Z/evaluation'
s=R/'runs/920/P920-STORAGE-v2_20261105T062810Z';sp=read(s/'REPLAY_PLAN.json')
base=read(old/'Pine_frozen_base_knowledge_feedback_17/PLAN.json')
keep=['designer','scenesmith','model_directory','model_python','model_revision','model_weight_files','model_configuration_sha256','encoder','encoder_sha256','gpu_uuid','gpu_lock_paths','inference']
base={k:base[k] for k in keep}
base.update(run_id=rid,frozen_at=now.isoformat(),question='Can the existing base and two trained adapters execute a source-seen feasible storage edit under common input and feedback?',
 prepared_directory=str(T/'prepare'),stop_at_complete_single_action=True,
 task_scope=read(Path(sp['bundle'])/'task.json')['design_request'],
 budget=dict(model_calls=6,tool_calls=6,max_new_tokens=1024,max_input_tokens=24576,max_pixels=65536,seconds_per_model_call=180,gpu_seconds_per_cell=900),
 inputs=dict(manifest=ref(Path(sp['bundle'])/'manifest.json'),knowledge=ref(R/'runs/920/P920-DECISIONAUDIT-v1_20261106T154100Z/audit_v2/KNOWLEDGE_CANDIDATE.json'),reference_state=ref(s/'after_state.json'),reference_measurement=ref(s/'after_measurement.json')))
for k in ['storage_ids','living_region_xyxy_m','access_settings']:base[k]=sp[k]
# Bind all repository execution helpers, including transitive dependencies.
base['code_sha256']={p.name:ref(p)['sha256'] for p in sorted((R/'scripts/920').glob('*.py')) if p.name!='extract_swiss_development_input.py'}
plans=[]
for arm in ['prepare','frozen_base_knowledge_feedback','action_only','decision_action']:
 p=copy.deepcopy(base);p.update(cell_id=arm,public_directory=str(P/arm),runtime_directory=str(T/arm))
 p['adapter']=None if arm in ['prepare','frozen_base_knowledge_feedback'] else read(old/f'Pine_{arm}_17/PLAN.json')['adapter']
 write(P/arm/'PLAN.json',p);plans.append(str(P/arm/'PLAN.json'))
write(P/'PROTOCOL.json',dict(run_id=rid,frozen_at=now.isoformat(),plans=plans,training_updates=0,independent_test=False,
 comparison='Three existing weight sets; source/task exposed during training. One seed 17, turns use seed 17+turn.',
 stop='One preparation; one rollout per arm; no retries, no prompt/geometry repair. Infrastructure failure stops remaining arms.',
 criteria='Same 26 IDs; 23 fixed objects unchanged; cabinet identity/shape retained; 7/7 fixed and 15/15 front probes at all three body widths; positive living free-area gain; no positive AABB overlap/outside.',
 caveats=['Not a professional room-quality or generalization test.','Common new input differs from old training representation.','Only native move tool advertised; no causal comparison to historical general-edit rejection counts.','Gold reference used only for offline feasibility, never included in prompts.','Static geometric probes do not establish storage capacity, physical movement or full human usability.']))
write(P/'STATUS.json',dict(run_id=rid,question=base['question'],status='frozen_not_started',updated_at=now.isoformat(),completed_planned=[0,18],run_directory=str(T),configuration_hash=ref(P/'PROTOCOL.json')['sha256'],latest_verifiable_artifact=str(P/'PROTOCOL.json'),failure_reason=None,next_step='One native preparation, then three finite inference arms.'))
write(R/'runs/920/latest.json',dict(run_id=rid,stage='frozen_not_started',updated_at=now.isoformat(),run_directory=str(T),status_file=str(P/'STATUS.json')))
print(P)
