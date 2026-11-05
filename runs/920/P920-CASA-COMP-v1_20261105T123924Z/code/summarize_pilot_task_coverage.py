"""Read recorded source-action bindings; do not run a model or modify data."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
from native_project import sha256
from native_checkpoint import canonical_hash

p=argparse.ArgumentParser();p.add_argument('--loss-run',type=Path,required=True);p.add_argument('--offload-plan',type=Path,required=True);p.add_argument('--comparison-plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
loss=json.loads((a.loss_run/'PLAN.json').read_text());bindings=json.loads((a.loss_run/'STATE_ACTION_BINDINGS.json').read_text());attrib=json.loads((a.loss_run/'SOURCE_ATTRIBUTION.json').read_text());enc=json.loads((a.loss_run/'ENCODING.json').read_text());pilot=json.loads(a.offload_plan.read_text());comparison=json.loads(a.comparison_plan.read_text())
assert sha256(Path(pilot['encoding']['path']))==pilot['encoding']['sha256']==sha256(a.loss_run/'ENCODING.json')
byid={x['id']:x for x in bindings};rows=[]
for source in loss['sources']:
 for i,unit in enumerate(source['units'],1):
  uid=source['id']+f'_step_{i:02d}';state=json.loads(Path(unit['state']).read_text());record=byid[uid]
  assert canonical_hash(state)==record['prestate_sha256']
  encoding=[x for x in enc['records'] if x['decision_id']==uid and x['component']=='action'];assert len(encoding)==2 and encoding[0]['target_ids']==encoding[1]['target_ids']
  rows.append({'decision_step':uid,'source_decision':source['id'],'family_id':source['family_id'],'before_furniture_instances':sum(v['object_type']=='furniture' for v in state['objects'].values()),'prestate_sha256':record['prestate_sha256'],'recorded_action':record['actual_command'],'paired_and_separate_target_ids_equal':True,'action_target_tokens':encoding[0]['target_count']})
assert len(rows)==pilot['dependent_steps']==10
result={'recorded_at':datetime.now(timezone.utc).isoformat(),'analysis_timing':'Post hoc after base final output, before remaining arms completed; no inference conditions changed.','sources':{str(f):sha256(f) for f in [a.loss_run/'PLAN.json',a.loss_run/'STATE_ACTION_BINDINGS.json',a.loss_run/'SOURCE_ATTRIBUTION.json',a.loss_run/'ENCODING.json',a.offload_plan,a.comparison_plan]},'source_decisions':len(loss['sources']),'related_action_steps':len(rows),'action_tool_counts':dict(Counter(x['recorded_action']['name'] for x in rows)),'initial_furniture_count_range':[min(x['before_furniture_instances'] for x in rows),max(x['before_furniture_instances'] for x in rows)],'source_principle_containing_reasons':sum(x['source_principle_in_reason'] for x in attrib),'all_reason_records_include_researcher_plan':all(x['researcher_execution_plan'] for x in attrib),'rows':rows,'comparison_task':'Empty native furniture catalog construction, with nine retained whole-room requirements; initial furniture count zero.','interpretation':'The approved two-update adapters were adapted on furnished-scene movement demonstrations. Casa also requires asset selection, additions and whole-room completion. This is task transfer, not a matched direct test of learned local edit quality. Do not attribute a failure solely to professional knowledge or use it as evidence that pairing cannot work.','next_data_requirement':'Retain sourced professional decisions as the paired subset; separately include legal whole-room construction/action demonstrations in both controls with identical material and budgets. Mark reconstructed plans and action-only examples explicitly. Do not invent designer reasons for every addition. New training still requires approval.','model_calls':0,'scene_edits':0,'training_updates':0,'script_sha256':sha256(Path(__file__))}
with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ['source_decisions','related_action_steps','action_tool_counts','initial_furniture_count_range','source_principle_containing_reasons']}))
