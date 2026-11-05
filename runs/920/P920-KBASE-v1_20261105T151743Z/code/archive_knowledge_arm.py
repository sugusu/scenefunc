"""Archive completed knowledge-baseline evidence; never alter live inference."""
import argparse
import json
from pathlib import Path
import shutil
from native_project import sha256

p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['runtime_directory']);out=Path(plan['public_directory']);roll=root/'rollout'
result=json.loads((roll/'RESULT.json').read_text());assert result['status']=='policy_attempt_completed_pending_independent_audit'
assert json.loads((root/'rollout_execution/receipt.json').read_text())['exit_code']==0

def cp(src,dest):
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for name in ['RESULT.json','final_state.json','final_observation.json']:cp(roll/name,out/'rollout'/name)
for stage in ['rollout_execution','audit_execution','schema_review_execution']:
 if (root/stage).exists():
  for name in ['receipt.json','stage.log']:cp(root/stage/name,out/'execution'/stage/name)
for rd in sorted(roll.glob('round_*')):
 for name in ['request.json','parsed_calls.json','tool_reply.json','observation.json']:
  if (rd/name).exists():cp(rd/name,out/'rollout'/rd.name/name)
 for name in ['RESULT.json','ENCODING.json','raw_output.txt','output_ids.json']:
  if (rd/'model'/name).exists():cp(rd/'model'/name,out/'rollout'/rd.name/'model'/name)
 for name in ['receipt.json','stage.log']:
  if (rd/'model_execution'/name).exists():cp(rd/'model_execution'/name,out/'rollout'/rd.name/'model_execution'/name)
for name in ['manifest.json','house_state.json','runtime_state.json','task.json','session_checkpoint.json']:
 if (roll/'final_checkpoint'/name).exists():cp(roll/'final_checkpoint'/name,out/'checkpoint_metadata'/name)
obs=json.loads((roll/'final_observation.json').read_text());cp(Path(obs['state_binding']),out/'final_state_binding.json')
for image in obs['image_paths']:cp(Path(image),out/'views'/Path(image).name)
for stage in ['export','report']:
 folder=root/(stage+'_execution')
 for name in ['receipt.json','stage.log']:cp(folder/name,out/'execution'/stage/name)
if (root/'ACTION_SCHEMA_REVIEW.json').exists():cp(root/'ACTION_SCHEMA_REVIEW.json',out/'ACTION_SCHEMA_REVIEW.json')
for name in ['REPORT.json','REPORT.md']:cp(root/'report'/name,out/name)
cp(root/'export/RESULT.json',out/'EXPORT_RESULT.json')
cp(root/'FINAL_AUDIT.json',out/'FINAL_AUDIT.json')
cp(root/'cold_restore/RESULT.json',out/'RESTORE_RESULT.json')
for name in ['receipt.json','stage.log']:cp(root/'restore_execution'/name,out/'execution/restore'/name)
manifest={'files':[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':sha256(f)} for f in sorted(out.rglob('*')) if f.is_file() and f.name!='EVIDENCE_MANIFEST.json']}
(out/'EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'arm':plan['comparison_arm'],'files':len(manifest['files']),'bytes':sum(x['bytes'] for x in manifest['files']),'large_glb_left_on_server':True}))
