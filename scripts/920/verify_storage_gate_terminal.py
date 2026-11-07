"""Cold-restore one completed existing-object gate checkpoint, without edits."""
import argparse,json,os
from pathlib import Path
from datetime import datetime,timezone
from existing_project_session import create_app
from native_checkpoint import canonical_hash,rotation_roundtrip_differences
from native_project import sha256,validate_bundle
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(exist_ok=False)
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH','SCENESMITH_LLM_API_KEY','SCENESMITH_LLM_BASE_URL','GOOGLE_API_KEY','OPENAI_API_KEY']:os.environ.pop(k,None)
os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_PROXY='127.0.0.1,localhost',OPENSCENEAGENT_STUB_ASSETS='0')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
now=lambda:datetime.now(timezone.utc).isoformat()
result=dict(status='running',started_at=now(),model_calls=0,tool_calls=0,training_updates=0,plan_sha256=sha256(a.plan));client=None
print(now(),'begin storage terminal restore',flush=True)
try:
 for name,digest in plan['code_sha256'].items():
  if sha256(Path(__file__).with_name(name))!=digest:raise ValueError('Changed code '+name)
 source=Path(plan['bundle']);assert sha256(source/'manifest.json')==plan['manifest_sha256']
 manifest,state,runtime,task=validate_bundle(source);rid=manifest['selected_room_id'];original=json.loads((source/'session_checkpoint.json').read_text())
 from fastapi.testclient import TestClient
 app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
 response=client.post('/session/existing',json=dict(bundle_directory=str(source),output_directory=str(a.out/'restored')))
 write(a.out/'LOAD_REPLY.json',dict(status_code=response.status_code,body=response.json()));assert response.status_code==200
 session=server._state.session;after=session.scene.to_state_dict()
 differences=rotation_roundtrip_differences(state['rooms'][rid],after)
 assert session.receipts==original['receipts'] and len(session.receipts)==plan['expected_tool_calls']
 assert session.project.task==task and session.project.runtime==runtime
 assert session.active_environment is None and response.json()['backend_generation_started'] is False
 assert sha256(source/'manifest.json')==plan['manifest_sha256']
 write(a.out/'RESTORED_STATE.json',after)
 result.update(status='passed',identities=len(after['objects']),receipts=len(session.receipts),rotation_roundtrip_differences=differences,task_and_asset_bindings_exact=True,state_sha256=canonical_hash(after))
except Exception as exc:result.update(status='failed',error=str(exc));raise
finally:
 if client is not None:
  try:client.post('/session/close',json={})
  finally:client.close()
 result['ended_at']=now();write(a.out/'RESULT.json',result);print(now(),result['status'],flush=True)
