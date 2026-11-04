import json,sys,os
from pathlib import Path
D=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920');R=D/'scenefunc';run=D/'runtime/P920-RESUME-v2_20261104T144744Z';sys.path[:0]=[str(R/'scripts/920'),str(D/'resources/full_runtime_20261104/scenesmith')];os.environ['CUDA_VISIBLE_DEVICES']=''
from native_project import import_project,read_json
from native_checkpoint import canonical_hash
project=import_project(run/'checkpoint_after_resume',run/'roundtrip_diagnosis_copy')
stored=read_json(run/'checkpoint_after_resume/house_state.json')['rooms']['store_b'];restored=project.room.to_state_dict();record=read_json(run/'checkpoint_after_resume/session_checkpoint.json');diff=[]
def compare(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):compare(a.get(k),b.get(k),path+'/'+k)
 elif isinstance(a,list) and isinstance(b,list):
  assert len(a)==len(b)
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
 elif a!=b:diff.append({'path':path,'stored':a,'restored':b,'absolute_delta':abs(a-b) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None})
compare(stored,restored);result={'serialized_sha256':canonical_hash(stored),'restored_sha256':canonical_hash(restored),'last_tool_after_sha256':record['receipts'][-1]['after_sha256'],'differences':diff,'tools_executed':0};(run/'STATE_ROUNDTRIP_DIFFERENCE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
