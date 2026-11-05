"""Repair only a provenance enum in a copied rejected bundle, then cold import."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,sys
import numpy as np
R=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc');D=R.parent;c=json.loads((D/'preflight/ONO_V1_CURRENT.json').read_text());P=Path(c['public']);T=Path(c['runtime'])
sys.path.insert(0,str(R/'scripts/920'));sys.path.insert(0,str(D/'resources/full_runtime_20261104/scenesmith'))
from native_project import validate_bundle,import_project,sha256
from inspect_native_mesh_contacts import world_transform
print(datetime.now(timezone.utc).isoformat(),'copy rejected bundle; provenance enum repair only',flush=True)
source=T/'build/bundle';out=T/'input_repair';out.mkdir(exist_ok=False);target=out/'bundle';shutil.copytree(source,target)
manifest=json.loads((target/'manifest.json').read_text());task=json.loads((target/'task.json').read_text());assert task['request_provenance']=='researcher_adapted_from_sourced_conditions';task['request_provenance_detail']=task['request_provenance'];task['request_provenance']='adapted';(target/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2)+'\n');manifest['files']['task.json']={'bytes':(target/'task.json').stat().st_size,'sha256':sha256(target/'task.json')};(target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for name in manifest['files']:
 if name!='task.json':assert sha256(source/name)==sha256(target/name)
m,state,runtime,_=validate_bundle(target);restored=import_project(target,out/'restored');room=state['rooms'][m['selected_room_id']]
assert set(room['objects'])==set(restored.room.objects)
for oid,obj in room['objects'].items():assert np.allclose(world_transform(obj),restored.room.objects[oid].transform.GetAsMatrix4(),rtol=0,atol=1e-12)
assert restored.room._surface_id_counter==runtime['rooms'][m['selected_room_id']]['next_surface_id']
result={'status':'metadata_repaired_and_native_restored','created_at':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':sha256(source/'manifest.json'),'manifest_sha256':sha256(target/'manifest.json'),'bundle':str(target),'native_objects':len(room['objects']),'changed_payload_files':['task.json'],'preserved_payload_files':len(manifest['files'])-1,'geometry_and_candidate_unchanged':True,'model_calls':0,'tool_calls':0,'training_updates':0}
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
