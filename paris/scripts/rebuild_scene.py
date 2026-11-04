"""Rebuild a saved scene from licensed source meshes and fixed transforms.

This is selected-asset replay, NOT retrieval reproduction or new generation.
Raw meshes remain outside Git. Old evidence is never changed.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 cfg=read(a.run/'run_config.json');work=Path(cfg['work']);data=Path(cfg['data']);out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(work/'paris_A_inference_20261104'));import first_tools as tools
 state=read(a.checkpoint)['payload'];original_payload_sha=read(a.checkpoint)['payload_sha256']
 replacements={};records=[];hashes={};asset_proofs=[]
 source_lock={x['id']:x for x in read(ROOT/'config/assets.lock.json')['rows']}
 for r in state['asset_records']:
  old_geometry=r['geometry_path'];aid=r['asset_id_source'];f=source_lock[aid];source=data/'hssd-models'/f['path']
  assert sha(source)==f['sha256']==r['source_sha256'],aid
  if old_geometry not in replacements:
   mesh=trimesh.load(source,force='mesh');frame=r['coordinate_frame'];up=np.asarray(frame['up'],dtype=float);up/=np.linalg.norm(up);front=np.asarray(frame['front'],dtype=float);front-=up*np.dot(front,up);front/=np.linalg.norm(front)
   tf=np.eye(4);tf[:3,:3]=np.column_stack([np.cross(front,up),front,up]).T;mesh.apply_transform(tf)
   center=mesh.bounds.mean(axis=0);mesh.apply_translation([-center[0],-center[1],-mesh.bounds[0,2]])
   target=out/(r['object_id']+'.glb');mesh.export(target);replacements[old_geometry]=str(target);hashes[str(target)]=sha(target)
   old_hash=state['geometry_hashes'][old_geometry]
   asset_proofs.append({'id':aid,'original_transformed_sha256':old_hash,'rebuilt_sha256':sha(target),'byte_identical':old_hash==sha(target),'source_sha256':f['sha256']})
  r['geometry_path']=replacements[old_geometry];r['source_path']=str(source);records.append(r)
 for group in ['asset_templates','placed_instances']:
  for obj in state[group]:obj['geometry_path']=replacements[obj['geometry_path']]
 shell=work/'paris_A_inference_20261104/frozen_initial.json';state['frozen_shell']={'path':str(shell),'sha256':sha(shell)}
 state['geometry_hashes']=hashes;state['asset_records']=records
 scene=tools.shell_meshes(read(shell))
 for item in state['placed_instances']:
  obj=tools.G['SceneObject'].from_dict(item);mesh=trimesh.load(obj.geometry_path,force='mesh');mesh.apply_transform(obj.transform.GetAsMatrix4());scene.add_geometry(mesh,node_name=str(obj.object_id))
 scene.export(out/'partial_scene.glb')
 canonical=json.dumps(state,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
 (out/'checkpoint.json').write_text(json.dumps({'payload':state,'payload_sha256':hashlib.sha256(canonical.encode()).hexdigest()},ensure_ascii=False,indent=2)+'\n')
 (out/'asset_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
 proof={'mode':'selected-asset reconstruction','source_checkpoint':str(a.checkpoint),'source_packaged_checkpoint_sha256':sha(a.checkpoint),'historical_payload_sha256_before_path_tokenization':original_payload_sha,'objects':len(state['placed_instances']),'assets':asset_proofs,'scene_sha256':sha(out/'partial_scene.glb'),'note':'Paths and payload checksum relocated. Object identities and transforms retained. Re-run audit; GLB byte equality alone is not promised.'}
 (out/'reconstruction.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'objects':proof['objects'],'asset_exports_byte_identical':sum(x['byte_identical'] for x in asset_proofs),'assets':len(asset_proofs)}))
if __name__=='__main__':main()
