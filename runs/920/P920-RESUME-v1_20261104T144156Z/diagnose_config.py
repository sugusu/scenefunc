from pathlib import Path
import json,os,sys
D=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920');U=D/'resources/full_runtime_20261104/Scene-Designer';R=D/'scenefunc';run=D/'runtime/P920-RESUME-v1_20261104T144156Z'
os.environ['SCENESMITH_ROOT']=str(U.parent/'scenesmith');os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(U));from src import runtime
from Designer.replay_chained import compose_full_cfg
old=json.loads((run/'prepare/house/session_configuration.json').read_text())['effective_config'];_,new=compose_full_cfg('scene_designer',run/'resume/house')
cfg=new['furniture_agent'];cfg['asset_manager']['router']['enabled']=False
for strategy in cfg['asset_manager']['router']['strategies'].values():strategy['enabled']=False
cfg['rendering']['blender_server_port_range']=[18100,18120];cfg['collision_geometry']['server_port_range']=[18130,18150];new['experiment']['num_workers']=max(1,((os.cpu_count() or 1)+3)//4)
def normalize(v,root):
 if isinstance(v,str):return v.replace(str(root),'{PROJECT_DIRECTORY}')
 if isinstance(v,list):return [normalize(x,root) for x in v]
 if isinstance(v,dict):return {k:normalize(x,root) for k,x in v.items()}
 return v
old=normalize(old,run/'prepare/house');new=normalize(new,run/'resume/house');diff=[]
def compare(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):compare(a.get(k),b.get(k),path+'/'+k)
 elif a!=b:diff.append({'path':path,'before':a,'after':b})
compare(old,new);(run/'CONFIGURATION_DIFFERENCE.json').write_text(json.dumps({'differences_after_directory_relocation':diff,'service_constructions':0},indent=2)+'\n');print(json.dumps(diff));assert len(diff)==1 and diff[0]['path']=='/name'
