from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy,shutil
R=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc');D=R.parent;now=datetime.now(timezone.utc);run='P920-ONO-v1_'+now.strftime('%Y%m%dT%H%M%SZ');P=R/'runs/920'/run;T=D/'runtime'/run;P.mkdir();T.mkdir();(P/'code').mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
source=next(x for x in json.loads((R/'research/920/decision_packets_v1/SOURCED_DECISIONS.json').read_text())['records'] if x['decision_id']=='ono_low_furniture_entry_storage');save(P/'SOURCE_RECORD.json',source)
protocol={'run_id':run,'frozen_at':now.isoformat(),'previous_goal_turn':'progress: Bukit v2 actual execution, objective review and verified publication 2b148155c2b930170bdc23f8b9333b2762493096','family_id':'paris_19_remote_studio','allocation':'development','question':'In a declared source-guided reconstruction, can relocating the same tall storage proxy to a screened entry reduce geometric sightline obstruction while retaining its identity, envelope and approach access?','unit':'one source-guided reconstructed subdecision, not a historical trajectory or independent test project','initial_reference_finding':'Existing recovered Ono initial/final generated layouts lack the required entry tall closet. Neither is treated as a professional demonstration. No reference final layout is used by this new builder.','candidate_sequences':1,'budget':{'new_asset_downloads':0,'build_seconds':180,'preflight_seconds':120,'native_seconds':480,'native_tool_calls':4,'native_observations':2,'measure_seconds':120,'model_calls':0,'training_updates':0,'max_server_bytes':600000000},'stop_rule':'Keep failures; no new candidate sequence, dimension expansion or asset rescaling after measuring consequences.','limits':['All dimensions/placements/initial conditions are researcher assumptions; not a survey or historical before-state.','Opaque entry screen is a fixed geometric curtain substitute; no cloth/articulation/material simulation.','Window rays measure geometric visibility only, not daylight, spaciousness or aesthetics.','Same cabinet envelope does not establish usable storage capacity; physical moving and installation unknown.','Whole dwelling functions are represented by static proxies, not validated plumbing/cooking/sleeping/seating.']};save(P/'PROTOCOL.json',protocol)
structure=[]
def structural(name,x,y,z,w,h,d,facing=None,color='#EAE7E0'):
 v={'name':name,'location':[x,z,-y],'dimensions':[w,h,d],'color':color,'holes':[]}
 if facing is not None:v['facing']=facing
 structure.append(v)
structural('floor',0,0,0,5.3,.04,5.3,color='#CFC7B6');structural('ceiling',0,0,2.5,5.3,.04,5.3,color='#FFFFFF')
for name,x,y,w,f in [('north_wall',0,2.65,5.3,180),('south_wall',0,-2.65,5.3,0),('east_wall',2.65,0,5.3,270),('west_wall',-2.65,0,5.3,90),('bathroom_south_wall',-1.65,.65,2,0),('bathroom_east_wall',-.65,1.65,2,270)]:structural(name,x,y,1.25,w,2.5,.02,f)
structural('entry_door',.6,2.65,1.05,.9,2.1,.04,180);structural('bathroom_door',-.65,1.1,1.05,.7,2.1,.04,270);structural('south_window',0,-2.65,1.6,1.8,1.4,.02,0,color='#C7D7DB')
objects={};furniture=[]
def obj(name,x,y,w,h,d,yaw=0,kind='procedural_proxy',asset=None,mount=None,parent=None,z=None,color='#B6A48B',support=False,mutable=False):
 f={'name':name,'location':[x,h/2 if z is None else z,-y],'dimensions':[w,h,d],'facing':(-yaw)%360,'color':color};furniture.append(f)
 spec={'kind':kind,'object_type':'wall_mounted' if mount=='wall' else 'ceiling_mounted' if mount=='ceiling' else 'furniture','immutable':not mutable,'note':'Declared substitute; geometry only, no original identity, internal operation or functional acceptance.'}
 if asset:spec['asset_id']=asset
 if mount:spec['mount']=mount
 if parent:spec['parent']=parent
 if support:spec['support_top']=True
 objects[name]=spec
obj('bed',-1.85,-1.6,1.4,.55,1.9,color='#ADB8C4')
obj('sofa',-1.6,0,1.8,.91,.96,yaw=180,kind='mesh',asset='4979cfe2e691ac484078b909543db3f21b80462f')
obj('table',1.65,-.95,.95,.711,.945,kind='mesh',asset='5d70cc76c99acee4513ec5f7fad497d2baebca9e')
obj('chair_north',1.65,-.15,.45,.85,.45,yaw=180)
obj('chair_east',2.4,-.95,.45,.85,.45,yaw=90)
obj('high_storage',.3,-1.6,1,2.3,.6,mutable=True,color='#94795B')
objects['high_storage']['front_status']='Local +Y is the researcher-defined proxy approach face, not an inferred commercial-asset front.'
obj('kitchen_counter',1.5,-2.3,1.8,.9,.6,support=True)
obj('cooktop',1.05,-2.3,.45,.06,.45,parent='kitchen_counter',color='#454545')
obj('kitchen_sink',1.95,-2.3,.5,.08,.45,parent='kitchen_counter',color='#AFB5B8')
obj('low_fridge',2.3,.35,.6,.85,.6,color='#B7B7B7')
obj('bath_shower',-2.1,2.1,.9,2,.9,color='#B0C6CC')
obj('bath_toilet',-1.05,2.2,.5,.8,.7,yaw=180,color='#DDDDDD')
obj('bath_vanity',-2.2,1,.6,.85,.5,yaw=0,color='#BBBBBB')
obj('entry_screen',2.015,.85,1.25,2.4,.04,mount='wall',z=1.2,color='#B9B0A0')
obj('main_light',0,0,.4,.05,.4,mount='ceiling',color='#E1DFD8')
obj('bath_light',-1.65,1.65,.25,.05,.25,mount='ceiling',color='#E1DFD8')
ref={'_convention':['Axes +X east,+Y up,+Z south; width,height,depth; facing is clockwise from native +Y.','Researcher-made counterfactual; original source explains a decision but does not provide these dimensions or historical initial positions.'],'room_shape':'rectangle','room_structure':structure,'furniture':furniture};save(P/'INITIAL_REFERENCE.json',ref)
old=json.loads((R/'runs/920/P920-BUKIT-v2_20261105T025100Z/BUILD_PLAN.json').read_text());assets=old['assets']
for value in assets.values():value['rotation_basis']='Reuse fixed cached asset frame from prior accepted build; no new semantic-front claim.'
access={'grid_m':.02,'assumed_body_height_m':1.8,'square_body_widths_m':[.4,.5,.6],'entry_center_xy_m':[.6,2.65],'entry_inward_normal_xy':[0,-1],'approach_points_xy_m':{'bed_side':[-.8,-1.6],'living_edge':[-.4,0],'table_north':[1.65,.3],'kitchen_prep':[1.5,-1.6],'bath_center':[-1.5,1.2],'entry_storage_zone':[1.45,1.7]}}
commands=[{'name':'move_furniture_tool','arguments':{'object_id':'recon_high_storage_0','x':x,'y':y,'yaw':yaw}} for x,y,yaw in [(.3,.2,0),(.3,1.7,0),(.3,1.7,90),(2.2,1.7,90)]]
plan={'run_id':run,'frozen_at':now.isoformat(),'room_id':'ono_conditional_storage','room_description':'Declared counterfactual based on source conditions, not original floorplan reconstruction.','project':{'project_id':'ono_storage_reconstruction_v1','family_id':'paris_19_remote_studio','provenance':'researcher_reconstruction_from_professional_project_conditions'},'initial_reference':{'path':str(P/'INITIAL_REFERENCE.json'),'sha256':sha(P/'INITIAL_REFERENCE.json')},'assets':assets,'objects':objects,'task':{'design_request':'研究重建：低层高开间中需要保留储物，同时减少主要空间朝窗方向的遮挡；入口与其他用途仍须能够接近。只允许调整高收纳代理的位置与朝向，其他对象固定，记录收益、代价和未知项。输入的尺寸、对象及实例约束均为研究者设定。','original_request':None,'request_provenance':'researcher_adapted_from_sourced_conditions','unknowns':protocol['limits']},'source_record':str(P/'SOURCE_RECORD.json'),'source_record_sha256':sha(P/'SOURCE_RECORD.json'),'code_sha256':sha(R/'scripts/920/build_reference_room.py'),'budget':protocol['budget'],'training_authorized':False};save(P/'BUILD_PLAN.json',plan)
save(P/'FROZEN_TRAJECTORY.json',{'frozen_at':now.isoformat(),'candidate_count':1,'origin':'Researcher constructed actions, not designer history or model decisions','tool_calls':commands})
metric={'frozen_at':now.isoformat(),'observers':{'living_standing':[-.4,-.4,1.6],'dining_seated':[1.65,-.15,1.2]},'window':{'object_id':'recon_south_window','center':[0,-2.65,1.6],'width_m':1.8,'height_m':1.4,'samples_x':7,'samples_z':5},'storage_object_id':'recon_high_storage_0','storage_front_local':'+Y','storage_front_samples_x':5,'storage_front_samples_z':5,'screen_object_id':'recon_entry_screen_0','body_widths_m':[.4,.5,.6],'grid_m':.02,'eye_height_sensitivity_m':[1.2,1.6],'screen_height_sensitivity_m':[1.8,2.1,2.4],'counterfactuals':'Analytic saved-state checks only; no extra executions or independent sample counts.','criteria':'Report all sightlines, obstruction IDs, unchanged storage envelope and access without aggregate professional preference.','limits':['Opaque oriented bounding boxes, not daylight/reflectance/perception/photometry.','Sample fractions are not people, significance or whole-window quality.','Screen heights are counterfactual assumptions, not modified actual scene.']};save(P/'METRIC_DEFINITION.json',metric)
save(P/'STATUS.json',{'run_id':run,'question':protocol['question'],'updated_at':now.isoformat(),'status':'input_frozen','completed':0,'planned':4,'config_sha256':sha(P/'BUILD_PLAN.json'),'input_sha256':sha(P/'INITIAL_REFERENCE.json'),'latest_verifiable_artifact':str(P/'BUILD_PLAN.json'),'failure_reason':None,'next_step':'Build once, then fixed geometric preflight before native actions.','run_directory':str(T)})
save(R/'runs/920/latest.json',{'run_id':run,'stage':'build','updated_at':now.isoformat(),'status':'input_frozen','run_directory':str(T),'status_path':str(P/'STATUS.json')})
for name in ['build_reference_room.py','replay_sourced_decision.py','preflight_sourced_trajectory.py','measure_free_floor.py','measure_room_access.py','native_project.py','native_checkpoint.py','existing_project_session.py','inspect_native_mesh_contacts.py','run_stage.py']:shutil.copy2(R/'scripts/920'/name,P/'code'/name)
shutil.copy2(Path(__file__),P/'code/prepare_ono_v1.py');save(D/'preflight/ONO_V1_CURRENT.json',{'run_id':run,'public':str(P),'runtime':str(T)})
print(run,len(furniture),len(structure))
