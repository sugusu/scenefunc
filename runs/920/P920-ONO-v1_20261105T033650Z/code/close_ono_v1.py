"""Check and package actual Ono evidence; no new scene or model execution."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,sys
R=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc');D=R.parent;c=json.loads((D/'preflight/ONO_V1_CURRENT.json').read_text());P=Path(c['public']);T=Path(c['runtime']);run=c['run_id'];now=datetime.now(timezone.utc).isoformat()
sys.path.insert(0,str(R/'scripts/920'))
from native_checkpoint import canonical_hash
from preflight_sourced_trajectory import inspect
from audit_native_review import observation
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cp(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 if b.exists():assert sha(a)==sha(b)
 else:shutil.copy2(a,b)
ref=lambda p:{'path':str(p.relative_to(D)),'sha256':sha(p)}
print(now,'check Ono actual chain and source-to-consequence delivery',flush=True)
result=load(T/'replay/RESULT.json');steps=load(T/'replay/STEPS.json');assert result['status']=='native_sequence_completed' and result['tool_calls']==4
assert len(result['checks'])==20 and all(x['passed'] for x in result['checks']);assert [x['command'] for x in steps]==load(P/'FROZEN_TRAJECTORY.json')['tool_calls']
initial=load(T/'replay/initial_state.json');after=load(T/'replay/after_state.json');previous=canonical_hash(initial);chain=[]
for step in steps:
 cpdir=Path(step['checkpoint']['directory']);manifest=load(cpdir/'manifest.json');house=load(cpdir/'house_state.json');state=house['rooms'][manifest['selected_room_id']]
 assert sha(cpdir/'manifest.json')==step['checkpoint']['manifest_sha256'] and canonical_hash(house)==step['checkpoint']['state_sha256']
 assert step['before_sha256']==previous and canonical_hash(state)==step['after_sha256'];previous=step['after_sha256']
 for name,row in manifest['files'].items():assert sha(cpdir/name)==row['sha256'] and (cpdir/name).stat().st_size==row['bytes']
 endpoint=inspect(state);assert not endpoint['positive_3d_AABB_pairs'] and not endpoint['outside_declared_wall_inner_faces']
 assert step['actual_sweep']['clear_under_declared_sweep_model']
 chain.append({'index':step['index'],'manifest_sha256':sha(cpdir/'manifest.json'),'verified_files':len(manifest['files']),'endpoint':endpoint})
assert previous==canonical_hash(after)
for label in ['initial','after']:
 obs=load(T/'replay'/f'{label}_observation.json');candidate={'state':ref(T/'replay'/f'{label}_state.json'),'observation':ref(T/'replay'/f'{label}_observation.json'),'binding':ref(Path(obs['state_binding'])),'context':ref(T/'replay'/f'{label}_context.json')};observation(D,candidate);binding=load(Path(obs['state_binding']));save(P/f'{label.upper()}_VIEWS.json',{'state':candidate['state'],'binding':candidate['binding'],'images':binding['images'],'native_ID_annotations':True});cp(Path(obs['image_paths'][0]),P/'views'/f'{label}.topdown.png')
for name in ['RESULT.json','STEPS.json','HTTP_REQUESTS.json','initial_state.json','after_state.json','initial_context.json','after_context.json','initial_measurement.json','after_measurement.json']:cp(T/'replay'/name,P/name)
cp(T/'MEASUREMENTS.json',P/'MEASUREMENTS.json')
for label in ['initial','after']:cp(T/f'export_{label}/RESULT.json',P/f'EXPORT_{label.upper()}_RESULT.json')
receipts={}
for directory in sorted(T.glob('*_execution')):
 d=load(directory/'receipt.json');assert d['status'] in ['completed','failed'];name=directory.name.removesuffix('_execution');receipts[name]=d
 for f in ['receipt.json','stage.log']:cp(directory/f,P/'execution'/name/f)
assert receipts['build']['exit_code']==receipts['preflight']['exit_code']==1
assert all(d['exit_code']==0 for name,d in receipts.items() if name not in ['build','preflight'])
m=load(T/'MEASUREMENTS.json');b,e=m['states']['initial'],m['states']['after'];assert b['storage_envelope_m3']==e['storage_envelope_m3']==1.38
assert all(initial['objects'][key]==after['objects'][key] for key in initial['objects'] if key!='recon_high_storage_0')
fixed=lambda s:[{'body_width_m':v['square_body_width_m'],'reachable':[k for k,x in v['targets'].items() if x['reachable_from_entry']],'unreachable':[k for k,x in v['targets'].items() if not x['reachable_from_entry']]} for v in s['access']]
assert fixed(b)==fixed(e)
views=lambda s:{name:{kind:{k:v[k] for k in ['clear_rays','total_rays','first_occluder_counts']} for kind,v in row.items() if kind in ['window','storage_front']} for name,row in s['views'].items()}
summary={'run_id':run,'created_at':now,'status':'completed_conditional_geometric_gain_partial_source_implementation','source_decision':'ono_low_furniture_entry_storage','family_id':'paris_19_remote_studio','allocation':'development','native_actions':4,'native_observations':2,'model_calls':0,'training_updates':0,'objects':25,'visual_furniture_and_fixture_objects':16,'mesh_assets':2,'static_proxies':14,'views':{'initial':views(b),'after':views(e)},'storage_external_envelope_m3':{'initial':1.38,'after':1.38,'internal_capacity_verified':False},'access':{'initial':fixed(b),'after':fixed(e)},'analytic_sensitivity':{label:[{'screen_height_m':row['screen_height_m'],'eye_height_m':row['eye_height_m'],'views':{name:{key:v[key]['clear_rays'] for key in ['window','storage_front']} for name,v in row['views'].items()}} for row in s['sensitivity']] for label,s in m['states'].items()},'interpretation':{'verified_local_gain':'Selected geometric aperture rays increase while named approach outcomes and the storage instance/envelope are retained.','source_implementation_gap':'Screen hides the cabinet from the declared dining station, but most sampled front points remain visible from living station.','preexisting_limit':'Kitchen prep fixed point remains unreachable for all three square widths; other point failures remain.','conditional_objective_demonstration':True,'overall_professional_recommendation':False,'daylight_or_spaciousness_gain_verified':False,'whole_dwelling_function_verified':False,'independent_professional_review':False,'training_admitted':False},'execution':{k:{q:v[q] for q in ['started_at','ended_at','elapsed_seconds','exit_code','first_log']} for k,v in receipts.items()},'sum_stage_elapsed_seconds':sum(v['elapsed_seconds'] for v in receipts.values()),'time_note':'Sum includes two failed setup stages; exports overlapped, so not exclusive wall time.','limits':m['limits']+load(P/'PROTOCOL.json')['limits']}
save(P/'SUMMARY.json',summary)
save(P/'FINAL_CHECKS.json',{'checked_at':now,'status':'completed','native_assertions':20,'native_checkpoint_chain':chain,'state_context_and_image_bindings':2,'analytic_ray_checks':m['analytic_ray_checks_passed'],'final_cold_service_restore_performed':False,'source_of_acceptance':'Assistant-operated objective program checks, not independent professional review.'})
packet={'schema':'scenefunc-sourced-decision-packet-v1','decision_id':'ono_low_furniture_entry_storage','run_id':run,'created_at':now,'family_id':'paris_19_remote_studio','split':'development','unit':'One source-guided reconstructed subdecision; original project historical initial state/action sequence unavailable.','source':{'record':'SOURCE_RECORD.json','record_sha256':sha(P/'SOURCE_RECORD.json'),'supported_scope':'Low main-area furniture and tall storage located in a curtained entry; no source support for our coordinates or proxy geometry.'},'conditions':{'source_conditions':'Compact low-ceiling studio balancing visual openness and storage.','researcher_conditions':'Assumed 5.3m square, 2.5m high studio; fixed low furniture, bathroom/kitchen proxies and an opaque entry screen. Cabinet starts near the window as a counterfactual task.','raw_resident_request':None,'reference_warning':'Recovered initial/final generated reference has no target entry closet; it was audited but not adopted as a professional example.'},'objects':{'edited':'recon_high_storage_0','protected':'All other 24 objects, target instance identity and bounding dimensions','source_identity_claim':False,'cabinet_is_static_proxy':True},'action':{'plan':'FROZEN_TRAJECTORY.json','actual':'STEPS.json','execution_origin':'researcher reconstructed sequence, not original designer or model trace','tool_calls':4,'success_receipts':4},'states':{label:{'path':f'{label}_state.json','sha256':sha(P/f'{label}_state.json'),'views':f'{label.upper()}_VIEWS.json','geometry_export':f'EXPORT_{label.upper()}_RESULT.json'} for label in ['initial','after']},'outcome':summary['interpretation'],'metrics':'SUMMARY.json','full_measurements':'MEASUREMENTS.json','scope_for_supervision':{'knowledge':True,'executed_source_conditioned_program':True,'objective_local_benefit_label':True,'full_source_decision_implemented':False,'professional_preference_label':False,'historical_designer_trajectory':False,'meaning':'A conditional construction/diagnostic record can exist without independent preference labels. This episode remains a development case; no training admission or approval is implied.'},'rights':'Attributed brief paraphrase and own reconstruction evidence; source text/image training reuse not established. HSSD meshes/checkpoints stay on server under source terms.','training_authorized':False}
save(P/'DECISION_PACKET.json',packet)
save(P/'COMMANDS.json',{'cwd':str(R),'commands':{k:v['command'] for k,v in receipts.items()},'repair':'Only task enum normalized in a copied rejected bundle; original failures retained.','reproduction':'Use frozen plans/code and fresh output paths; do not overwrite this run or repeat completed actions on the final checkpoint.'})
files=[{'path':str(f.relative_to(D)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(T.rglob('*')) if f.is_file()];total=sum(x['bytes'] for x in files);assert total<load(P/'PROTOCOL.json')['budget']['max_server_bytes'];save(P/'SERVER_FILES.json',{'root':str(D),'files':files,'total_bytes':total,'recovery':'Editable final bundle is replay/checkpoint_4. Verify all manifest bytes with the pinned runtime before recovery. GLBs are previews; this check did not cold-restore final native service.'})
cp(Path(__file__),P/'code/close_ono_v1.py')
print(json.dumps({'status':summary['status'],'checkpoint_payload_checks':sum(x['verified_files'] for x in chain),'server_files':len(files),'server_bytes':total,'native_seconds':receipts['replay']['elapsed_seconds'],'raw_failures_retained':2}),flush=True)
