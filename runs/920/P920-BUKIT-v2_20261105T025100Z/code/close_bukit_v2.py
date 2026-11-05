"""Package already completed Bukit v2 evidence; perform no new scene edits."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,sys
R=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc');D=R.parent
run='P920-BUKIT-v2_20261105T025100Z';P=R/'runs/920'/run;T=D/'runtime'/run
sys.path.insert(0,str(R/'scripts/920'))
from native_checkpoint import canonical_hash
from preflight_sourced_trajectory import inspect
from audit_native_review import observation
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
now=datetime.now(timezone.utc).isoformat()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert sha(src)==sha(dst)
 else:shutil.copy2(src,dst)
result=load(T/'replay/RESULT.json');steps=load(T/'replay/STEPS.json');trajectory=load(P/'FROZEN_TRAJECTORY.json');plan=load(P/'REPLAY_PLAN.json')
assert result['status']=='native_sequence_completed' and result['tool_calls']==len(steps)==9
assert all(c['passed'] for c in result['checks']) and len(result['checks'])==40
assert [s['command'] for s in steps]==plan['tool_calls']
initial=load(T/'replay/initial_state.json');after=load(T/'replay/after_state.json');last=canonical_hash(initial);checkpoints=[];endpoints=[]
for step in steps:
 assert step['before_sha256']==last
 assert step['planned_sweep']['clear_under_declared_sweep_model'] and step['actual_sweep']['clear_under_declared_sweep_model']
 c=Path(step['checkpoint']['directory']);manifest=load(c/'manifest.json')
 assert sha(c/'manifest.json')==step['checkpoint']['manifest_sha256']
 for name,row in manifest['files'].items():
  f=c/name;assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
 state=load(c/'house_state.json')['rooms'][manifest['selected_room_id']]
 assert canonical_hash(state)==step['after_sha256']
 assert canonical_hash(load(c/'house_state.json'))==step['checkpoint']['state_sha256']
 endpoint=inspect(state);assert not endpoint['positive_3d_AABB_pairs'] and not endpoint['outside_declared_wall_inner_faces']
 endpoints.append({'index':step['index'],**endpoint});last=step['after_sha256']
 checkpoints.append({'index':step['index'],'manifest_sha256':sha(c/'manifest.json'),'verified_files':len(manifest['files'])})
assert last==canonical_hash(after)
for candidate in load(P/'OBJECTIVE_REVIEW.json')['candidates']:observation(D,candidate)
receipts={}
for directory in sorted(T.glob('*_execution')):
 receipt=load(directory/'receipt.json');assert receipt['status']=='completed' and receipt['exit_code']==0
 stage=directory.name.removesuffix('_execution');receipts[stage]=receipt
 cp(directory/'receipt.json',P/'execution'/stage/'receipt.json');cp(directory/'stage.log',P/'execution'/stage/'stage.log')
for filename in ['RESULT.json','STEPS.json','HTTP_REQUESTS.json','initial_state.json','after_state.json','initial_context.json','after_context.json','initial_measurement.json','after_measurement.json']:
 cp(T/'replay'/filename,P/filename)
cp(T/'APPROACHES.json',P/'APPROACHES.json');cp(T/'review/RESULT.json',P/'REVIEW_LOADER_RESULT.json')
for label in ['initial','after']:
 cp(T/f'export_{label}/RESULT.json',P/f'EXPORT_{label.upper()}_RESULT.json')
 obs=load(T/'replay'/f'{label}_observation.json');binding=load(Path(obs['state_binding']))
 cp(Path(obs['image_paths'][0]),P/'views'/f'{label}.topdown.png')
 save(P/f'{label.upper()}_VIEWS.json',{'state_sha256':binding['state_sha256'],'binding_path':obs['state_binding'],'binding_sha256':sha(Path(obs['state_binding'])),'images':binding['images'],'native_ID_annotations':True})
for variant in ['forward','reversed']:cp(T/'review'/variant/'episode.json',P/'review_episodes'/f'{variant}.json')
checks={'status':'all_completed_evidence_verified','checked_at':now,'native_checks_passed':len(result['checks']),'checkpoint_chain':checkpoints,'actual_endpoints':endpoints,'state_and_image_bindings_verified':2,'final_cold_service_restore_performed':False,'scope':'Checks are objective and assistant-operated, not independent professional acceptance.'}
save(P/'FINAL_CHECKS.json',checks)
metric=lambda label:load(T/'replay'/f'{label}_measurement.json')
bm,am=metric('initial'),metric('after');a=load(T/'APPROACHES.json');review=load(T/'review/RESULT.json')
compact={key:[{'body_width_m':x['body_width_m'],'groups':{k:{'reachable':v['reachable_samples'],'total':v['total_samples']} for k,v in x['groups'].items()}} for x in rows] for key,rows in a['states'].items()}
summary={'run_id':run,'created_at':now,'status':'completed_mixed_outcome','family_id':'singapore_bukit_batok_interim_spaces','unique_professional_decisions_added':0,'reconstruction_variant_number':2,'native_tool_calls':9,'native_observations':2,'model_calls':0,'training_updates':0,'independent_professional_reviews':0,
 'reserved_unoccupied_area_m2':{'initial':bm['reserved_floor']['unoccupied_AABB_m2'],'after':am['reserved_floor']['unoccupied_AABB_m2'],'gain':am['reserved_floor']['unoccupied_AABB_m2']-bm['reserved_floor']['unoccupied_AABB_m2']},
 'largest_empty_rectangle_m2':{'initial':bm['floor_rectangle'],'after':am['floor_rectangle']},
 'legacy_fixed_target_counts':{label:[{'body_width_m':row['square_body_width_m'],'reachable':sum(t['reachable_from_entry'] for t in row['targets'].values()),'total':len(row['targets'])} for row in metric(label)['scenarios']] for label in ['initial','after']},
 'additive_approach_sample_counts':compact,'input_repairs':'West fixed cabinets shifted +0.02 m before the episode; not a model or professional action. Stools marked mutable in this reconstruction.',
 'asset_choice':'Inspected one wider two-seater; not selected. Retained unscaled FRANK three-seater substitute; semantic front remains a declared diagnostic assumption.',
 'cost':{'stages':{k:{n:v[n] for n in ['started_at','ended_at','elapsed_seconds','exit_code','first_log']} for k,v in receipts.items()},'sum_stage_elapsed_seconds':sum(x['elapsed_seconds'] for x in receipts.values()),'elapsed_sum_note':'Stage-duration sum, not exclusive wall time; exports overlapped.'},
 'review_data':{'unique_objective_reviews':1,'presentation_variants':2,'masked_input_tokens_per_variant':4869,'supervised_tokens_per_variant':562,'native_images_per_variant':10,'professional_review':False,'model_or_training_performed':False},
 'outcome_limits':['Still loses bookcase and right-window approach samples; left-window samples retained.','V1/V2 comparison is descriptive: both input repair and editable object set changed. Not a controlled A/B or independent project.','Square bodies, AABB routing and assumed fronts do not establish human handling, ergonomics, daylight or furniture operation.','No folded-bed articulation or whole dwelling replay. All main-room dimensions are assumptions; V1 dimension sensitivity is retained, not a V2 robustness test.','No human professional preference, positive training admission or learning/generalization claim.']}
save(P/'SUMMARY.json',summary)
packet=load(R/'runs/920/P920-BUKIT-v1_20261105T022427Z/DECISION_PACKET.json')
packet.update(run_id=run,created_at=now,unit='Second researcher reconstruction of the same sourced decision; not an independent project or historical trajectory.')
packet['source'].update(source_snapshot_sha256=sha(P/'SOURCE_PACKET_AT_FREEZE.json'))
packet['conditions']['researcher_adaptation']='Move the same sofa and two stool proxies through nine prospectively fixed native actions, retaining instances; compare floor release and effects on other approach locations.'
packet['reconstruction']={'initial_reference':'INITIAL_REFERENCE.json','initial_reference_sha256':sha(P/'INITIAL_REFERENCE.json'),'input_repairs':'INPUT_REPAIRS.json','asset_selection':'ASSET_SELECTION.json','dimensions_m':[5.2,3,2.6],'dimensions_origin':'researcher assumptions, not original survey','furniture_visuals':10,'mesh_backed_objects':2,'static_furniture_proxies':8,'object_identity_origin':'researcher instances; not source requirement to retain old furniture','scope':'Main-room reconstruction only; not full38m2 home or physical activity demonstration.'}
packet['action']={'commands':'FROZEN_TRAJECTORY.json','actual_steps':'STEPS.json','tool_calls':9,'planned_coordinates_origin':'researcher constructed; source does not contain these waypoints','historical_trajectory':False,'native_elapsed_seconds':receipts['replay']['elapsed_seconds'],'actual_exit_code':0}
packet['states']={label:{'path':f'{label}_state.json','sha256':sha(P/f'{label}_state.json'),'views':f'{label.upper()}_VIEWS.json','export':f'EXPORT_{label.upper()}_RESULT.json'} for label in ['initial','after']}
packet['outcome']={'summary':'SUMMARY.json','conditional_gain_m2':summary['reserved_unoccupied_area_m2']['gain'],'cost':'Bookcase and right-window approaches lost; original fixed diagnostics not all restored.','label':'mixed; no whole-room preference supported','objective_review':'OBJECTIVE_REVIEW.json','review_loader':'REVIEW_LOADER_RESULT.json','positive_professional_demonstration':False,'unknowns':summary['outcome_limits']}
save(P/'DECISION_PACKET.json',packet)
# Reproducible commands are exact argv arrays from the real execution receipts.
save(P/'COMMANDS.json',{'working_directory':str(R),'stages':{k:v['command'] for k,v in receipts.items()},'review_record_preparation':['python3','scripts/920/prepare_bukit_review.py','--root','..','--run',run,'--out',f'runs/920/{run}/OBJECTIVE_REVIEW.json'],'rerun_rule':'Receipts and outputs are write-once. Recovery uses saved checkpoints; a deliberate replication uses a fresh directory/run ID and preserves all historical outputs.'})
serverfiles=[{'path':str(f.relative_to(D)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(T.rglob('*')) if f.is_file()]
assert sum(x['bytes'] for x in serverfiles)<load(P/'PROTOCOL.json')['budget']['max_server_bytes']
save(P/'SERVER_FILES.json',{'created_at':now,'root':str(D),'files':serverfiles,'total_bytes':sum(x['bytes'] for x in serverfiles),'recovery':'Use replay/checkpoint_9 with its manifest and fixed runtime; full file SHA verification precedes restore. This closeout checked bytes, not a new cold-service restore. Source HSSD access and original resource terms still apply.'})
for name in ['audit_native_review.py','prepare_bukit_review.py']:assert sha(R/'scripts/920'/name)==sha(P/'code'/name)
cp(Path(__file__),P/'code/close_bukit_v2.py')
print(json.dumps({'status':summary['status'],'nine_checkpoint_files_verified':sum(x['verified_files'] for x in checkpoints),'server_files':len(serverfiles),'server_bytes':sum(x['bytes'] for x in serverfiles),'native_seconds':receipts['replay']['elapsed_seconds'],'review_seconds':receipts['review']['elapsed_seconds']}))
