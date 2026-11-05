"""Freeze review inputs and separate evaluation targets before any model call."""
import argparse, ast, hashlib, json, re, shutil
from datetime import datetime, timezone
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[2];base=root.parent;pub=root/'runs/920'/a.run_id;runtime=base/'runtime'/a.run_id
 protocol_path=pub/'PROTOCOL.json';protocol=read(protocol_path);assert not (pub/'PLAN.json').exists();(runtime/'requests').mkdir(exist_ok=True);assert not list((runtime/'requests').iterdir());(pub/'code').mkdir(exist_ok=True)
 old=read(root/'runs/920/P920-WHOLE-v2_20261104T191710Z/POLICY_PLAN.json')
 for script in ['run_review_calibration.py','prepare_review_calibration.py','run_stage.py']:
  ast.parse((root/'scripts/920'/script).read_text());assert not (pub/'code'/script).exists();shutil.copy2(root/'scripts/920'/script,pub/'code'/script)
 shutil.copy2(protocol_path,pub/'SELECTION_PROTOCOL.json');timestamp=datetime.now(timezone.utc).isoformat()
 protocol['amended_before_inference_at']=timestamp;protocol['arms']=['image_inferred_legend','image_inferred_legend_inventory'];protocol['legend_provenance']='Researcher inference from release category/color presence, not an officially labeled legend. Whole Mini is development; selected cases agree with category presence.';protocol['source_conflict']='14678 critique says no kitchen but category and pink region indicate kitchen. Preserve record and scores; exclude this kitchen claim from primary missing-function agreement.';save(protocol_path,protocol)
 source=read(protocol['source_index']['path']);assert sha(protocol['source_index']['path'])==protocol['source_index']['sha256'];index={r['record_id']:r for r in source['records']}
 functions=['living_room','bedroom','bathroom','kitchen','balcony']
 legend='Researcher-inferred color key (not an official labeled legend): off-white/cream RGB(244,242,229)=living_room; pale yellow RGB(253,244,171)=bedroom; pale blue RGB(205,233,252)=bathroom; pale pink RGB(234,216,214)=kitchen; olive/light green RGB(208,216,135)=balcony. Black outlines are drawn boundaries, not a verified door/connectivity map.'
 task='Review this residential floor plan. No resident brief, metric scale, furniture, verified doors or window geometry is supplied. Give provisional 1..10 ratings for Flow, Functionality and Overall (higher is better), or null where you cannot support a rating. missing_functions lists functions not depicted among living_room, bedroom, bathroom, kitchen, balcony; absence does not mean the resident requires that function. Give at most three specific issues with visible evidence and conditional consequences. Do not invent dimensions, clearances, door connections, daylight measurements or resident preferences. Explicitly state unknowns. '+legend+' Return only a compact JSON object with exactly these keys: scores:{Flow:number_or_null,Functionality:number_or_null,Overall:number_or_null}, missing_functions:[function_names], issues:[{region:string,evidence:string,consequence:string,certainty:string}], unknowns:[strings]. Use certainty="observed" or "conditional". Keep the complete answer short enough for 768 tokens.'
 system='You assess only evidence supplied in a floor plan. Distinguish visible facts, conditional design concerns, and unknowns. Return the requested JSON only.'
 labels=[];requests=[];audits=[]
 for number,selected in enumerate(protocol['selected'],1):
  row=index[selected['record_id']];assert row['files']==selected['files']
  for entry in row['files'].values():assert sha(base/entry['path'])==entry['sha256']
  counts=dict((fn,0) for fn in functions)
  for count,fn in re.findall(r'(\d+)_(living_room|bedroom|bathroom|kitchen|balcony)',row['category']):counts[fn]=int(count)
  excluded=['kitchen'] if row['record_id']=='14678' else []
  missing=[f for f in functions if counts[f]==0 and f not in excluded]
  labels.append({'record_id':row['record_id'],'scores_raw':row['scores_raw'],'missing_functions':missing,'excluded_functions':excluded,'source_files':row['files']})
  audits.append({'record_id':row['record_id'],'missing_functions':missing,'excluded_functions':excluded,'category_inventory':counts,'category_role':'Published metadata, not a ground-truth resident brief.'})
  arms=protocol['arms'] if number%2 else list(reversed(protocol['arms']))
  for arm in arms:
   rid=f'c{number:02d}_{arm}';extra='' if arm==protocol['arms'][0] else '\nPublished room inventory: '+json.dumps(counts,sort_keys=True)+'. This is a count of depicted functions, not a resident requirement.'
   messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'image','image':str(base/row['files']['plan']['path'])},{'type':'text','text':task+extra}]}]
   path=runtime/'requests'/f'{rid}.json';save(path,messages);requests.append({'id':rid,'record_id':row['record_id'],'arm':arm,'path':str(path),'sha256':sha(path),'source_image_sha256':row['files']['plan']['sha256']})
 save(runtime/'EVALUATION_LABELS.json',{'frozen_at':timestamp,'records':labels,'scale':'raw 1..10; null predictions excluded from score MAE with explicit denominator','missing_function_rule':'Compare binary absence for five classes, exclude only predeclared conflicting14678 kitchen field; no universal requirement implied. All cases retained; failed responses retained in completion denominator.'})
 save(pub/'PREOUTPUT_AUDIT.json',{'frozen_at':timestamp,'role':'Assistant source audit, not an independent designer annotation','records':audits,'labels_path':str(runtime/'EVALUATION_LABELS.json'),'labels_sha256':sha(runtime/'EVALUATION_LABELS.json'),'original_scores_and_critique_in_model_input':False,'excluded_claim':'14678 kitchen absence conflicts with category and inferred palette; source scores remain unchanged.','unknowns':['project family and original brief','true dimensions/doors/furniture','individual annotator identities and QC','dataset training/redistribution license'],'metric_scope':'Score agreement descriptive only. Missing-function agreement is a narrow perception check, with inventory arm explicitly giving this information; cannot demonstrate professional judgement or learning.'})
 fields=['model_directory','model_weight_files','model_configuration_sha256','encoder','encoder_sha256','model_python','gpu_lock_paths','gpu_uuid'];plan={k:old[k] for k in fields};plan.update(run_id=a.run_id,frozen_at=timestamp,model_revision='63e235a0c0d6e5bc2f8f3485214df4457fd2e672',budget=protocol['budget'],requests=requests,protocol_sha256=sha(protocol_path),preoutput_audit_sha256=sha(pub/'PREOUTPUT_AUDIT.json'),code_sha256={f:sha(pub/'code'/f) for f in ['run_review_calibration.py','prepare_review_calibration.py','run_stage.py']},encoding_directory=str(runtime/'encode'),status_path=str(pub/'STATUS.json'),inference_mode='FP16 SDPA greedy, one model load, one attempt per request, seed 0; no updates',input_scope='One released colored floor plan; not official five-view Scene-Designer scene critique.')
 save(pub/'PLAN.json',plan)
 status=read(pub/'STATUS.json');status.update(status='inputs_frozen',phase='encoding_next',updated_at=timestamp,config_sha256=sha(pub/'PLAN.json'),protocol_sha256=sha(protocol_path),input_hashes={'source_index':protocol['source_index']['sha256'],'evaluation_labels':sha(runtime/'EVALUATION_LABELS.json'),'requests':{q['id']:q['sha256'] for q in requests}},next_step='Encode frozen requests, then bounded offline GPU0 inference with original targets unavailable to model.');save(pub/'STATUS.json',status)
 print(json.dumps({'requests':len(requests),'cases':len(labels),'plan_sha256':sha(pub/'PLAN.json'),'evaluation_labels_sha256':sha(runtime/'EVALUATION_LABELS.json'),'training':0}))
if __name__=='__main__':main()
