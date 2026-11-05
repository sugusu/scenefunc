"""Evaluate the frozen finite review batch; never modify, retry or repair outputs."""
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[2];pub=root/'runs/920'/a.run_id;runtime=root.parent/'runtime'/a.run_id
 plan=read(pub/'PLAN.json');audit=read(pub/'PREOUTPUT_AUDIT.json');labels_path=runtime/'EVALUATION_LABELS.json';assert sha(labels_path)==audit['labels_sha256']
 labels={r['record_id']:r for r in read(labels_path)['records']};batch=read(runtime/'generate/RESULT.json');assert batch['plan_sha256']==sha(pub/'PLAN.json') and batch['status'] in ['completed','completed_with_response_failures']
 encoded=read(runtime/'encode/RESULT.json');assert encoded['plan_sha256']==sha(pub/'PLAN.json') and encoded['pair_image_tensors_equal']
 requested={r['id']:r for r in plan['requests']};assert len(batch['records'])==len(requested)==12;assert batch['model_calls']==12;assert len(set(r['id'] for r in batch['records']))==12
 arms=read(pub/'PROTOCOL.json')['arms'];summary={};details=[];public_outputs=pub/'outputs';public_outputs.mkdir(exist_ok=False)
 for arm in arms:
  records=[r for r in batch['records'] if r['arm']==arm];valid=[r for r in records if r['status']=='completed'];errors={k:[] for k in ['Flow','Functionality','Overall']};tp=fp=fn=tn=0;exact=0;output_tokens=0
  for row in records:
   assert row['id'] in requested;req=requested[row['id']];assert row['request_sha256']==req['sha256']==sha(req['path'])
   directory=runtime/'generate'/row['id'];dest=public_outputs/row['id'];dest.mkdir();shutil.copy2(directory/'RESULT.json',dest/'RESULT.json')
   assert sha(directory/'raw_output.txt')==row['raw_sha256']
   for name in ['raw_output.txt','text_output.txt','output_ids.json']:
    shutil.copy2(directory/name,dest/name)
   output_tokens+=row['output_tokens'];detail={'id':row['id'],'record_id':row['record_id'],'arm':arm,'status':row['status']}
   if row['status']=='completed':
    answer=read(directory/'answer.json');shutil.copy2(directory/'answer.json',dest/'answer.json');target=labels[row['record_id']];detail['score_errors']={}
    for k,pred in answer['scores'].items():
     if pred is not None:
      e=abs(pred-target['scores_raw'][k]);errors[k].append(e);detail['score_errors'][k]=e
    allowed=set(['living_room','bedroom','bathroom','kitchen','balcony'])-set(target['excluded_functions']);p=set(answer['missing_functions'])&allowed;t=set(target['missing_functions'])&allowed
    tp+=len(p&t);fp+=len(p-t);fn+=len(t-p);tn+=len(allowed-(p|t));exact+=int(p==t);detail.update(predicted_missing=sorted(p),reference_missing=sorted(t),excluded=target['excluded_functions'])
   details.append(detail)
  summary[arm]={'planned':len(records),'model_calls':len(records),'valid_responses':len(valid),'response_failures':len(records)-len(valid),'score_mae':{k:sum(v)/len(v) if v else None for k,v in errors.items()},'score_coverage':{k:{'non_null':len(v),'planned':len(records)} for k,v in errors.items()},'missing_functions':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,'evaluated_labels':tp+fp+fn+tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'case_exact':exact,'valid_cases':len(valid),'planned_cases':len(records)},'input_tokens':sum(r['input_tokens'] for r in records),'output_tokens':output_tokens,'generation_seconds':sum(r['generation_seconds'] for r in records)}
 paired={k:[] for k in ['Flow','Functionality','Overall']}
 indexed={(r['record_id'],r['arm']):r for r in details}
 for record_id in labels:
  left=indexed[(record_id,arms[0])].get('score_errors',{});right=indexed[(record_id,arms[1])].get('score_errors',{})
  for k in paired:
   if k in left and k in right:paired[k].append(right[k]-left[k])
 score_delta={k:{'paired_cases':len(v),'mean_inventory_minus_image_absolute_error':sum(v)/len(v) if v else None,'improved':sum(x<0 for x in v),'same':sum(x==0 for x in v),'worse':sum(x>0 for x in v)} for k,v in paired.items()}
 result={'run_id':a.run_id,'evaluated_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(pub/'PLAN.json'),'labels_sha256':sha(labels_path),'batch_result_sha256':sha(runtime/'generate/RESULT.json'),'status':'completed','arms':summary,'paired_score_error':score_delta,'image_tensor_pairs_equal':True,'cases':6,'model_calls':12,'training_updates':0,'scene_actions':0,'limitations':['Small stratified development subset; no independent test, statistical generalization or training effect.','Inventory arm explicitly supplies function existence; missing-function agreement is a perception/information check, not professional judgement.','Score labels reflect source opinions; original brief/metric scale/individual raters unknown.','Researcher-inferred palette; predeclared conflicting14678 kitchen claim excluded, source scores unchanged.','One-image direct QA with construction-trained fixed base, not official five-view OTC evaluation.','Original source payload remains on server; this exports only our model responses and derived aggregate statistics.']}
 save(runtime/'EVALUATION_DETAILS.json',{'records':details});save(pub/'METRICS.json',result)
 shutil.copy2(Path(__file__),pub/'code'/Path(__file__).name)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
