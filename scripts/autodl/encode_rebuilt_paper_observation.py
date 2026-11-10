"""Encode the actual restored observation for future free-model deployment.

Only source task/context and measured current state enter requests. No teacher
action/reason is loaded into the inference API. This does not promote newly
rendered images into the historical training corpus or evaluate a policy.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

from phase_deployment import deployment_input
from phase_objective import PHASE_MODE
from routed_multimodal import tensor_digest
from scene_evidence import HOST_MARKER,visible_scene_state,text_slots

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['material','annotation','export','processor','source','original-topdown','out']:
  ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 import numpy as np
 import torch
 from PIL import Image
 from transformers import AutoProcessor
 torch.set_num_threads(1)
 unit='paper_table_modes_build_07'
 c=next(json.loads(line) for line in a.material.read_text().splitlines()
        if json.loads(line)['unit_id']==unit and json.loads(line)['arm']=='separate_phase_v3')
 messages=copy.deepcopy(c['messages'])
 suffix=PHASE_MODE['action'];assert messages[0]['content'].endswith(suffix)
 messages[0]['content']=messages[0]['content'][:-len(suffix)]
 # Canonical host state is replaced with the actual exported runtime state.
 state=json.loads((a.export/'actual_room_state.json').read_text())
 annotation=json.loads(a.annotation.read_text())
 assert annotation['status']=='native_five_view_observation_completed'
 slots=[(s,k) for s,k in text_slots(messages) if s[k].startswith(HOST_MARKER)]
 assert len(slots)==1
 slot,key=slots[0];evidence=json.loads(slot[key][len(HOST_MARKER):])
 old_visible=evidence['actual_scene_state'];new_visible=visible_scene_state(state)
 changes=[]
 def differences(x,y,p='$'):
  if isinstance(x,dict) and isinstance(y,dict):
   for k in x.keys()-y.keys():changes.append({'path':p+'.'+k,'kind':'source_only_field','source':x[k]})
   for k in y.keys()-x.keys():changes.append({'path':p+'.'+k,'kind':'new_visible_field','runtime':y[k]})
   for k in x.keys()&y.keys():differences(x[k],y[k],p+'.'+k)
  elif isinstance(x,list) and isinstance(y,list):
   if len(x)!=len(y):raise ValueError('State list differs')
   if x and all(isinstance(v,dict) and 'object_id' in v for v in x+y):
    xx={v['object_id']:v for v in x};yy={v['object_id']:v for v in y}
    if len(xx)!=len(x) or xx.keys()!=yy.keys():raise ValueError('Object identity set differs')
    if list(xx)!=list(yy):changes.append({'path':p,'kind':'object_order','source':list(xx),'runtime':list(yy)})
    for oid in xx:differences(xx[oid],yy[oid],p+'['+oid+']')
   else:
    for i,(aa,bb) in enumerate(zip(x,y)):differences(aa,bb,f'{p}[{i}]')
  elif x!=y:changes.append({'path':p,'kind':'value','source':x,'runtime':y})
 differences(old_visible,new_visible)
 # Record differences rather than editing frozen evidence or asserting that
 # a small numeric difference implies image equivalence.
 evidence['actual_scene_state']=new_visible
 slot[key]=HOST_MARKER+json.dumps(evidence,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
 paths=[Path(r['path']) for r in annotation['images']];images=[]
 for row,p in zip(annotation['images'],paths):
  assert sha(p)==row['sha256']
  with Image.open(p) as im:images.append(im.convert('RGB'))
 image_slot=0
 for message in messages:
  if not isinstance(message.get('content'),list):continue
  for part in message['content']:
   if part.get('type')=='image':part['image']=str(paths[image_slot]);image_slot+=1
 assert image_slot==5
 (a.out/'live_request.json').write_text(json.dumps(messages,ensure_ascii=False,indent=2)+'\n')
 # The five images are new observations. Only the archived topdown exists for
 # pixel comparison; absent old oblique images are not invented.
 with Image.open(a.original_topdown) as im:old=np.asarray(im.convert('RGB'),dtype=np.int16)
 new=np.asarray(images[0],dtype=np.int16)
 assert old.shape==new.shape
 diff=np.abs(old-new)
 pixel={'shape':list(old.shape),'rgb_exact':bool(np.array_equal(old,new)),
        'different_pixel_fraction':float(np.any(diff!=0,axis=2).mean()),
        'mean_absolute_rgb_difference_0_to_255':float(diff.mean()),
        'max_absolute_channel_difference':int(diff.max()),
        'original_sha256':sha(a.original_topdown),'new_sha256':sha(paths[0]),
        'perceptual_or_semantic_equivalence_claimed':False}
 processor=AutoProcessor.from_pretrained(a.processor,local_files_only=True,trust_remote_code=False)
 spec=importlib.util.spec_from_file_location('source_observation_encoder',a.source/'src/scene_observation.py')
 native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 native.configure_image_budget(processor,65536)
 rows=[]
 for arm,stage in [('separate_phase_v3','action'),('paired_phase_v3','decision')]:
  batch=deployment_input(processor,messages,images,arm=arm,stage=stage)
  assert not any(k in batch for k in ['labels','token_objective_weights','action_mask','decision_mask'])
  rows.append({'arm':arm,'stage':stage,'input_tokens':int(batch['input_ids'].shape[1]),
    'tensor_sha256':{k:tensor_digest(v) for k,v in batch.items() if isinstance(v,torch.Tensor)},
    'image_grid_thw':batch['image_grid_thw'].tolist()})
  del batch
 assert rows[0]['tensor_sha256']['pixel_values']==rows[1]['tensor_sha256']['pixel_values']
 assert rows[0]['tensor_sha256']['image_grid_thw']==rows[1]['tensor_sha256']['image_grid_thw']
 try:deployment_input(processor,messages,images,arm='paired_phase_v3',stage='action')
 except ValueError as e:
  assert 'requires its actual generated decision' in str(e);paired_without_D_rejected=True
 else:raise AssertionError('Paired action silently accepted a missing generated decision')
 result={'status':'rebuilt_observation_encoded_for_inference','unit_id':unit,
   'source_material_sha256':sha(a.material),'source_room_sha256':annotation['source_room_sha256'],
   'runtime_room_sha256':annotation['runtime_room_sha256'],'state_field_differences':changes,
   'historical_request_feature_equivalence':not changes,
   'new_visible_field_count':sum(r['kind']=='new_visible_field' for r in changes),
   'topdown_pixel_comparison':pixel,'encoded_inputs':rows,'five_views_equal_across_arms':True,
   'teacher_reason_or_action_passed_to_deployment':False,
   'paired_action_without_generated_D_rejected':paired_without_D_rejected,
   'source_unit_has_D_supervision':False,'historical_complete_view_units_remain':9,
   'historical_total_units':24,'new_observation_set_count':1,'training_corpus_modified':False,
   'model_calls':0,'optimizer_updates':0,'free_model_trajectory_executed':False,
   'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (a.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['encoded_inputs','state_field_differences']},ensure_ascii=False))

if __name__=='__main__':main()
