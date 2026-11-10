"""Version frozen training components with measured native observations.

No new teacher decisions/actions are supplied. Retain every source component,
explicitly mark the unrebuilt units, and encode the changed subset with the
actual processor. This does not certify rollout prompt equivalence or train.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

from native_state_override import canonical
from phase_deployment import deployment_input
from phase_objective import PHASE_MODE, attach_objective_weights
from routed_multimodal import encode_component, tensor_digest
from scene_evidence import HOST_MARKER, text_slots, visible_scene_state, refresh_visible_evidence


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ['material','registry','state-ledger','processor','scene-observation','out']:
        ap.add_argument('--'+arg,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    original=[json.loads(line) for line in a.material.read_text().splitlines()]
    registry=json.loads(a.registry.read_text())
    states={r['unit_id']:r for r in json.loads(a.state_ledger.read_text())}
    known_units={r['unit_id'] for r in original}
    if set(registry)-known_units:raise ValueError('Unknown unit in observation registry')
    assets={};binding=[]
    for unit,loc in registry.items():
        export=json.loads(Path(loc['export']).read_text())
        ann=json.loads(Path(loc['annotation']).read_text())
        expected=states[unit]['state_canonical_sha256']
        if export['source_room_sha256']!=expected or ann['source_room_sha256']!=expected:
            raise ValueError('Wrong observed state: '+unit)
        if ann['status']!='native_five_view_observation_completed':raise ValueError('Incomplete observation')
        state_path=Path(loc['export']).parent/'actual_room_state.json'
        state=json.loads(state_path.read_text())
        if canonical(state)!=ann['runtime_room_sha256'] or canonical(state)!=export['runtime_room_sha256']:
            raise ValueError('State changed after annotation')
        rows=ann['images']
        if [r['view'] for r in rows]!=['topdown','south','east','west','north']:
            raise ValueError('Image order differs')
        paths=[Path(r['path']) for r in rows];images=[]
        for r,p in zip(rows,paths):
            if sha(p)!=r['sha256']:raise ValueError('Annotated image changed')
            with Image.open(p) as im:images.append(im.convert('RGB'))
        assets[unit]=(state,paths,images)
        binding.append({'unit_id':unit,'state_binding_source':states[unit]['state_source'],
                        'source_state_sha256':expected,'runtime_state_sha256':canonical(state),
                        'export_receipt':loc['export'],'export_receipt_sha256':sha(loc['export']),
                        'annotation_receipt':loc['annotation'],'annotation_receipt_sha256':sha(loc['annotation']),
                        'images':[{'path':str(p),'sha256':sha(p)} for p in paths],
                        'historical_pixel_equivalence':False})
    processor=AutoProcessor.from_pretrained(a.processor,local_files_only=True,trust_remote_code=False)
    spec=importlib.util.spec_from_file_location('native_observation',a.scene_observation)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    native.configure_image_budget(processor,65536)
    records=[];updated=[];visions={};units=[]
    for c in original:
        out=copy.deepcopy(c);unit=c['unit_id']
        if unit not in assets:
            updated.append(out)
            records.append({'unit_id':unit,'arm':c['arm'],'stage':c['stage'],
                            'status':'retained_unmodified_not_encoded_in_this_batch'})
            continue
        state,paths,images=assets[unit]
        slots=[(s,k) for s,k in text_slots(out['messages']) if s[k].startswith(HOST_MARKER)]
        if len(slots)!=1:raise ValueError('Canonical state envelope missing')
        slot,key=slots[0];evidence=json.loads(slot[key][len(HOST_MARKER):])
        old=evidence['actual_scene_state'];new=visible_scene_state(state)
        old_ids={o['object_id'] for o in old['objects']};new_ids={o['object_id'] for o in new['objects']}
        if old_ids!=new_ids:raise ValueError('Source observation identity differs')
        evidence=refresh_visible_evidence(evidence,state)
        slot[key]=HOST_MARKER+json.dumps(evidence,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        index=0
        for message in out['messages']:
            if not isinstance(message.get('content'),list):continue
            for part in message['content']:
                if part.get('type')=='image':part['image']=str(paths[index]);index+=1
        if index!=5:raise ValueError('Expected five image slots')
        # Only messages change. Targets, masks, coefficients and original
        # source IDs remain the existing training material, not new labels.
        if {k:v for k,v in out.items() if k!='messages'}!={k:v for k,v in c.items() if k!='messages'}:
            raise AssertionError('Teacher metadata altered')
        cc=dict(out,component=c['stage'])
        batch,boundary=encode_component(processor,cc,out['messages'],images)
        attach_objective_weights(batch,cc)
        vision={k:tensor_digest(batch[k]) for k in ('pixel_values','image_grid_thw')}
        if unit in visions and visions[unit]!=vision:raise AssertionError('Arms received different vision inputs')
        visions[unit]=vision
        raw=copy.deepcopy(out['messages']);suffix=PHASE_MODE[c['stage']]
        if not raw[0]['content'].endswith(suffix):raise ValueError('Wrong output-task suffix')
        raw[0]['content']=raw[0]['content'][:-len(suffix)]
        paired_missing=c['arm']=='paired_phase_v3' and c['stage']=='action' and not c['assistant_prefix']
        if paired_missing:
            prefix_status='not_matched: no original D target; live paired policy generates D'
        else:
            # Teacher D is used only as a serialization fixture. No claim of
            # self-generated decision, inference quality or policy execution.
            prefix=deployment_input(processor,raw,images,arm=c['arm'],stage=c['stage'],
                                    generated_decision=c['assistant_prefix'] or None)
            if not torch.equal(prefix['input_ids'],batch['input_ids'][:,:boundary]):
                raise AssertionError('Training/deployment encoding boundary mismatch')
            if any(not torch.equal(prefix[k],batch[k]) for k in vision):raise AssertionError('Vision prefix mismatch')
            prefix_status='exact_with_teacher_D_only_as_serialization_fixture'
        records.append({'unit_id':unit,'arm':c['arm'],'stage':c['stage'],'status':'encoded',
                        'context_tokens':boundary,'sequence_tokens':int(batch['input_ids'].shape[1]),
                        'target_tokens':int((batch['labels']!=-100).sum()),
                        'target_sha256':c['target_sha256'],'objective_mass':float(batch['token_objective_weights'].sum()),
                        'tensor_sha256':{k:tensor_digest(v) for k,v in batch.items()},
                        'prefix_check':prefix_status,'historical_visible_state_exact':old==new,
                        'full_native_rollout_prompt_equivalence':False})
        updated.append(out);del batch
        print(unit,c['arm'],c['stage'],'encoded',flush=True)
    (a.out/'components.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in updated))
    dump(a.out/'observation_bindings.json',binding)
    dump(a.out/'encoding_records.json',records)
    encoded=[r for r in records if r['status']=='encoded']
    reason_units={c['unit_id'] for c in updated if c['stage']=='decision' and c['unit_id'] in registry}
    summary={'status':'versioned_observation_subset_encoded','original_components_retained':len(updated),
             'source_units_retained':len(known_units),'new_observation_units':len(registry),
             'new_observation_distinct_states':len({b['source_state_sha256'] for b in binding}),
             'new_encoded_components':len(encoded),'paired_reason_units_with_new_observations':sorted(reason_units),
             'invented_decision_targets':0,'target_tokens_or_weights_changed':False,
             'all_source_units_retained':True,'old_material_sha256':sha(a.material),
             'new_material_sha256':sha(a.out/'components.jsonl'),'old_material_replaced':False,
             'new_observation_is_historical_pixel_equivalent':False,
             'free_model_rollout_executed':False,'model_calls':0,'training_updates':0,
             'new_scene_generations':0,'full_scene_success':None,'training_ready':False,
             'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'scope':'Same-family development observation recovery; not a new independent benchmark or method result'}
    dump(a.out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
