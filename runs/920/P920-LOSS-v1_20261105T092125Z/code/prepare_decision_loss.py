"""Prepare identical sourced/reconstructed material for frozen-model loss evaluation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from native_checkpoint import canonical_hash, rotation_roundtrip_differences
from native_project import sha256


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    print(datetime.now(timezone.utc).isoformat(),'prepare three-source actual loss inputs; no model',flush=True)
    root=Path(cfg['input_root']);pub=Path(cfg['public_directory'])
    for name,digest in cfg['scripts'].items():assert sha256(Path(__file__).parent/name)==digest,name
    for row in cfg['sources']:
        for x in row['locked_files']:assert sha256(Path(x['path']))==x['sha256'],x['path']
    from decision_pairing import validate_triplet,DecisionUnitDataset
    decisions=[];attribution=[];binding_checks=[]
    for source in cfg['sources']:
        for index,unit in enumerate(source['units'],1):
            read=lambda f:json.loads(Path(f).read_text())
            doc=read(unit['episode']);state=read(unit['state']);obs=read(unit['observation']);binding=read(obs['state_binding'])
            assert canonical_hash(binding['native_state'])==binding['state_sha256']
            differences=rotation_roundtrip_differences(binding['native_state'],state)
            step=read(unit['step_file'])
            if unit.get('step_array_index') is not None:step=step[unit['step_array_index']]
            assert doc['turns'][0]['tool_calls']==[step['command']]
            assert step['reply']['results'][0]['output']['success'] is True
            actual_before=step['before_sha256']
            assert canonical_hash(state)==actual_before or unit.get('restored_state')
            if unit.get('restored_state'):
                actual=read(unit['restored_state']);assert canonical_hash(actual)==actual_before
                differences+=rotation_roundtrip_differences(state,actual)
            image_refs=[]
            for old,image in zip(doc['turns'][0]['images'],binding['images']):
                path=root/old;assert path.resolve()==Path(image['path']).resolve() and sha256(path)==image['sha256']
                image_refs.append({'path':old,'sha256':image['sha256']})
            assert len(image_refs)==5
            context=json.loads(doc['task_context']['scene_description'])
            removed=context.pop('source_principle',None)
            # Original facts/current measurements remain verbatim, and are masked.
            doc['task_context']['scene_description']=json.dumps(context,ensure_ascii=False,sort_keys=True)
            doc['provenance'].update(audit_only=True,training_eligible=False,training_authorized=False)
            doc.setdefault('evaluation_only',{})['sentinel']='EVAL_ONLY_SENTINEL_LOSS'
            triplet={arm:copy.deepcopy(doc) for arm in ('reason','paired','separate')}
            triplet['reason']['turns'][0]['tool_calls']=[];triplet['separate']['turns'][0]['content']=''
            validate_triplet(triplet['reason'],triplet['paired'],triplet['separate'])
            uid=source['id']+f'_step_{index:02d}';refs={}
            for arm,episode in triplet.items():
                path=a.out/'episodes'/uid/arm/'episode.json';save(path,episode)
                refs[arm]={'path':str(path.relative_to(root)),'sha256':sha256(path)}
            decisions.append({'id':uid,**refs,'images':image_refs})
            attribution.append({'id':uid,'family_id':source['family_id'],'source_decision':source['id'],
                'source_principle_in_reason':index==1,'researcher_execution_plan':True,
                'reason':doc['turns'][0]['content'],'original_episode':unit['episode'],'original_sha256':sha256(Path(unit['episode'])),
                'removed_input_source_principle':removed,'historical_designer_trace':False})
            binding_checks.append({'id':uid,'prestate_sha256':canonical_hash(state),'action_prestate_sha256':actual_before,
                'view_binding_sha256':sha256(Path(obs['state_binding'])),'numeric_restoration_differences':differences,
                'actual_command':step['command'],'actual_reply_sha256':canonical_hash(step['reply'])})
    manifest={'schema':'scenefunc-decision-budget-audit-v1','audit_only':True,'fixed_length':cfg['budget']['max_sequence_tokens'],'decisions':decisions}
    save(a.out/'MANIFEST.json',manifest);save(pub/'SOURCE_ATTRIBUTION.json',attribution);save(pub/'STATE_ACTION_BINDINGS.json',binding_checks)
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    source=Path(cfg['official_source']);sys.path.insert(0,str(source/'src'))
    for name,digest in cfg['official_source_hashes'].items():assert sha256(source/name)==digest,name
    spec=importlib.util.spec_from_file_location('pinned_loss_dataset',source/'src/sft_dataset.py');up=importlib.util.module_from_spec(spec);sys.modules[spec.name]=up;spec.loader.exec_module(up)
    from scene_observation import configure_image_budget
    processor=AutoProcessor.from_pretrained(cfg['model_directory'],local_files_only=True)
    configure_image_budget(processor,cfg['budget']['max_pixels'])
    pad=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collator=up.StepwiseCollator(pad)
    datasets={arm:DecisionUnitDataset(a.out/'MANIFEST.json',root,processor,up,arm,cfg['budget']['max_sequence_tokens']) for arm in ('paired','separate')}
    tensors=a.out/'tensors';tensors.mkdir();records=[];arm_rows={x:[] for x in datasets};checks=[]
    tensor_hash=lambda x:hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    for i,decision in enumerate(decisions):
        examples={arm:ds[i]['examples'] for arm,ds in datasets.items()}
        for k in examples['paired'][0]:assert torch.equal(examples['paired'][0][k],examples['separate'][0][k])
        p,s=examples['paired'][1],examples['separate'][1]
        assert torch.equal(p['labels'][p['labels']!=-100],s['labels'][s['labels']!=-100])
        for k in ('pixel_values','image_grid_thw'):assert torch.equal(p[k],s[k])
        for arm,exs in examples.items():
            for role,ex in zip(('reason','action'),exs):
                labels=ex['labels'];positions=(labels!=-100).nonzero().flatten();assert positions.min()>0
                assert torch.equal(labels[positions],ex['input_ids'][positions])
                text=processor.tokenizer.decode(ex['input_ids'],skip_special_tokens=False);assert 'EVAL_ONLY_SENTINEL' not in text
                key=decision['id']+'_'+('reason_shared' if role=='reason' else 'action_'+arm)
                row={'id':key,'decision_id':decision['id'],'component':role,'arm':arm,
                    'input_tokens':len(ex['input_ids']),'target_count':len(positions),'target_ids':labels[positions].tolist(),
                    'first_target_position':int(positions[0]),'input_hash':tensor_hash(ex['input_ids']),
                    'image_hash':tensor_hash(ex['pixel_values']),'labels_hash':tensor_hash(labels)}
                arm_rows[arm].append(row)
                if role=='reason' and arm=='separate':continue
                batch=collator([ex]);path=tensors/(key+'.pt');torch.save(batch,path)
                records.append({**row,'family_id':attribution[i]['family_id'],'source_decision':attribution[i]['source_decision'],
                    'source_principle_in_reason':attribution[i]['source_principle_in_reason'],
                    'tensor_path':str(path),'tensor_sha256':sha256(path),'tensor_shapes':{k:list(v.shape) for k,v in batch.items()}})
        # Both explanation and action supervision currently use the same prompt.
        r_ex=examples['separate'][0];a_ex=examples['separate'][1]
        rp=int((r_ex['labels']!=-100).nonzero()[0]);sp=int((a_ex['labels']!=-100).nonzero()[0])
        checks.append({'id':decision['id'],'same_reason_action_prompt':rp==sp and torch.equal(r_ex['input_ids'][:rp],a_ex['input_ids'][:sp]),
                       'reason_first_token':int(r_ex['labels'][rp]),'action_first_token':int(a_ex['labels'][sp])})
        print('encoded',decision['id'],flush=True)
    totals={arm:{'examples':len(rows),'target_tokens':sum(x['target_count'] for x in rows),'input_tokens':sum(x['input_tokens'] for x in rows),
                 'sum_sequence_lengths_squared':sum(x['input_tokens']**2 for x in rows),'image_instances':5*len(rows)} for arm,rows in arm_rows.items()}
    assert totals['paired']['target_tokens']==totals['separate']['target_tokens']
    result={'status':'encoded_without_model','ended_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha256(a.plan),
        'manifest_sha256':sha256(a.out/'MANIFEST.json'),'records':records,'arms':totals,'arm_rows':arm_rows,
        'same_prompt_different_first_target_diagnostic':checks,'source_decisions':3,'dependent_steps':len(decisions),
        'actual_unique_forward_records':len(records),'shared_reason_forwards_reused_for_both_arms':len(decisions),
        'cuda_initialized':torch.cuda.is_initialized(),'optimizer_updates':0,'training_authorized':False,
        'scope':'Actual pinned dataset/collator inputs for observational loss; source boundaries and same-prompt supervision ambiguity retained, not a training admission.'}
    assert not torch.cuda.is_initialized();save(a.out/'RESULT.json',result);save(pub/'ENCODING.json',result)
    print(json.dumps({'status':result['status'],'arms':totals,'unique_records':len(records)}),flush=True)


if __name__=='__main__':main()
