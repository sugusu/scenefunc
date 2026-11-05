"""Check actual official-loader supervision for the revised corpus on CPU only."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    plan=read(a.plan);repo=Path(__file__).resolve().parents[2];source=Path(plan['official_source']);base=repo.parent
    for n,h in plan['code_sha256'].items():assert sha(Path(__file__).parent/n)==h,n
    for n,h in plan['official_source_hashes'].items():assert sha(source/n)==h,n
    for n,h in plan['processor_hashes'].items():assert sha(Path(plan['model'])/n)==h,n
    path=repo/plan['dataset']['path'];assert sha(path)==plan['dataset']['sha256'];data=read(path)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    import torch
    from transformers import AutoProcessor
    from decision_pairing import mask_role,validate_triplet
    torch.set_num_threads(4);sys.path.insert(0,str(source/'src'))
    spec=importlib.util.spec_from_file_location('curated_official_dataset',source/'src/sft_dataset.py')
    upstream=importlib.util.module_from_spec(spec);sys.modules[spec.name]=upstream;spec.loader.exec_module(upstream)
    from scene_observation import configure_image_budget
    processor=AutoProcessor.from_pretrained(plan['model'],local_files_only=True)
    configure_image_budget(processor,plan['budget']['max_pixels'])
    digest=lambda t:hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
    encoded={};records=[]
    for row in data['rows']:
        docs={};examples={}
        for image in row['images']:assert sha(Path(image['path']))==image['sha256']
        for kind,ref in row['documents'].items():
            p=base/ref['path'];assert sha(p)==ref['sha256'];doc=read(p);docs[kind]=doc
            assert doc['turns'][0]['images']==[x['path'] for x in row['images']]
            ds=upstream.StepwiseSFTDataset(p.parent,base,processor,max_length=plan['budget']['max_sequence_tokens'])
            assert len(ds)==1;raw=ds[0];ex=mask_role(raw,processor.tokenizer,ref['supervised_role']);examples[kind]=ex
            positions=(ex['labels']!=-100).nonzero().flatten();target=ex['labels'][positions]
            assert len(target)>0 and len(ex['image_grid_thw'])==5
            assert torch.equal(target,ex['input_ids'][positions]);assert bool((ex['labels'][raw['labels']==-100]==-100).all())
            text=processor.tokenizer.decode(ex['input_ids'],skip_special_tokens=False)
            assert doc['task_context']['scene_description'] in text
            assert doc['evaluation_only']['sentinel'] not in text
            original=doc['evaluation_only']['curation']['original_reason']
            if original and original!=row['reason']:assert original not in text,'Researcher plan leaked'
            if row['input_change']:
                old=doc['evaluation_only']['curation']['original_task_context'].get('design_request')
                assert old not in text,'Old operation directive leaked'
            future=doc['evaluation_only'].get('after_state') or doc['evaluation_only']['actual_after_state']['path']
            if isinstance(future,dict):future=future['path']
            assert future not in text,'Future path leaked'
            category='reason' if kind=='reason' else 'source_action' if row['reason'] else 'ordinary_action'
            record=dict(id=row['id'],family_id=row['family_id'],kind=kind,path=ref['path'],category=category,
                input_tokens=len(ex['input_ids']),supervised_tokens=len(target),input_sha256=digest(ex['input_ids']),
                target_sha256=digest(target),pixel_sha256=digest(ex['pixel_values']),
                target_text=processor.tokenizer.decode(target,skip_special_tokens=False))
            encoded[ref['path']]=dict(record,target_ids=target.tolist(),prompt_ids=raw['input_ids'][:int((raw['labels']!=-100).nonzero()[0])].tolist())
            records.append(record)
        if row['reason']:
            validate_triplet(*(docs[k] for k in ['reason','paired','separate']))
            p,s=(encoded[row['documents'][k]['path']] for k in ['paired','separate'])
            assert p['prompt_ids']==s['prompt_ids'] and p['input_sha256']!=s['input_sha256']
            batch=upstream.StepwiseCollator(processor.tokenizer.pad_token_id)([examples['paired'],examples['separate']])
            pad=batch['attention_mask']==0;assert bool(pad.any()) and bool((batch['labels'][pad]==-100).all())
        print(row['id'],len(records),'components',flush=True)
    armrows={k:[encoded[x['path']] for x in refs] for k,refs in data['arms'].items()}
    assert len(armrows['paired'])==len(armrows['separate'])==41 and len(records)==46
    for p,s in zip(armrows['paired'],armrows['separate']):
        assert p['id']==s['id'] and p['category']==s['category']
        assert p['target_ids']==s['target_ids'] and p['pixel_sha256']==s['pixel_sha256']
    totals={}
    for arm,rows in armrows.items():
        families=defaultdict(lambda:defaultdict(int))
        for r in rows:families[r['family_id']][r['category']]+=r['supervised_tokens']
        totals[arm]=dict(samples=len(rows),images=len(rows)*5,input_tokens=sum(x['input_tokens'] for x in rows),
            target_tokens=sum(x['supervised_tokens'] for x in rows),max_sequence=max(x['input_tokens'] for x in rows),
            family_target_tokens=dict(families),equal_family_token_mean_loss_mass={k:sum(v[k]/sum(v.values()) for v in families.values())/len(families) for k in ['reason','source_action','ordinary_action']})
    assert not torch.cuda.is_initialized()
    result=dict(status='curated_corpus_official_cpu_encoding_verified',ended_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(a.plan),dataset_sha256=sha(path),totals=totals,records=records,
        checks=dict(same_targets_and_pixels=True,same_prompts_at_paired_decisions=True,different_teacher_forced_prefix=True,
            prompt_padding_and_tool_receipts_not_supervised=True,old_mixed_reason_and_operation_directive_excluded=True,
            source_reason_only_targets=True,evaluation_and_future_paths_excluded=True),
        model_loaded=False,model_calls=0,training_updates=0,cuda_initialized=False,
        limitation='Structural/encoding check, not professional approval, a new native execution or training admission. Input lengths differ; equal targets do not equalize compute.')
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(totals))


if __name__=='__main__':main()
