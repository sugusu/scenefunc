"""Load the new shared/action-reason corpus with the pinned official dataset; no model."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();plan=read(a.plan)
    for name,h in plan['code_sha256'].items():assert sha(Path(__file__).parent/name)==h,name
    source=Path(plan['official_source']);root=Path(plan['input_root'])
    for name,h in plan['official_source_hashes'].items():assert sha(source/name)==h,name
    for name,h in plan['processor_hashes'].items():assert sha(Path(plan['model'])/name)==h,name
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    import torch
    from transformers import AutoProcessor
    from decision_pairing import mask_role
    torch.set_num_threads(4)
    sys.path.insert(0,str(source/'src'))
    spec=importlib.util.spec_from_file_location('official_construction_dataset',source/'src/sft_dataset.py')
    up=importlib.util.module_from_spec(spec);sys.modules[spec.name]=up;spec.loader.exec_module(up)
    from scene_observation import configure_image_budget
    processor=AutoProcessor.from_pretrained(plan['model'],local_files_only=True)
    configure_image_budget(processor,plan['budget']['max_pixels'])
    data=read(Path(plan['dataset']));rows={};examples={}
    for arm,refs in data['arms'].items():
        for ref in refs:
            path=root/ref['path'];assert sha(path)==ref['sha256']
            if ref['path'] in rows:continue
            ds=up.StepwiseSFTDataset(path.parent,root,processor,max_length=plan['budget']['max_sequence_tokens'])
            assert len(ds)==1
            ex=mask_role(ds[0],processor.tokenizer,ref['supervised_role'])
            assert len(ex['input_ids'])<=plan['budget']['max_sequence_tokens']
            assert len(ex['image_grid_thw'])==5
            targets=ex['labels'][ex['labels']!=-100]
            assert len(targets)>0
            text=processor.tokenizer.decode(ex['input_ids'],skip_special_tokens=False)
            assert 'EVAL_ONLY_SOURCE_CONSTRUCTION' not in text
            h=lambda x:hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
            rows[ref['path']]={'role':ref['supervised_role'],'input_tokens':len(ex['input_ids']),
                'supervised_tokens':len(targets),'target_ids':targets.tolist(),'input_sha256':h(ex['input_ids']),
                'target_sha256':h(targets),'image_sha256':h(ex['pixel_values']),
                'image_instances':len(ex['image_grid_thw']),'evaluation_metadata_in_input':False}
            examples[ref['path']]=ex
    arm_rows={arm:[rows[x['path']] for x in refs] for arm,refs in data['arms'].items()}
    assert len(rows)==plan['budget']['unique_examples']==11
    assert len(arm_rows['paired'])==len(arm_rows['separate'])==10
    for p,s in zip(arm_rows['paired'],arm_rows['separate']):
        assert p['target_ids']==s['target_ids'] and p['image_sha256']==s['image_sha256']
    shared=[x for x in data['rows'] if x['ordinary_action_only']]
    assert len(shared)==8 and sum(x['source_principle_target'] for x in data['rows'])==1
    source_row=data['rows'][6];pp=source_row['documents']['paired']['path'];sp=source_row['documents']['separate']['path']
    assert rows[pp]['input_sha256']!=rows[sp]['input_sha256']
    for x in [pp,sp]:
        batch=up.StepwiseCollator(processor.tokenizer.pad_token_id)([examples[x]])
        assert batch['input_ids'].shape[0]==1 and bool((batch['labels'][batch['attention_mask']==0]==-100).all())
    assert not torch.cuda.is_initialized()
    result={'status':'new_construction_candidates_encoded_without_model',
        'ended_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),
        'dataset_sha256':sha(Path(plan['dataset'])),'unique_examples':len(rows),'examples_per_arm':10,
        'shared_action_only_steps':8,'source_decisions':1,
        'supervised_tokens_per_arm':{arm:sum(x['supervised_tokens'] for x in rr) for arm,rr in arm_rows.items()},
        'input_tokens_per_arm':{arm:sum(x['input_tokens'] for x in rr) for arm,rr in arm_rows.items()},
        'rows':rows,'arm_target_sequences_equal':True,'source_action_context_differs':True,
        'model_loaded':False,'model_forward_calls':0,'optimizer_updates':0,'cuda_initialized':False,
        'training_admitted':False,'training_authorized':False,
        'limits':['Actual dataset/processor/mask/collator only; no new Trainer or optimization path.',
                  'Shared target counts do not imply equal input tokens, FLOPs or professional quality.',
                  'One source decision; ordinary additions have no fabricated professional rationale.']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','supervised_tokens_per_arm','input_tokens_per_arm']}))


if __name__=='__main__':main()
