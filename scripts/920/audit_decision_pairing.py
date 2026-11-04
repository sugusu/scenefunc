"""CPU inspection of actual official tensors under a matched decision budget."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from decision_pairing import DecisionUnitDataset, DecisionUnitCollator, validate_triplet


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan=json.loads(a.plan.read_text()); out=a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    now=lambda:datetime.now(timezone.utc).isoformat()
    result={'started_at':now(),'status':'running','model_calls':0,'model_weights_loaded':False,
            'training_updates':0,'professional_samples_admitted':0,'plan_sha256':digest(a.plan)}
    save(out/'RESULT.json',result)
    print(now(),'begin actual-loader decision-budget audit; no model loading',flush=True)
    try:
        root,source=Path(plan['root']),Path(plan['source'])
        for name,h in plan['scripts'].items():
            assert digest(Path(__file__).parent/name)==h, name
        lock=json.loads(Path(plan['source_lock']).read_text())
        assert digest(plan['source_lock'])==plan['source_lock_sha256']
        for f in lock['files']:assert digest(source/f['source_path'])==f['sha256']
        fixture=Path(plan['fixture']);assert digest(fixture)==plan['fixture_sha256']
        original=json.loads(fixture.read_text())
        documents={'paired':deepcopy(original),'separate':deepcopy(original),'reason':deepcopy(original)}
        documents['separate']['turns'][0]['content']=''
        documents['reason']['turns'][0]['tool_calls']=[]
        validate_triplet(documents['reason'],documents['paired'],documents['separate'])
        refs={}
        for name,doc in documents.items():
            p=out/'fixtures'/name/'episode.json';save(p,doc)
            refs[name]={'path':str(p.relative_to(root)),'sha256':digest(p)}
        image_refs=[]
        for ref in original['turns'][0]['images']:
            p=root/ref;image_refs.append({'path':ref,'sha256':digest(p)})
        manifest={'schema':'scenefunc-decision-budget-audit-v1','audit_only':True,
                  'fixed_length':plan['fixed_length'],'decisions':[{'id':'engineering_paris_identity',**refs,'images':image_refs}]}
        save(out/'MANIFEST.json',manifest)
        os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'src'))
        import sft_dataset as upstream
        import torch
        from transformers import AutoProcessor
        from scene_observation import configure_image_budget
        processor=AutoProcessor.from_pretrained(plan['processor'],local_files_only=True)
        configure_image_budget(processor,plan['max_pixels'])
        datasets={arm:DecisionUnitDataset(out/'MANIFEST.json',root,processor,upstream,arm,
                    max_length=plan['fixed_length']) for arm in ['paired','separate']}
        units={arm:ds[0] for arm,ds in datasets.items()}
        collator=DecisionUnitCollator(upstream,processor.tokenizer.pad_token_id,plan['fixed_length'])
        batches={arm:collator([unit]) for arm,unit in units.items()}
        labels=lambda ex:ex['labels'][ex['labels']!=-100]
        checks={}
        for i,role in enumerate(['reason','action']):
            x,y=units['paired']['examples'][i],units['separate']['examples'][i]
            checks[role+'_supervised_ids_exact_match']=torch.equal(labels(x),labels(y))
            checks[role+'_nonempty_supervision']=len(labels(x))>0
        checks['same_reason_example']=all(torch.equal(units['paired']['examples'][0][k],v)
            for k,v in units['separate']['examples'][0].items())
        x,y=units['paired']['examples'][1],units['separate']['examples'][1]
        text=lambda ex:processor.tokenizer.decode(ex['input_ids'],skip_special_tokens=False)
        reason=original['turns'][0]['content']
        checks['reason_present_only_in_paired_action_context']=reason in text(x) and reason not in text(y)
        checks['paired_reason_prefix_masked']=reason not in processor.tokenizer.decode(labels(x),skip_special_tokens=False)
        checks['two_examples_same_five_views_each']=all(
            len(ex['image_grid_thw'])==5 for u in units.values() for ex in u['examples'])
        checks['same_dense_batch_shapes']=all(batches['paired'][k].shape==v.shape for k,v in batches['separate'].items())
        checks['same_image_tensors']=all(torch.equal(batches['paired'][k],batches['separate'][k])
            for k in ['pixel_values','image_grid_thw'])
        checks['padded_labels_and_mask_valid']=all(bool((b['labels'][b['attention_mask']==0]==-100).all()) for b in batches.values())
        checks['same_causal_target_count']=int((batches['paired']['labels'][:,1:]!=-100).sum())==int((batches['separate']['labels'][:,1:]!=-100).sum())
        checks['no_evaluation_truth_in_context']=all('EVAL_ONLY_SENTINEL' not in text(ex) for u in units.values() for ex in u['examples'])
        checks['no_cuda_initialization']=not torch.cuda.is_initialized()
        for mutation in ['action','reason','state']:
            docs=deepcopy(documents)
            if mutation=='action':docs['separate']['turns'][0]['tool_calls'][0]['arguments']['x']+=.01
            elif mutation=='reason':docs['reason']['turns'][0]['content']+=' changed'
            else:docs['separate']['task_context']['scene_description']+=' changed'
            try:validate_triplet(docs['reason'],docs['paired'],docs['separate'])
            except ValueError:checks[mutation+'_mismatch_rejected']=True
            else:checks[mutation+'_mismatch_rejected']=False
        records={}
        for arm,u in units.items():
            components=[]
            for role,ex in zip(['reason','action'],u['examples']):
                ids=labels(ex);components.append({'role':role,'input_tokens':len(ex['input_ids']),
                    'supervised_tokens':len(ids),'decoded_target':processor.tokenizer.decode(ids,skip_special_tokens=False),
                    'supervised_ids_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    'input_ids_sha256':hashlib.sha256(ex['input_ids'].numpy().tobytes()).hexdigest(),
                    'labels_sha256':hashlib.sha256(ex['labels'].numpy().tobytes()).hexdigest()})
            records[arm]={'decisions':1,'examples':2,'action_examples':1,'reason_examples':1,'image_instances':10,
                'supervised_tokens':sum(c['supervised_tokens'] for c in components),'unpadded_input_tokens':sum(c['input_tokens'] for c in components),
                'batch_shapes':{k:list(v.shape) for k,v in batches[arm].items()},'components':components}
        result.update(checks=checks,arms=records)
        save(out/'RESULT.json',result)
        assert all(checks.values()),checks
        result.update(status='passed_engineering_only',exit_code=0,
            limitation='No optimization, trained model, timing/FLOP parity, qualified professional sample or design-quality inference. Equal shapes do not prove equal runtime; unpadded conditioning differs by design.')
    except Exception as e:
        result.update(status='failed',exit_code=1,error_type=type(e).__name__,error=str(e));raise
    finally:
        result['ended_at']=now();save(out/'RESULT.json',result);print(now(),result['status'],flush=True)


if __name__=='__main__':
    main()
