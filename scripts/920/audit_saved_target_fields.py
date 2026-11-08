"""Attribute saved teacher-forced loss to target fields; no model or update.

Offsets are verified by exact tokenizer roundtrip. Tokens spanning categories
are reported as mixed, never silently assigned to a convenient field.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def field_labels(text):
    terminal = text.startswith('<done>')
    labels = ['terminal_payload' if terminal else 'syntax'] * len(text)
    opener = '<done>' if terminal else '<tool_call>'
    if not text.startswith(opener):
        raise ValueError('Unexpected target opening')
    labels[:len(opener)] = ['control_open'] * len(opener)
    for marker in ('</done>', '</tool_call>', '<|im_end|>'):
        for match in re.finditer(re.escape(marker), text):
            labels[match.start():match.end()] = ['syntax'] * len(marker)
    numeric = []
    if not terminal:
        pattern = r'"(?P<key>[^"\\]+)"\s*:\s*(?P<value>"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'
        for match in re.finditer(pattern, text):
            key, value = match['key'], match['value']
            kind = ('tool_name' if key == 'name' else 'object_asset'
                    if key.endswith('_id') else 'coordinate' if key in ('x','y','z','yaw')
                    else 'other_argument')
            start, end = match.span('value')
            labels[start:end] = [kind] * (end-start)
            if kind == 'coordinate':
                parsed = json.loads(value)
                if not isinstance(parsed, (int,float)) or not math.isfinite(parsed):
                    raise ValueError('Invalid numeric target')
                numeric.append(dict(key=key, literal=value, fractional_digits=len(value.split('.')[1]) if '.' in value else 0))
                # This is descriptive, not an assertion these digits are harmless.
                if '.' in value and 'e' not in value.lower():
                    tail = start + value.index('.') + 4
                    if tail < end:
                        labels[tail:end] = ['coordinate_after_3_decimals'] * (end-tail)
    return labels, numeric


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    args=ap.parse_args();plan=read(args.plan)
    if plan['model_calls'] or plan['training_updates'] or plan['scene_edits']:
        raise ValueError('Read-only audit required')
    for path,digest in plan['inputs'].items():
        if sha(path)!=digest:raise ValueError('Input changed: '+path)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(plan['model_directory'],local_files_only=True)
    encoding=read(plan['encoding']); raw=read(plan['raw_result'])
    records=[r for r in encoding['records'] if r['arm']=='action_only']
    if len(records)!=24:raise ValueError('Expected frozen24records')
    forwards={(r['arm'],r['id']):r for r in raw['forwards'] if r['setting']=='common'}
    if len(forwards)!=72:raise ValueError('Expected72uniquecommonforwards')
    families=sorted({r['family_id'] for r in records})
    denominators={f:sum(r['action_tokens'] for r in records if r['family_id']==f) for f in families}
    totals={arm:defaultdict(lambda:dict(tokens=0,nll_sum=0.,objective_contribution=0.)) for arm in plan['arms']}
    samples=[]
    print('begin read-only attribution of24savedtargetstreams/72savedforwards',flush=True)
    for record in records:
        if sha(record['tensor_path'])!=record['tensor_sha256']:raise ValueError('Tensor changed')
        batch=torch.load(record['tensor_path'],map_location='cpu',weights_only=True)
        ids=batch['labels'][batch['action_mask']].tolist()
        target=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        encoded=tok(target,add_special_tokens=False,return_offsets_mapping=True)
        if encoded['input_ids']!=ids:raise ValueError('Tokenizer target roundtrip changed')
        chars,numeric=field_labels(target);categories=[]
        for start,end in encoded['offset_mapping']:
            if end<=start:raise ValueError('Empty token offset')
            kinds=set(chars[start:end]);categories.append(next(iter(kinds)) if len(kinds)==1 else 'mixed_boundary')
        losses={}
        for arm in plan['arms']:
            row=forwards[arm,record['id']]
            if row['tensor_sha256']!=record['tensor_sha256']:raise ValueError('Wrong forward input')
            comp=row['components']['action'];values=comp['token_nll']
            if len(values)!=len(ids) or comp['tokens']!=len(ids):raise ValueError('Loss stream length')
            if abs(sum(values)-comp['nll_sum'])>1e-3:raise ValueError('Loss sum changed')
            losses[arm]=values
            for category,value in zip(categories,values,strict=True):
                if not math.isfinite(value) or value<0:raise ValueError('Invalid saved loss')
                bucket=totals[arm][category];bucket['tokens']+=1;bucket['nll_sum']+=value
                bucket['objective_contribution']+=value/denominators[record['family_id']]/len(families)
        samples.append(dict(id=record['id'],family_id=record['family_id'],target=target,tensor_sha256=record['tensor_sha256'],numeric_literals=numeric,
            tokens=[dict(id=i,text=target[a:b],category=c,nll={arm:losses[arm][j] for arm in plan['arms']})
                    for j,(i,(a,b),c) in enumerate(zip(ids,encoded['offset_mapping'],categories,strict=True))]))
    for arm,buckets in totals.items():
        assert sum(v['tokens'] for v in buckets.values())==1647
        for value in buckets.values():value['mean_nll']=value['nll_sum']/value['tokens']
        expected=sum(forwards[arm,r['id']]['components']['action']['nll_sum']/denominators[r['family_id']]/len(families) for r in records)
        assert abs(sum(v['objective_contribution'] for v in buckets.values())-expected)<1e-6
    if torch.cuda.is_initialized():raise ValueError('Unexpected CUDA')
    output=Path(plan['output_directory']);output.mkdir(exist_ok=True)
    result=dict(status='verified_read_only_attribution',created_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(args.plan),
        samples=samples,totals=totals,records=24,forwards_reused=72,control_tokens_per_arm=1647,
        model_calls=0,training_updates=0,scene_edits=0,cuda_initialized=False,
        limits='Posthoc teacher-forced attribution, not causal diagnosis or free-action quality. Decimal suffix after3places is descriptive; safe rounding has not been established. Terminal payload includes necessary unknowns; reporting them is not inherently wasteful. Same6family denominators as the published common action objective.')
    write(output/'RESULT.json',result)
    print(json.dumps(dict(status=result['status'],totals=totals)),flush=True)


if __name__=='__main__':main()
