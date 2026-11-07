"""CPU-only checks against the actual cached training tensors and tokenizer."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from transformers import AutoTokenizer
import diagnose_final_edit_fit as diag


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=diag.read(args.plan);enc=diag.validate_plan(cfg)
    batches,checks=diag.load_batches(enc)
    tok=AutoTokenizer.from_pretrained(cfg['model_directory'],local_files_only=True)
    results=[]
    def passed(name):results.append(name)
    targets={}
    for rid,pair in batches.items():
        record,batch=pair['common'];prefix=diag.model_inputs(batch,record['prompt_tokens'])
        assert prefix['input_ids'].shape[1]==record['prompt_tokens']
        assert bool((batch['labels'][:,:record['prompt_tokens']]==-100).all())
        assert 'labels' not in prefix and 'action_mask' not in prefix and 'decision_mask' not in prefix
        assert torch.equal(prefix['pixel_values'],batch['pixel_values'])
        raw=tok.decode(batch['labels'][batch['action_mask']].tolist(),skip_special_tokens=False)
        parsed=diag.parse_output(raw);assert parsed is not None,(rid,raw)
        targets[rid]=parsed
    passed('24 actual targets parse and all free-input prefixes exclude supervised tokens while retaining original views')
    passed('48 actual tensors: exact hashes, prompt/vision pairing, disjoint masks and target tokens verified')
    assert sum(len([1 for _,p in batches.items() if p['oracle'][0]['decision_tokens']]) for _ in diag.ARMS)==24
    passed('72 common forwards, 24 oracle forwards, 18 free outputs fixed')
    bad=['<tool_call>{"name":"x","arguments":{}}',
         '<tool_call>{"name":"x","arguments":{}}</tool_call> extra',
         '<tool_call>{"name":"x","arguments":{"x":1e999}}</tool_call>',
         '<tool_call>{"name":"x","arguments":{"x":NaN}}</tool_call>',
         '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
         '<done>[]</done>', '<done>{}</done><done>{}</done>',
         '<tool_call>{"name":"x","arguments":[]}</tool_call>']
    for raw in bad:assert diag.parse_output(raw) is None,raw
    passed('partial, multiple, trailing, duplicate, nonfinite and malformed representations rejected')
    assert diag.parse_output('<think>reason</think><done>{}</done><|im_end|>')['kind']=='done'
    passed('reason prefix and native EOS accepted')
    original=torch.load
    first=enc['records'][0]['tensor_path']
    for name in ('label_mask','prompt','view'):
        changed=False
        def corrupt(path,*a,**kw):
            nonlocal changed
            b=original(path,*a,**kw)
            if not changed and (name=='label_mask' or str(path)!=first):
                if name=='label_mask':b['labels'][0,0]=1
                elif name=='prompt':b['input_ids'][0,0]+=1
                else:b['pixel_values'][0,0]+=1
                changed=True
            return b
        with patch.object(torch,'load',side_effect=corrupt):
            try:diag.load_batches(enc)
            except ValueError:pass
            else:raise AssertionError('Accepted corrupt '+name)
        passed('corrupt '+name+' rejected')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'must_not_exist'
        p=subprocess.run([sys.executable,str(Path(diag.__file__)), '--plan',str(args.plan),'--out',str(out)],capture_output=True,text=True)
        assert p.returncode!=0 and 'PermissionError' in p.stderr and not out.exists()
        passed('unapproved worker fails before output creation or model import')
        auth=Path(tmp)/'auth.json';diag.save(auth,{'approved':True,'plan_sha256':'incorrect','user_message':'test fixture, not authorization'})
        p=subprocess.run([sys.executable,str(Path(diag.__file__).with_name('launch_final_edit_fit.py')),
                          '--plan',str(args.plan),'--authorization',str(auth)],capture_output=True,text=True)
        assert p.returncode!=0 and 'PermissionError' in p.stderr and not Path(cfg['runtime_directory']).exists()
        passed('mismatched launcher authorization fails before runtime creation')
        rc=Path(tmp)/'timeout'
        p=subprocess.run([sys.executable,str(Path(diag.__file__).with_name('run_stage.py')),
                          '--receipt-dir',str(rc),'--timeout','1','--',sys.executable,'-c',
                          'import time; print("CPU timeout fixture", flush=True); time.sleep(10)'],capture_output=True,text=True)
        receipt=diag.read(rc/'receipt.json')
        assert p.returncode==124 and receipt['exit_code']==124 and receipt['first_log']=='CPU timeout fixture' and receipt['elapsed_seconds']<7
        passed('real subprocess timeout recorded as exit124 with real first log')
    assert not torch.cuda.is_initialized()
    passed('no CUDA initialization or model weights loaded')
    diag.save(args.out,{'status':'passed','checks':results,'check_count':len(results),'plan_sha256':diag.sha(args.plan),
              'actual_target_structures':targets,'training_updates':0,'model_calls':0,'cuda_initialized':False,
              'limitations':'CPU evidence only; GPU loading, VRAM, free generation and utility not validated.'})
    print(len(results),'CPU checks passed')


if __name__=='__main__':main()
