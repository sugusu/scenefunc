"""Finite free-generation check on two seen training prompts, without optimizer or edits."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(a):
    assert not a.out.exists()
    a.out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    torch.set_num_threads(2)
    enc=read(a.study/'ENCODING_PLAN.json')
    source=Path(enc['official_source'])
    sys.path.insert(0,str(source/'src'))
    spec=importlib.util.spec_from_file_location('probe_official_dataset',source/'src/sft_dataset.py')
    up=importlib.util.module_from_spec(spec);sys.modules[spec.name]=up;spec.loader.exec_module(up)
    template=read(a.study/'evaluation/Milan_frozen_base_knowledge_feedback_17/PLAN.json')
    sys.path.insert(0,str(Path(template['encoder']).parent))
    from local_vlm import LocalQwenEngine
    engine=LocalQwenEngine(template['model_directory'],a.out/'unused',max_pixels=65536,max_input_tokens=18432,max_new_tokens=1024)
    data=read(a.study/'DATASET.json');cache=read(a.study/'ENCODING.json')
    selected=['clothing_build_07','waterloo_edit_02']
    cells=[]
    for ident in selected:
        row=next(x for x in data['rows'] if x['id']==ident)
        docpath=Path(row['documents']['action_only']['path'])
        assert sha(docpath)==row['documents']['action_only']['sha256']
        ds=up.StepwiseSFTDataset(docpath.parent,a.study.resolve().parents[3],engine.processor,max_length=16384)
        assert len(ds)==1
        _,prompt=up._build_messages(ds.samples[0])
        inputs,receipt=engine.encode(prompt)
        cached=next(x for x in cache['records'] if x['id']==ident and x['arm']=='action_only')
        assert sha(cached['tensor_path'])==cached['tensor_sha256']
        batch=torch.load(cached['tensor_path'],map_location='cpu',weights_only=True)
        length=inputs['input_ids'].shape[1]
        assert torch.equal(inputs['input_ids'],batch['input_ids'][:,:length])
        assert torch.equal(inputs['pixel_values'],batch['pixel_values'])
        request=a.out/'inputs'/ident/'request.json';write(request,prompt)
        write(request.with_name('CONTEXT.json'),json.loads(read(docpath)['task_context']['scene_description']))
        write(request.with_name('REFERENCE.json'),dict(command=row['command'],decision=row['decision_target'],not_supplied_to_inference=True))
        write(request.with_name('ENCODING.json'),receipt)
        for arm in ['frozen_base_knowledge_feedback','action_only','decision_action']:
            original=a.study/f'evaluation/Milan_{arm}_17/PLAN.json'
            p=read(original);p['inference']['seed']=17
            p['budget'].update(model_calls=1,tool_calls=0,seconds_per_model_call=180,gpu_seconds_per_cell=180)
            p.update(purpose='seen_training_prompt_free_generation_no_scene_execution',cell_id=ident+'_'+arm)
            planpath=a.out/'cells'/p['cell_id']/'PLAN.json';write(planpath,p)
            cells.append(dict(id=p['cell_id'],example=ident,arm=arm,plan=str(planpath.resolve()),plan_sha256=sha(planpath),
                request=str(request.resolve()),request_sha256=sha(request),training_prefix_ids_and_pixels_exact=True))
    protocol=dict(run_id=a.out.name,created_at=now(),question='Can frozen/adapted models freely generate a legal edit and explicit decision from two exact seen preassistant prompts?',
        source_study=str(a.study.resolve()),selected_examples=selected,selection='One supported addition and one supported table move; fixed before any probe output.',
        budgets=dict(model_calls=6,model_subprocess_seconds=1080,max_new_tokens_each=1024,optimizer_updates=0,native_edits=0,retries=0),
        script_sha256=sha(__file__),cells=cells,
        limits=['Training examples, not generalization evidence.','No gold decision or action supplied.','Exact reference action matching is diagnostic, not unique correct design.','Schema validity is not geometry or room success.','One seed; no causal attribution of failures.'],
        stopping_rule='Stop on infrastructure error or six completed calls. Never append updates, retries, alternate seeds or room rollouts.')
    write(a.out/'PROTOCOL.json',protocol)
    write(a.out/'STATUS.json',dict(status='frozen_not_started',updated_at=now(),completed=0,planned=6,protocol_sha256=sha(a.out/'PROTOCOL.json')))
    print(json.dumps(dict(status='frozen',run_id=a.out.name,cells=len(cells))),flush=True)


def run(a):
    from shared_action_contract import ActionContract,schemas_from_observation
    from jsonschema.exceptions import ValidationError
    protocol=read(a.out/'PROTOCOL.json');assert sha(__file__)==protocol['script_sha256']
    status=read(a.out/'STATUS.json');assert status['status']=='frozen_not_started'
    locks=[];results=[];elapsed=0.
    first=read(protocol['cells'][0]['plan'])
    for path in first['gpu_lock_paths']:
        lock=Path(path).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
    assert gpu[0].strip()==first['gpu_uuid'] and int(gpu[1])<512 and int(gpu[2])>23000
    status.update(status='running',started_at=now());write(a.out/'STATUS.json',status)
    try:
        for cell in protocol['cells']:
            assert sha(cell['plan'])==cell['plan_sha256'] and sha(cell['request'])==cell['request_sha256']
            directory=Path(cell['plan']).parent
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=first['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            command=[first['model_python'],str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(directory/'receipt'),'--timeout','180','--',first['model_python'],str(Path(__file__).with_name('edit_study_model_step.py')),'--plan',cell['plan'],'--request',cell['request'],'--out',str(directory/'model'),'--mode','generate']
            code=subprocess.call(command,env=env);receipt=read(directory/'receipt/receipt.json');elapsed+=receipt['elapsed_seconds']
            if code:raise RuntimeError(f'{cell["id"]}: subprocess exit {code}; no retry')
            model=read(directory/'model/RESULT.json');raw=(directory/'model/raw_output.txt').read_text()
            blocks=re.findall(r'<tool_call>(.*?)</tool_call>',raw,re.S)
            context=read(Path(cell['request']).with_name('CONTEXT.json'))
            request=read(cell['request']);contract=ActionContract(schemas_from_observation(dict(system=request[0]['content'])),context)
            row=dict(id=cell['id'],arm=cell['arm'],example=cell['example'],explicit_decision='<think>' in raw,
                model_result=str(directory/'model/RESULT.json'),output_sha256=sha(directory/'model/raw_output.txt'),schema_and_identity_valid=False)
            try:
                if len(blocks)!=1 or model.get('output_rejected_reason'):raise ValueError('Incomplete or non-single action')
                row['call']=contract.validate(blocks[0],context)['call'];row['schema_and_identity_valid']=True
            except (ValueError,ValidationError) as exc:
                row['validation_error']=str(exc)[:1000]
                if len(blocks)==1:
                    try:row['call']=json.loads(blocks[0])
                    except ValueError:pass
            reference=read(Path(cell['request']).with_name('REFERENCE.json'))['command'][0]
            row['exact_reference_action']=row.get('call')==reference
            row['reference_tool_match']=row.get('call',{}).get('name')==reference['name']
            results.append(row);write(directory/'CHECK.json',row)
            status.update(updated_at=now(),completed=len(results),elapsed_model_subprocess_seconds=elapsed,latest_verifiable_artifact=str(directory/'CHECK.json'));write(a.out/'STATUS.json',status)
        status.update(status='completed',ended_at=now(),exit_code=0)
    except BaseException as exc:
        status.update(status='failed',ended_at=now(),exit_code=1,failure_reason=str(exc));raise
    finally:
        write(a.out/'SUMMARY.json',dict(status=status['status'],results=results,model_calls=len(results),model_subprocess_seconds=elapsed,training_updates=0,native_edits=0))
        write(a.out/'STATUS.json',status)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','run'],required=True);ap.add_argument('--study',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();prepare(args) if args.phase=='prepare' else run(args)
