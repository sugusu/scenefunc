"""Guarded single-GPU probe/formal generation for the frozen four requests."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

UUID = 'GPU-6df2660c-5a77-4e41-0b1a-c6208368fcd1'
MODEL_PYTHON = '/data/wyh/conda/envs/c15_genrec/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, data):
    temp = Path(p).with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(p)


def progress(root, stage, launch):
    run_id = 'P920-DD-v1_20261104T083214Z'
    pub = root/'scenefunc/runs/920'/run_id
    run = root/'runtime'/run_id
    results = [read(p) for p in sorted((run/'model_runs').glob('*/result.json'))]
    formal = [x for x in results if x['kind']=='formal']
    status = read(pub/'STATUS.json')
    status.update(updated_at=now(), stage=stage, status='running' if launch['status']=='running' else 'partial',
                  active_model_launch=launch, formal_model_results=formal,
                  next_step='Complete exactly four frozen G/U revision attempts, execute bounded calls, then independently audit and score all resulting scenes.')
    status['completed_planned']['formal_revision_rounds'].update(completed=len([x for x in formal if x['status']!='running']), planned=4)
    status['configuration_hashes']['comparison_protocol'] = sha(run/'comparison_inputs/PROTOCOL.json')
    dump(pub/'STATUS.json', status)
    dump(pub.parent/'latest.json', {'run_id':run_id,'stage':stage,'status':status['status'],'updated_at':now(),'run_directory':str(root),'status_path':str(pub/'STATUS.json')})
    local = read(root/'STATUS.json')
    local.update(updated_at=now(), stage='running', milestone=stage, process={k:launch.get(k) for k in ['pid','start_ticks','child_pid']}, logs=list(set(local['logs']+[launch['log']])), next_step=status['next_step'])
    local['actual_calls']['revision_model_rounds']={'successful':sum(x['status']=='completed' for x in formal),'failed':sum(x['status'] not in ['running','completed'] for x in formal)}
    dump(root/'STATUS.json', local)


def child(a, run, target):
    import socket
    def deny(*args, **kwargs):
        raise RuntimeError('Network is disabled during local model generation')
    socket.socket.connect=deny
    socket.create_connection=deny
    import resource
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.94)
    source=run/'comparison_inputs'/a.label
    record={'kind':a.kind,'label':a.label,'started_at':now(),'status':'running',
            'precision':'float16','attention':'sdpa','seed':0,'use_cache':True,'batch_size':1,
            'max_new_tokens':32 if a.kind=='probe' else 2048,
            'request_sha256':sha(source/'request.json'),'input_tensor_sha256':sha(source/'inputs.pt'),
            'formal_round':a.kind=='formal','paid_api_calls':0,'training':False}
    start=time.monotonic()
    dump(target/'result.json',record)
    print(now(),'loading fixed FP16 model for',a.kind,a.label,flush=True)
    try:
        expected=read(source/'receipt.json')
        assert record['input_tensor_sha256']==expected['input_tensor_sha256'] and record['request_sha256']==expected['request_sha256']
        inputs=torch.load(source/'inputs.pt',weights_only=True,map_location='cpu')
        record['input_tokens']=int(inputs['input_ids'].shape[1])
        modeldir=read(a.root/'MODEL_DOWNLOAD_PLAN.json')['destination']
        model=Qwen3VLForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,
                 torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        assert next(model.parameters()).dtype==torch.float16
        torch.cuda.synchronize()
        record['load_seconds']=time.monotonic()-start
        inputs={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        t=time.monotonic()
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=record['max_new_tokens'],do_sample=False,use_cache=True)
        torch.cuda.synchronize()
        ids=output[0,record['input_tokens']:].cpu().tolist()
        record.update(generation_seconds=time.monotonic()-t,output_tokens=len(ids))
        dump(target/'output_ids.json',ids)
        processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
        raw=processor.decode(ids,skip_special_tokens=False)
        (target/'raw_output.txt').write_text(raw,encoding='utf-8')
        record['raw_output_sha256']=sha(target/'raw_output.txt')
        eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        record['ended_with_eos']=bool(ids and ids[-1] in (eos or []))
        record['status']='completed' if a.kind=='probe' or record['ended_with_eos'] else 'length_limit_no_execution'
        if a.kind=='formal' and record['status']=='completed':
            sys.path.insert(0,str(run/'work/paris_A_inference_20261104'))
            from run import parser
            _,calls=parser()(raw)
            dump(target/'parsed_calls.json',calls)
            record['parsed_call_count']=len(calls)
            if len(calls)>12:record['status']='call_budget_exceeded_no_execution'
    except Exception as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        record.update(ended_at=now(),total_seconds=time.monotonic()-start,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(target/'result.json',record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    return 0 if record['status']=='completed' else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--kind',choices=['probe','formal'],required=True)
    p.add_argument('--label',choices=['A_G','A_U','B_G','B_U'],required=True)
    p.add_argument('--child',action='store_true')
    a=p.parse_args();a.root=a.root.resolve()
    run=a.root/'runtime/P920-DD-v1_20261104T083214Z'
    target=run/'model_runs'/((a.kind+'_')+a.label)
    if a.child:return child(a,run,target)
    assert not target.exists(),'Prior attempt exists; never overwrite or automatically retry'
    inputs=read(run/'comparison_inputs'/a.label/'receipt.json')
    protocol=read(run/'comparison_inputs/PROTOCOL.json')
    budget=32 if a.kind=='probe' else 2048
    required=17534247392+(inputs['input_tokens']+budget)*147456+4*1024**3
    probe=None
    if a.kind=='formal':
        probe=read(run/'model_runs'/('probe_'+protocol['probe_label'])/'result.json')
        assert probe['status']=='completed' and probe['input_tokens']>=inputs['input_tokens']
        required=max(required,probe['peak_reserved_bytes']+(2048-32)*147456+1024**3)
    with (a.root.parent/'work/gpu0.lock').open('r') as old, (a.root/'gpu0.lock').open('a') as lock:
        fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
        def query(q):return subprocess.check_output(['nvidia-smi','-i','0',q,'--format=csv,noheader,nounits'],env=env,text=True,timeout=15).strip()
        gpu=[x.strip() for x in query('--query-gpu=uuid,memory.used,memory.free,memory.total,driver_version').split(',')]
        assert gpu[0]==UUID and int(gpu[1])<=512 and not query('--query-compute-apps=pid')
        assert int(gpu[2])*1024**2>=required,'Insufficient analytic/measured full-budget headroom'
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        assert mem>=17534247392*2+8*1024**3
        target.mkdir(parents=True)
        env.update(CUDA_VISIBLE_DEVICES=UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        command=[MODEL_PYTHON,str(Path(__file__).resolve()),'--root',str(a.root),'--kind',a.kind,'--label',a.label,'--child']
        launch={'started_at':now(),'status':'running','kind':a.kind,'label':a.label,'command':command,
                'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
                'gpu_uuid':UUID,'free_mib':int(gpu[2]),'total_mib':int(gpu[3]),'driver':gpu[4],
                'required_free_bytes':required,'host_available_bytes':mem,'probe_source':protocol['probe_label'] if probe else None,
                'log':str(target/'model.log'),'timeout_seconds':1200}
        with (target/'model.log').open('x',encoding='utf-8') as log:
            proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launch['child_pid']=proc.pid
            launch['child_start_ticks']=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19]
            dump(target/'launch.json',launch);progress(a.root,a.kind+'_'+a.label+'_running',launch)
            print(json.dumps(launch),flush=True)
            try:code=proc.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        lines=(target/'model.log').read_text(encoding='utf-8',errors='replace').splitlines()
        launch.update(ended_at=now(),status='completed' if code==0 else 'failed',exit_code=code,first_log=lines[0] if lines else None)
        dump(target/'launch.json',launch);progress(a.root,a.kind+'_'+a.label+'_finished',launch)
        print(json.dumps(launch),flush=True)
        return code


if __name__=='__main__':
    raise SystemExit(main())
