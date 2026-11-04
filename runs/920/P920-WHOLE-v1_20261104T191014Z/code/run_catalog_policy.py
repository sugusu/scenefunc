"""A finite whole-room policy attempt over actual native catalog tools.

Prepare renders and encodes the empty room without a model call. Rollout resumes
that checkpoint, uses the exact prepared request, then observes each real edit.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from catalog_project_session import create_app, canonical_hash
from native_project import sha256, validate_bundle


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def request_messages(session,observation,remaining):
    from Designer.agent_rollout import build_messages
    context=session.current_context()
    # Preserve exact numeric values and identities; omit renderer resource paths,
    # empty support metadata and repeated historical receipts from the prompt.
    context['objects']={oid:{key:obj[key] for key in
        ('object_id','object_type','name','transform','bbox_min','bbox_max','immutable')}
        for oid,obj in context['objects'].items()}
    receipts=context.pop('prior_tool_receipts')
    contract=('\nEmit exactly ONE executable action per turn as <tool_call>{"name": "...", "arguments": {...}}</tool_call>. '
        'You will receive the actual result and five updated views before your next turn. '
        'If finished or unable to proceed, output <done> with a concise account of unmet or unknown requirements and no tool call. '
        'Use only advertised tools and exact catalog IDs. No asset scaling, articulation, new walls or service installation is available. '
        'Do not invent successful checks. Coordinates are metres, +Z up; canonical furniture front is +Y, yaw degrees around +Z. '
        'Treat fixture operation, stability and sanitation as unverified. '
        f'Remaining action calls including rejections: {remaining}.')
    brief=session.project.task['design_request']+'\nCURRENT NATIVE STATE:\n'+json.dumps(context,ensure_ascii=False,separators=(',',':'))
    memory=json.dumps({'recent_actual_batches':receipts,'history_policy':'Complete current object table plus latest four batches; full history retained in checkpoint.'},ensure_ascii=False,separators=(',',':'))
    return build_messages(observation['system']+contract,brief,memory,observation['image_paths'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','rollout'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for name,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Frozen code changed: '+name)
    for name in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                 'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(name,None)
    os.environ.pop('LD_LIBRARY_PATH',None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0',
        TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    now=lambda:datetime.now(timezone.utc).isoformat()
    result={'phase':a.phase,'status':'running','started_at':now(),'rounds':[],
        'tool_calls':0,'model_calls':0,'training_updates':0,'whole_room_success':'not_evaluated'}
    public=Path(plan['public_directory']);runtime=Path(plan['runtime_directory'])
    def progress():
        write(a.out/'RESULT.json',result)
        status={'run_id':plan['run_id'],'updated_at':now(),'question':plan['question'],
            'stage':a.phase,'status':result['status'],'run_directory':str(runtime),
            'completed_planned':{'tool_calls':{'completed':result['tool_calls'],'planned':plan['budget']['tool_calls']},
                'model_calls':{'completed':result['model_calls'],'planned':plan['budget']['model_calls']}},
            'configuration_hashes':{'plan':sha256(a.plan),'input_manifest':plan['manifest_sha256']},
            'latest_verifiable_artifact':str(a.out/'RESULT.json'),'failure_reason':result.get('error'),
            'started_at':result['started_at'],'ended_at':result.get('ended_at'),
            'next_step':'Prepare CPU input and execute exactly one frozen whole-room attempt, then independently audit final state.'}
        write(public/'STATUS.json',status)
        write(public.parent/'latest.json',{'run_id':plan['run_id'],'stage':a.phase,'status':result['status'],
            'updated_at':now(),'run_directory':str(runtime),'status_path':str(public/'STATUS.json')})
    print(now(),'begin whole-room',a.phase,flush=True);progress();client=None;locks=[]
    def child(request,out,mode):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'] if mode=='generate' else '')
        command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(out.parent/(out.name+'_execution')),
            '--timeout',str(plan['budget']['seconds_per_model_call']), '--',plan['model_python'],
            str(Path(__file__).with_name('catalog_model_step.py').resolve()),'--plan',str(a.plan.resolve()),
            '--request',str(request.resolve()),'--out',str(out.resolve()),'--mode',mode]
        code=subprocess.call(command,env=env)
        if (out/'RESULT.json').exists():
            record=json.loads((out/'RESULT.json').read_text());result['model_calls']+=record['model_calls'];progress()
        if code:raise RuntimeError('Model '+mode+' stage exited '+str(code)+'; no retry')
        return json.loads((out/'RESULT.json').read_text())
    def gpu_check():
        def q(query):return subprocess.check_output(['nvidia-smi','-i','0',query,'--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        gpu=q('--query-gpu=uuid,memory.used,memory.free').split(',')
        required=17534247392+(plan['budget']['max_input_tokens']+plan['budget']['max_new_tokens'])*147456+4*1024**3
        if gpu[0].strip()!=plan['gpu_uuid'] or int(gpu[1])>512 or q('--query-compute-apps=pid') or int(gpu[2])*1024**2<required:
            raise RuntimeError('Specified GPU occupied or insufficient full-budget memory')
        host_available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        if host_available<17534247392*2+8*1024**3:raise RuntimeError('Insufficient host model-loading headroom')
        return {'uuid':gpu[0].strip(),'used_mib':int(gpu[1]),'free_mib':int(gpu[2]),'required_free_bytes':required}
    try:
        if a.phase=='prepare':
            for item in plan['model_weight_files']:
                path=Path(plan['model_directory'])/item['path']
                if path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:raise ValueError('Fixed model weight differs')
            result['weight_stat']={x['path']:[(Path(plan['model_directory'])/x['path']).stat().st_size,
                (Path(plan['model_directory'])/x['path']).stat().st_mtime_ns] for x in plan['model_weight_files']}
        else:
            prepared=json.loads((runtime/'prepare/RESULT.json').read_text())
            if prepared['status']!='prepared_without_model':raise ValueError('Preparation not accepted')
            for name,expected in prepared['weight_stat'].items():
                s=(Path(plan['model_directory'])/name).stat()
                if [s.st_size,s.st_mtime_ns]!=expected:raise ValueError('Model weight changed since hash audit')
            for p in plan['gpu_lock_paths']:
                lock=Path(p).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
            result['initial_gpu']=gpu_check()
        from fastapi.testclient import TestClient
        app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
        def post(route,payload=None):
            response=client.post(route,json=payload or {})
            if response.status_code!=200:raise RuntimeError('Native route failed '+route+': '+response.text[:1000])
            return response.json()
        source=Path(plan['bundle']) if a.phase=='prepare' else runtime/'prepare/checkpoint'
        if a.phase=='prepare' and sha256(source/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Input manifest changed')
        post('/session/catalog',{'bundle_directory':str(source),'output_directory':str(a.out/'house')})
        session=server._state.session
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        if a.phase=='prepare':
            obs=post('/session/observation');write(a.out/'observation.json',obs)
            write(a.out/'request.json',request_messages(session,obs,plan['budget']['tool_calls']))
            result['encoding']=child(a.out/'request.json',a.out/'encode','encode')['encoding']
            result['checkpoint']=post('/session/checkpoint',{'output_directory':str(a.out/'checkpoint')})
            result['state_sha256']=canonical_hash(session.scene.to_state_dict())
            result['request_sha256']=sha256(a.out/'request.json');result['status']='prepared_without_model'
        else:
            if canonical_hash(session.scene.to_state_dict())!=prepared['state_sha256']:raise ValueError('Resumed state changed')
            if sha256(runtime/'prepare/request.json')!=prepared['request_sha256']:raise ValueError('Prepared request changed')
            request=runtime/'prepare/request.json';obs=json.loads((runtime/'prepare/observation.json').read_text())
            from Designer.agent_rollout import parse_assistant_text
            result['stop_reason']='budget_exhausted'
            for turn in range(plan['budget']['model_calls']):
                if result['tool_calls']>=plan['budget']['tool_calls']:break
                rounddir=a.out/f'round_{turn:03d}';rounddir.mkdir()
                write(rounddir/'request.json',json.loads(request.read_text()))
                state_before=canonical_hash(session.scene.to_state_dict());gpu_check()
                output=child(rounddir/'request.json',rounddir/'model','generate')
                raw=(rounddir/'model/raw_output.txt').read_text();_,calls=parse_assistant_text(raw)
                write(rounddir/'parsed_calls.json',calls)
                row={'turn':turn,'model':output,'calls':calls,'before_sha256':state_before}
                result['rounds'].append(row)
                if not calls:
                    result['stop_reason']='model_returned_no_action';progress();break
                if len(calls)!=1:raise ValueError('Policy violated one-action-per-turn contract; output retained, no execution')
                reply=post('/session/step',{'tool_calls':calls});result['tool_calls']+=1
                write(rounddir/'tool_reply.json',reply);row['reply']=reply
                row['checkpoint']=post('/session/checkpoint',{'output_directory':str(rounddir/'checkpoint')})
                obs=post('/session/observation');write(rounddir/'observation.json',obs)
                row['after_sha256']=canonical_hash(session.scene.to_state_dict())
                write(rounddir/'next_request.json',request_messages(session,obs,plan['budget']['tool_calls']-result['tool_calls']))
                request=rounddir/'next_request.json'
                progress();print(now(),'finished round',turn,'tool_calls',result['tool_calls'],flush=True)
                if sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file())>plan['budget']['output_max_bytes']:
                    raise RuntimeError('Output byte budget exceeded; stop without deleting evidence')
            write(a.out/'final_state.json',session.scene.to_state_dict());write(a.out/'final_observation.json',obs)
            result['checkpoint']=post('/session/checkpoint',{'output_directory':str(a.out/'final_checkpoint')})
            result['status']='policy_attempt_completed_pending_independent_audit'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['ended_at']=now();progress()
        if client is not None:
            try:client.post('/session/close',json={})
            finally:client.close()
        for lock in locks:lock.close()
        print(now(),result['status'],flush=True)


if __name__=='__main__':main()
