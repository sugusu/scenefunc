"""An opt-in frozen-base catalog policy using a shared strict JSON action boundary.

Prepare restores and encodes the declared room without a model call. Rollout
resumes that checkpoint, uses the prepared request, then observes each real edit.
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
from native_checkpoint import rotation_roundtrip_differences


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def request_messages(session,observation,remaining,plan=None,audit=None):
    from Designer.agent_rollout import build_messages
    context=session.current_context()
    compact=True
    if compact:
        from compact_catalog_context import pack_context,geometry_feedback,use_feedback
        context,memory_data=pack_context(context)
    else:
        context['objects']={oid:{key:obj[key] for key in
            ('object_id','object_type','name','transform','bbox_min','bbox_max','immutable')}
            for oid,obj in context['objects'].items()}
        receipts=context.pop('prior_tool_receipts')
        memory_data={'recent_actual_batches':receipts,'history_policy':'Complete current object table plus latest four batches; full history retained in checkpoint.'}
    contract=('\nOutput exactly ONE JSON object with keys "name" and "arguments", without markdown or tool_call tags. '
        'Use an advertised tool name and its exact argument schema, with actual current object IDs and catalog IDs. '
        'The output grammar enforces schema and current references, not geometry or satisfaction of the task. '
        'If finished or unable to proceed, output {"name":"__done__","arguments":{"unmet_or_unknown":"your account of unmet or unknown requirements"}}. '
        'Done ends the policy; it does not establish room success. '
        'You will receive the real result and five views before the next turn. '
        'No scaling, articulation, new walls or service installation is available. '
        'Do not invent successful checks. Coordinates are metres, +Z up; canonical furniture front is +Y, yaw degrees around +Z. '
        'Treat fixture operation, stability and sanitation as unverified. '
        f'Remaining action calls including rejections: {remaining}.')
    brief=session.project.task['design_request']+'\nCURRENT NATIVE STATE:\n'+json.dumps(context,ensure_ascii=False,separators=(',',':'))
    memory=json.dumps(memory_data,ensure_ascii=False,separators=(',',':'))
    item=plan['prompt_supplement']
    if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Frozen prompt supplement changed')
    memory+='\n\n## Frozen task-planning support\n'+Path(item['path']).read_text()
    if plan.get('audit_plan'):
        if audit is None:raise ValueError('Frozen feedback plan requires an actual state audit')
        feedback={'geometry':geometry_feedback(audit)}
        if plan['feedback_arm']=='U':feedback['use']=use_feedback(audit)
        elif plan['feedback_arm']!='G':raise ValueError('Unknown feedback arm')
        memory+='\n\n## Independent current-state feedback\n'+json.dumps(feedback,ensure_ascii=False,separators=(',',':'))
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
    result={'phase':a.phase,'arm':plan['comparison_arm'],'status':'running','started_at':now(),'rounds':[],
        'tool_calls':0,'model_calls':0,'training_updates':0,'whole_room_success':'not_evaluated'}
    public=Path(plan['public_directory']);runtime=Path(plan['runtime_directory']);prepared_dir=Path(plan['prepared_directory'])
    def progress():
        write(a.out/'RESULT.json',result)
        status={'run_id':plan['run_id'],'updated_at':now(),'question':plan['question'],
            'stage':a.phase,'arm':plan['comparison_arm'],'status':result['status'],'run_directory':str(runtime),
            'completed_planned':{'tool_calls':{'completed':result['tool_calls'],'planned':plan['budget']['tool_calls']},
                'model_calls':{'completed':result['model_calls'],'planned':plan['budget']['model_calls']}},
            'configuration_hashes':{'plan':sha256(a.plan),'input_manifest':plan['manifest_sha256']},
            'latest_verifiable_artifact':str(a.out/'RESULT.json'),'failure_reason':result.get('error'),
            'started_at':result['started_at'],'ended_at':result.get('ended_at'),
            'next_step':'Complete the frozen independent whole-room attempt and audit every retained requirement; no training.'}
        write(public/'STATUS.json',status)
        write(Path(plan.get('latest_file',public.parent/'latest.json')),{'run_id':plan['run_id'],'stage':str(plan.get('feedback_arm',''))+'_'+a.phase,'status':result['status'],
            'updated_at':now(),'run_directory':str(runtime),'status_path':str(public/'STATUS.json')})
    print(now(),'begin whole-room',a.phase,flush=True);progress();client=None;locks=[]
    feedback=None
    def child(request,out,mode):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'] if mode=='generate' else '',
            PYTHONPATH=plan['structured_output']['dependency_directory'])
        command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(out.parent/(out.name+'_execution')),
            '--timeout',str(plan['budget']['seconds_per_model_call']), '--',plan['model_python'],
            str(Path(__file__).with_name('structured_catalog_step.py').resolve()),'--plan',str(a.plan.resolve()),
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
        if plan.get('audit_plan') and sha256(Path(plan['audit_plan']))!=plan['audit_plan_sha256']:
            raise ValueError('Frozen feedback audit plan changed')
        if plan.get('feedback_source'):
            item=plan['feedback_source']
            if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Initial feedback changed')
            feedback=json.loads(Path(item['path']).read_text())
            if feedback['input_manifest_sha256']!=plan['manifest_sha256']:raise ValueError('Feedback/input manifest mismatch')
        if a.phase=='prepare':
            for item in plan['model_weight_files']:
                path=Path(plan['model_directory'])/item['path']
                if path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:raise ValueError('Fixed model weight differs')
            result['weight_stat']={x['path']:[(Path(plan['model_directory'])/x['path']).stat().st_size,
                (Path(plan['model_directory'])/x['path']).stat().st_mtime_ns] for x in plan['model_weight_files']}
        else:
            prepared=json.loads((prepared_dir/'RESULT.json').read_text())
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
        source=Path(plan['bundle']) if a.phase=='prepare' else prepared_dir/'checkpoint'
        if a.phase=='prepare' and sha256(source/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Input manifest changed')
        post('/session/catalog',{'bundle_directory':str(source),'output_directory':str(a.out/'house')})
        session=server._state.session
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        if a.phase=='prepare':
            if plan.get('source_observation'):
                item=plan['source_observation']
                if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Source observation changed')
                obs=json.loads(Path(item['path']).read_text())
                if sha256(Path(obs['state_binding']))!=item['state_binding_sha256']:raise ValueError('Frozen view binding changed')
                binding=json.loads(Path(obs['state_binding']).read_text())
                if canonical_hash(binding['native_state'])!=binding['state_sha256']:raise ValueError('View binding is internally inconsistent')
                current=session.scene.to_state_dict()
                differences=rotation_roundtrip_differences(binding['native_state'],current)
                if differences and plan.get('view_reuse_equivalence')!='native_quaternion_roundtrip_1e-12':
                    raise ValueError('Reused views describe another exact state')
                result['view_reuse_equivalence']={'source_state_sha256':binding['state_sha256'],
                    'current_state_sha256':canonical_hash(current),'differences':differences,
                    'rule':'Only the existing native checkpoint quaternion roundtrip allowance (<=1e-12); every other field exact. No pose edit or new render claimed.'}
                if len(binding['images'])!=5 or obs['image_paths']!=[x['path'] for x in binding['images']]:raise ValueError('Five-view record mismatch')
                for image in binding['images']:
                    if sha256(Path(image['path']))!=image['sha256']:raise ValueError('Source view changed')
                result['reused_observation']=item
            else:obs=post('/session/observation')
            write(a.out/'observation.json',obs)
            write(a.out/'request.json',request_messages(session,obs,plan['budget']['tool_calls'],plan,feedback))
            result['encoding']=child(a.out/'request.json',a.out/'encode','encode')['encoding']
            result['checkpoint']=post('/session/checkpoint',{'output_directory':str(a.out/'checkpoint')})
            result['state_sha256']=canonical_hash(session.scene.to_state_dict())
            result['request_sha256']=sha256(a.out/'request.json');result['status']='prepared_without_model'
        else:
            stored=json.loads((prepared_dir/'checkpoint/house_state.json').read_text())['rooms'][session.scene.room_id]
            if canonical_hash(stored)!=prepared['state_sha256']:raise ValueError('Prepared state record changed')
            differences=rotation_roundtrip_differences(stored,session.scene.to_state_dict())
            if differences and plan.get('view_reuse_equivalence')!='native_quaternion_roundtrip_1e-12':
                raise ValueError('Resumed exact state changed')
            if sha256(prepared_dir/'request.json')!=prepared['request_sha256']:raise ValueError('Prepared request changed')
            request=prepared_dir/'request.json';obs=json.loads((prepared_dir/'observation.json').read_text())
            if differences:
                result['rollout_restoration_differences']=differences
                request=a.out/'initial_request.json'
                write(request,request_messages(session,obs,plan['budget']['tool_calls'],plan,feedback))
            result['initial_request_sha256']=sha256(request)
            from shared_action_contract import ActionContract,schemas_from_observation
            from compact_catalog_context import pack_context
            result['stop_reason']='budget_exhausted'
            for turn in range(plan['budget']['model_calls']):
                if result['tool_calls']>=plan['budget']['tool_calls']:break
                rounddir=a.out/f'round_{turn:03d}';rounddir.mkdir()
                write(rounddir/'request.json',json.loads(request.read_text()))
                state_before=canonical_hash(session.scene.to_state_dict());gpu_check()
                output=child(rounddir/'request.json',rounddir/'model','generate')
                raw=(rounddir/'model/action_output.txt').read_text()
                current,_=pack_context(session.current_context())
                gate=ActionContract(schemas_from_observation(obs),current)
                try:
                    validated=gate.validate(raw,current)
                    child_validated=json.loads((rounddir/'model/VALIDATED_ACTION.json').read_text())
                    if validated!=child_validated:raise ValueError('Model/host contract or current context changed')
                except Exception:
                    result['tool_calls']+=1
                    raise
                write(rounddir/'VALIDATED_ACTION.json',validated)
                calls=[] if validated['kind']=='done' else [validated['call']]
                write(rounddir/'parsed_calls.json',calls)
                row={'turn':turn,'model':output,'calls':calls,'before_sha256':state_before,'validated_action':validated}
                result['rounds'].append(row)
                if validated['kind']=='done':
                    result['stop_reason']='model_done_not_room_acceptance';progress();break
                result['tool_calls']+=1
                reply=post('/session/step',{'tool_calls':calls})
                write(rounddir/'tool_reply.json',reply);row['reply']=reply
                row['checkpoint']=post('/session/checkpoint',{'output_directory':str(rounddir/'checkpoint')})
                obs=post('/session/observation');write(rounddir/'observation.json',obs)
                row['after_sha256']=canonical_hash(session.scene.to_state_dict())
                if feedback is not None and row['after_sha256']!=state_before:
                    command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(rounddir/'audit_execution'),
                        '--timeout','120','--',sys.executable,str(Path(__file__).with_name('audit_catalog_room.py')),
                        '--plan',plan['audit_plan'],'--bundle',str(rounddir/'checkpoint'),'--out',str(rounddir/'feedback_audit.json')]
                    if subprocess.call(command):raise RuntimeError('Actual changed-state feedback audit failed; no inference on stale feedback')
                    feedback=json.loads((rounddir/'feedback_audit.json').read_text())
                if feedback is not None:write(rounddir/'feedback_binding.json',{'state_sha256':row['after_sha256'],
                    'audit':feedback,'recomputed_after_state_change':row['after_sha256']!=state_before})
                write(rounddir/'next_request.json',request_messages(session,obs,plan['budget']['tool_calls']-result['tool_calls'],plan,feedback))
                request=rounddir/'next_request.json'
                progress();print(now(),'finished round',turn,'tool_calls',result['tool_calls'],flush=True)
                if sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file())>plan['budget']['output_max_bytes']:
                    raise RuntimeError('Output byte budget exceeded; stop without deleting evidence')
            write(a.out/'final_state.json',session.scene.to_state_dict());write(a.out/'final_observation.json',obs)
            result['checkpoint']=post('/session/checkpoint',{'output_directory':str(a.out/'final_checkpoint')})
            result['status']='policy_attempt_completed_pending_independent_audit'
    except Exception as exc:
        result.update(status='failed',stop_reason='execution_error',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['ended_at']=now()
        result['prior_lineage_tool_calls']=plan.get('prior_tool_calls',0)
        result['cumulative_lineage_tool_calls']=result['prior_lineage_tool_calls']+result['tool_calls']
        progress()
        if client is not None:
            try:client.post('/session/close',json={})
            finally:client.close()
        for lock in locks:lock.close()
        print(now(),result['status'],flush=True)


if __name__=='__main__':main()
