"""Bounded offline model decisions over one preserved native furnished room."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from existing_project_session import create_app, canonical_hash
from native_project import sha256
from native_checkpoint import rotation_roundtrip_differences
from evaluate_mode_policy import evaluate, feedback
from measure_free_floor import rigid_sweep_diagnostic


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def messages(session, obs, plan, evaluation, remaining):
    from Designer.agent_rollout import build_messages
    source = session.current_context()
    objects = {k:{f:v[f] for f in ('object_id','object_type','name','transform','bbox_min','bbox_max','immutable')}
               for k,v in source['objects'].items()}
    receipts = copy.deepcopy(source['prior_tool_receipts'][-2:])
    for batch in receipts:
        for row in batch['results']:
            output = row.get('output')
            if row['name']=='get_current_scene_state' and isinstance(output,dict) and isinstance(output.get('objects'),list):
                output['full_query_hash'] = canonical_hash(output)
                output['queried_object_ids'] = [x['object_id'] for x in output.pop('objects')]
                output['details_summarized_in_current_table'] = True
    system = obs['system'] + ('\nReturn exactly one <tool_call>{"name":"...","arguments":{...}}</tool_call> per turn, '
        'or <done> and a short account if finished or unable to continue. Do not put a tool call in a hypothetical example. '
        'Every actual or rejected tool call consumes one action. Coordinates are metres, +Z up, yaw degrees. '
        'State query arguments are {}. check_facing only measures orientation, not visibility. '
        'All original objects must remain; do not delete, replace, add or scale. Six listed tables/stools may move. '
        'Other objects are immutable. Only the current state, last two receipts and measured feedback are supplied; '
        'complete history is saved. Recheck use and access after editing. '
        f'Remaining model turns and action calls: {remaining}.')
    brief = plan['brief']+'\n\n'+plan['common_constraints']
    if plan['knowledge_text']: brief += '\n\nSOURCED DESIGN KNOWLEDGE:\n'+plan['knowledge_text']
    brief += '\n\nCURRENT OBJECTS:\n'+json.dumps(objects,ensure_ascii=False,separators=(',',':'))
    memory = {'recent_actual_batches':receipts, 'actual_current_feedback':feedback(evaluation)}
    return build_messages(system, brief, json.dumps(memory,ensure_ascii=False,separators=(',',':')), obs['image_paths'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','rollout'],required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for name,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Frozen code changed: '+name)
    for key in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(key,None)
    os.environ.pop('LD_LIBRARY_PATH',None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',
        OPENSCENEAGENT_STUB_ASSETS='0',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    now=lambda:datetime.now(timezone.utc).isoformat()
    result={'status':'running','phase':a.phase,'case':plan.get('case_id'), 'started_at':now(),
            'plan_sha256':sha256(a.plan),'model_calls':0,'tool_calls':0,'training_updates':0,'rounds':[]}
    pub=Path(plan['public_directory']); runtime=Path(plan['runtime_directory']); client=server=None;locks=[]
    def progress():
        write(a.out/'RESULT.json',result)
        write(pub/('STATUS_'+str(plan.get('case_id','prepare'))+'.json'),dict(result,
            question=plan['question'],latest_verifiable_artifact=str(a.out/'RESULT.json'),
            planned_model_calls=plan['budget']['model_calls'],updated_at=now()))
        write(pub/'STATUS.json',{'run_id':plan['run_id'],'status':result['status'],'phase':a.phase,
            'case':plan.get('case_id'),'question':plan['question'],'updated_at':now(),'started_at':result['started_at'],
            'ended_at':result.get('ended_at'),'actual_exit_code':result.get('actual_exit_code'),
            'completed_planned':{'model_calls_case':result['model_calls'],'planned_case':plan['budget']['model_calls']},
            'configuration_hashes':{'plan':sha256(a.plan),'bundle':plan['manifest_sha256']},
            'latest_verifiable_artifact':str(a.out/'RESULT.json'),'failure_reason':result.get('error'),
            'next_step':'Complete frozen cases without retries, then compare actual room outcomes.'})
    def child(request,out,mode,case_plan):
        command=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(out.parent/(out.name+'_execution')),
            '--timeout',str(plan['budget']['seconds_per_model_call']),'--',plan['model_python'],
            str(Path(__file__).with_name('catalog_model_step.py')),'--plan',str(case_plan),'--request',str(request),'--out',str(out),'--mode',mode]
        code=subprocess.call(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'] if mode=='generate' else ''))
        record=json.loads((out/'RESULT.json').read_text()) if (out/'RESULT.json').exists() else {'model_calls':0,'status':'failed_before_result'}
        result['model_calls']+=record['model_calls'];progress()
        if code:raise RuntimeError('Model stage exit '+str(code)+'; no retry')
        return record
    def gpu_check():
        def query(q):return subprocess.check_output(['nvidia-smi','-i','0',q,'--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        uuid,used,free=query('--query-gpu=uuid,memory.used,memory.free').split(',')
        need=17534247392+(plan['budget']['max_input_tokens']+plan['budget']['max_new_tokens'])*147456+4*1024**3
        if uuid.strip()!=plan['gpu_uuid'] or int(used)>512 or query('--query-compute-apps=pid') or int(free)*1024**2<need:
            raise RuntimeError('GPU0 occupied or insufficient full-budget memory; no model call')
        return {'uuid':uuid.strip(),'used_mib':int(used),'free_mib':int(free),'required_bytes':need}
    print(now(),'begin',a.phase,plan.get('case_id'),flush=True);progress()
    try:
        if a.phase=='prepare':
            result['weight_stat']={}
            for row in plan['model_weight_files']:
                p=Path(plan['model_directory'])/row['path'];assert p.stat().st_size==row['bytes'] and sha256(p)==row['sha256']
                result['weight_stat'][row['path']]=[p.stat().st_size,p.stat().st_mtime_ns]
        else:
            prepared=json.loads((runtime/'prepare/RESULT.json').read_text());assert prepared['status']=='prepared_without_model'
            for name,value in prepared['weight_stat'].items():
                p=Path(plan['model_directory'])/name;assert [p.stat().st_size,p.stat().st_mtime_ns]==value
            for p in plan['gpu_lock_paths']:
                lock=Path(p).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
            result['initial_gpu']=gpu_check()
        from fastapi.testclient import TestClient
        app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
        def post(route,payload=None):
            response=client.post(route,json=payload or {})
            if response.status_code!=200:raise RuntimeError(route+': '+response.text[:1000])
            return response.json()
        source=Path(plan['bundle']) if a.phase=='prepare' else runtime/'prepare/checkpoint'
        if a.phase=='prepare':assert sha256(source/'manifest.json')==plan['manifest_sha256']
        else:assert sha256(source/'manifest.json')==prepared['checkpoint_manifest_sha256']
        post('/session/existing',{'bundle_directory':str(source),'output_directory':str(a.out/'house')})
        session=server._state.session;post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        scene=session.scene;initial=scene.to_state_dict();room=scene.scene_dir;root=session.project.directory
        write(a.out/'initial_state.json',initial)
        if a.phase=='prepare':
            obs=post('/session/observation');write(a.out/'observation.json',obs)
            result['cases']={}
            for case_path in plan['case_plans']:
                cp=json.loads(Path(case_path).read_text()); dest=a.out/cp['case_id'];dest.mkdir()
                ev=evaluate(initial,initial,room,root,cp);write(dest/'evaluation.json',ev)
                write(dest/'request.json',messages(session,obs,cp,ev,cp['budget']['model_calls']))
                enc=child(dest/'request.json',dest/'encode','encode',Path(case_path))
                result['cases'][cp['case_id']]={'request_sha256':sha256(dest/'request.json'),
                    'input_tokens':enc['encoding']['input_tokens'],'initial_condition_pass':ev['condition_pass']}
            result['checkpoint']=post('/session/checkpoint',{'output_directory':str(a.out/'checkpoint')})
            result['checkpoint_manifest_sha256']=sha256(a.out/'checkpoint/manifest.json')
            result['status']='prepared_without_model'
        else:
            prior=json.loads((runtime/'prepare/initial_state.json').read_text())
            result['restoration_differences']=rotation_roundtrip_differences(prior,initial)
            obs=json.loads((runtime/'prepare/observation.json').read_text())
            binding=json.loads(Path(obs['state_binding']).read_text())
            rotation_roundtrip_differences(binding['native_state'],initial)
            for image in binding['images']:assert sha256(Path(image['path']))==image['sha256']
            ev=evaluate(initial,initial,room,root,plan);write(a.out/'initial_evaluation.json',ev)
            result['stop_reason']='budget_exhausted';result['observations_new']=0
            from Designer.agent_rollout import parse_assistant_text
            for turn in range(plan['budget']['model_calls']):
                rd=a.out/f'round_{turn:03d}';rd.mkdir();before=scene.to_state_dict()
                request=messages(session,obs,plan,ev,plan['budget']['model_calls']-turn)
                write(rd/'request.json',request);write(rd/'before_state.json',before)
                write(rd/'before_observation.json',obs);write(rd/'before_evaluation.json',ev)
                gpu_check();model=child(rd/'request.json',rd/'model','generate',a.plan)
                raw=(rd/'model/raw_output.txt').read_text();_,calls=parse_assistant_text(raw)
                row={'turn':turn,'model':model,'calls':calls,'before_sha256':canonical_hash(before)}
                result['rounds'].append(row);write(rd/'parsed_calls.json',calls)
                if not calls:
                    result['stop_reason']='done' if '<done>' in raw else 'no_action_without_done';progress();break
                if len(calls)!=1:
                    result['stop_reason']='multiple_actions_rejected';row['execution_rejected']=True;progress();break
                result['tool_calls']+=1
                reply=post('/session/step',{'tool_calls':calls});write(rd/'reply.json',reply);row['reply']=reply
                after=scene.to_state_dict();write(rd/'after_state.json',after)
                row['after_sha256']=canonical_hash(after)
                if calls[0]['name']=='move_furniture_tool':
                    oid=calls[0]['arguments'].get('object_id')
                    if oid in before['objects'] and oid in after['objects']:
                        row['actual_sweep']=rigid_sweep_diagnostic(before,after,oid,room,root)
                row['checkpoint']=post('/session/checkpoint',{'output_directory':str(rd/'checkpoint')})
                changed=row['after_sha256']!=row['before_sha256'];row['new_observation']=changed
                if changed:obs=post('/session/observation');result['observations_new']+=1
                write(rd/'after_observation.json',obs)
                ev=evaluate(after,initial,room,root,plan);write(rd/'after_evaluation.json',ev)
                row['condition_pass']=ev['condition_pass'];progress()
                print(now(),plan['case_id'],'round',turn,'condition_pass',ev['condition_pass'],flush=True)
            write(a.out/'final_state.json',scene.to_state_dict());write(a.out/'final_observation.json',obs)
            write(a.out/'final_evaluation.json',ev)
            result['checkpoint']=post('/session/checkpoint',{'output_directory':str(a.out/'final_checkpoint')})
            result.update(status='completed',final_condition_pass=ev['condition_pass'])
        result['actual_exit_code']=0
    except Exception as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc),actual_exit_code=1)
        raise
    finally:
        if server is not None:server._close_all()
        if client is not None:client.close()
        for lock in locks:lock.close()
        import psutil
        children=[]
        for proc in psutil.Process().children(recursive=True):
            try:
                if os.getpgid(proc.pid)==os.getpgrp():children.append(proc);proc.terminate()
            except (ProcessLookupError,psutil.Error):pass
        _,alive=psutil.wait_procs(children,timeout=5)
        for proc in alive:
            try:proc.kill()
            except psutil.Error:pass
        result['ended_at']=now();progress();print(now(),result['status'],flush=True)


if __name__=='__main__':main()
