"""One frozen researcher-constructed whole-room activity transition and return.

Real native tools and state-bound views; not a model policy or historical
professional demonstration. Every accepted step is checkpointed immediately.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from existing_project_session import canonical_hash,create_app
from native_project import sha256
from measure_room_access import measure
from measure_free_floor import largest_floor_rectangle,sweep_diagnostic


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());args.out.mkdir(parents=True,exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(),'begin bounded researcher activity replay',flush=True)
    source=Path(plan['bundle'])
    if sha256(source/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Input changed')
    for name in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                 'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(name,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    checks,requests,steps,states=[],[],[],{}
    result={'status':'running','model_calls':0,'training_updates':0,'policy_kind':'researcher_constructed_not_model_or_historical',
            'checks':checks,'steps':steps,'states':states,'tool_calls':0,'observations':0}
    server=client=None
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:raise ValueError(name)
    try:
        from fastapi.testclient import TestClient
        from PIL import Image
        app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
        def post(route,payload=None):
            started=time.monotonic();response=client.post(route,json=payload or {});data=response.json()
            requests.append({'route':route,'payload':payload,'status_code':response.status_code,
                             'elapsed_seconds':time.monotonic()-started,'response':data})
            dump(args.out/'HTTP_REQUESTS.json',requests)
            if response.status_code!=200:raise RuntimeError(str(requests[-1]))
            return data
        post('/session/existing',{'bundle_directory':str(source.resolve()),'output_directory':str((args.out/'house').resolve())})
        session=server._state.session;scene=session.scene
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        initial=scene.to_state_dict();oid=plan['target_object_id'];root=session.project.directory
        room=scene.scene_dir
        check('all_initial_objects_retained',len(initial['objects'])==32 and oid in initial['objects'])
        def capture(label):
            state=scene.to_state_dict();dump(args.out/(label+'_state.json'),state)
            context=session.current_context();dump(args.out/(label+'_context.json'),context)
            obs=post('/session/observation');result['observations']+=1
            check(label+'_five_state_bound_views',len(obs['image_paths'])==5 and
                  canonical_hash(json.loads(Path(obs['state_binding']).read_text())['native_state'])==canonical_hash(state))
            for path in obs['image_paths']:
                with Image.open(path) as image:image.verify()
            measurement=measure(state,room,root,plan['access_settings'])
            measurement['floor_rectangle']=largest_floor_rectangle(state,room,root,plan['access_settings'])
            dump(args.out/(label+'_measurement.json'),measurement);dump(args.out/(label+'_observation.json'),obs)
            states[label]={'state_sha256':canonical_hash(state),'measurement':measurement,
                'step_idx':server._state.env.step_idx,'image_paths':obs['image_paths'],
                'actual_target_transform':state['objects'][oid]['transform']}
            print(json.dumps({'state':label,'step':server._state.env.step_idx,'rectangle':measurement['floor_rectangle']}),flush=True)
        capture('initial')
        for index,command in enumerate(plan['tool_calls'],1):
            before=scene.to_state_dict();proposed=copy.deepcopy(before)
            arguments=command['arguments'];yaw=math.radians(arguments['yaw'])
            proposed['objects'][oid]['transform']={'translation':[arguments['x'],arguments['y'],0.0],
                'rotation_wxyz':[math.cos(yaw/2),0,0,math.sin(yaw/2)]}
            planned=sweep_diagnostic(before,proposed,oid,room,root)
            step={'index':index,'planned_tool_call':command,'before_sha256':canonical_hash(before),'planned_sweep':planned}
            steps.append(step);dump(args.out/'STEPS.json',steps)
            check('planned_sweep_clear_'+str(index),planned['clear_under_declared_sweep_model'])
            reply=post('/session/step',{'tool_calls':[command]});result['tool_calls']+=1
            after=scene.to_state_dict();step.update(reply=reply,after_sha256=canonical_hash(after),
                actual_transform=after['objects'][oid]['transform'],actual_sweep=sweep_diagnostic(before,after,oid,room,root))
            dump(args.out/'STEPS.json',steps)
            # Save real accepted/rejected state before any further assertion.
            checkpoint=post('/session/checkpoint',{'output_directory':str((args.out/('checkpoint_'+str(index))).resolve())})
            step['checkpoint']=checkpoint;dump(args.out/'STEPS.json',steps)
            check('native_tool_accepted_'+str(index),reply['results'][0]['output']['success'] is True)
            check('inventory_and_other_objects_unchanged_'+str(index),set(after['objects'])==set(initial['objects']) and
                  all(after['objects'][k]==initial['objects'][k] for k in initial['objects'] if k!=oid))
            check('actual_sweep_clear_'+str(index),step['actual_sweep']['clear_under_declared_sweep_model'])
            if index==2:capture('parked')
            if index==4:capture('returned')
        final=scene.to_state_dict()
        delta=[b-a for a,b in zip(initial['objects'][oid]['transform']['translation'],final['objects'][oid]['transform']['translation'])]
        result.update(status='four_native_moves_and_return_completed',return_translation_delta_m=delta,
            professional_sample_admitted=False,whole_room_usability='unknown',checkpoint=steps[-1]['checkpoint'],
            limitations=['Researcher-constructed actions; no model, trained adapter or historical professional trajectory.',
                        'Table mass/handling and actual human activity remain unverified.',
                        'Report all approach losses and the pre-existing 0.6 m route failure; empty area is not whole-room success.'])
    except BaseException as error:
        result.update(status='failed',failure=type(error).__name__+': '+str(error));raise
    finally:
        if server is not None:server._close_all()
        if client is not None:client.close()
        import psutil
        children=[]
        for process in psutil.Process().children(recursive=True):
            try:
                if os.getpgid(process.pid)==os.getpgrp():children.append(process);process.terminate()
            except (ProcessLookupError,psutil.Error):pass
        _,alive=psutil.wait_procs(children,timeout=5)
        for process in alive:
            try:process.kill()
            except psutil.Error:pass
        result['ended_at']=datetime.now(timezone.utc).isoformat();dump(args.out/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'tool_calls':result['tool_calls'],'observations':result['observations'],'failure':result.get('failure')}),flush=True)


if __name__=='__main__':main()
