"""Execute a frozen table-mode transition with every state, view and receipt retained."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from existing_project_session import canonical_hash, create_app
from native_project import sha256
from measure_room_access import measure, obstacles
from measure_free_floor import largest_floor_rectangle, rigid_sweep_diagnostic
from shapely.geometry import box
from shapely.ops import unary_union
from measure_table_modes import table_modes


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(),'begin frozen source-guided native edits',flush=True)
    bundle=Path(plan['bundle'])
    if sha256(bundle/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Input changed')
    for name,expected in plan['scripts'].items():
        if sha256(Path(__file__).parent/name)!=expected:raise ValueError('Frozen script changed: '+name)
    for name in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                 'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(name,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    result={'status':'running','model_calls':0,'training_updates':0,'tool_calls':0,'observations':0,
            'provenance':'researcher_reconstruction_based_on_sourced_professional_decision','checks':[],'states':{}}
    requests=[];server=client=None
    def check(name,condition):
        result['checks'].append({'name':name,'passed':bool(condition)})
        if not condition:raise ValueError(name)
    try:
        from fastapi.testclient import TestClient
        from PIL import Image
        app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
        def post(route,payload=None):
            t=time.monotonic();response=client.post(route,json=payload or {});data=response.json()
            requests.append({'route':route,'payload':payload,'status_code':response.status_code,
                             'elapsed_seconds':time.monotonic()-t,'response':data})
            dump(a.out/'HTTP_REQUESTS.json',requests)
            if response.status_code!=200:raise RuntimeError('HTTP failure: '+route)
            return data
        post('/session/existing',{'bundle_directory':str(bundle),'output_directory':str((a.out/'house').resolve())})
        session=server._state.session;scene=session.scene
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        root=session.project.directory;room=scene.scene_dir;initial=scene.to_state_dict();oid=plan['target_object_id']
        check('expected_inventory',len(initial['objects'])==plan['expected_objects'] and oid in initial['objects'])
        def capture(label):
            state=scene.to_state_dict();dump(a.out/(label+'_state.json'),state)
            dump(a.out/(label+'_context.json'),session.current_context())
            obs=post('/session/observation');result['observations']+=1
            check(label+'_five_state_bound_views',len(obs['image_paths'])==5 and
                  canonical_hash(json.loads(Path(obs['state_binding']).read_text())['native_state'])==canonical_hash(state))
            for path in obs['image_paths']:
                with Image.open(path) as image:image.verify()
            m=table_modes(state,room,root,plan)
            dump(a.out/(label+'_measurement.json'),m);dump(a.out/(label+'_observation.json'),obs)
            result['states'][label]={'native_state_sha256':canonical_hash(state),'measurement':m,'image_paths':obs['image_paths']}
            print(json.dumps({'state':label,'mode_geometry':m['mode_geometry'],'rectangle':m['floor_rectangle']}),flush=True)
            return state
        capture('initial')
        commands=plan.get('tool_calls') or [plan['tool_call']]
        check('frozen_tool_budget',0 < len(commands) <= plan['budget']['tool_calls'])
        steps=[]
        for index,command in enumerate(commands,1):
            args=command['arguments'];oid=args['object_id'];before=scene.to_state_dict()
            dump(a.out/('step_'+str(index)+'_before_state.json'),before)
            proposed=copy.deepcopy(before);yaw=math.radians(args['yaw'])
            proposed['objects'][oid]['transform']={'translation':[args['x'],args['y'],0], 'rotation_wxyz':[math.cos(yaw/2),0,0,math.sin(yaw/2)]}
            step={'index':index,'command':command,'planned_sweep':rigid_sweep_diagnostic(before,proposed,oid,room,root),'before_sha256':canonical_hash(before)}
            steps.append(step);dump(a.out/'STEPS.json',steps)
            check('planned_sweep_clear_'+str(index),step['planned_sweep']['clear_under_declared_sweep_model'])
            result['tool_calls']+=1
            reply=post('/session/step',{'tool_calls':[command]});after=scene.to_state_dict()
            dump(a.out/('step_'+str(index)+'_after_state.json'),after)
            step.update(reply=reply,after_sha256=canonical_hash(after),actual_transform=after['objects'][oid]['transform'])
            dump(a.out/'STEPS.json',steps)
            step['checkpoint']=post('/session/checkpoint',{'output_directory':str((a.out/('checkpoint_'+str(index))).resolve())})
            step['actual_sweep']=rigid_sweep_diagnostic(before,after,oid,room,root);dump(a.out/'STEPS.json',steps)
            check('identities_and_non_target_objects_unchanged_'+str(index),set(before['objects'])==set(after['objects']) and
                  all(before['objects'][key]==after['objects'][key] for key in before['objects'] if key!=oid))
            check('native_tool_accepted_'+str(index),reply['results'][0]['output']['success'] is True)
            check('actual_sweep_clear_'+str(index),step['actual_sweep']['clear_under_declared_sweep_model'])
            capture('after' if index==len(commands) else 'step_'+str(index))
        if len(steps)==1:dump(a.out/'STEP.json',steps[0])
        result.update(status='native_move_completed' if len(steps)==1 else 'native_sequence_completed',tool_accepted=True,
            checkpoint=steps[-1]['checkpoint'],sweep=steps[-1]['actual_sweep'],positive_professional_demonstration=False,
            whole_room_usability='unknown',historical_trajectory=False)
    except BaseException as e:
        result.update(status='failed',failure=type(e).__name__+': '+str(e));raise
    finally:
        if server is not None:server._close_all()
        if client is not None:client.close()
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
        result['ended_at']=datetime.now(timezone.utc).isoformat();dump(a.out/'RESULT.json',result)
        print(json.dumps({k:result[k] for k in ('status','tool_calls','observations')}),flush=True)


if __name__=='__main__':main()
