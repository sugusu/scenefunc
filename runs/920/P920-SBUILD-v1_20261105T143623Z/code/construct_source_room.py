"""Build a source reconstruction with a state-bound observation for every action."""
import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
from native_project import sha256, validate_bundle
from native_checkpoint import canonical_hash
from measure_room_access import measure
from preflight_sourced_trajectory import boxes


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=['preflight', 'execute'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, h in plan['code_sha256'].items():
        assert sha256(Path(__file__).parent / name) == h, name
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = Path(plan['bundle'])
    assert sha256(bundle / 'manifest.json') == plan['manifest_sha256']
    manifest, house, _, task = validate_bundle(bundle)
    rid = manifest['selected_room_id']
    initial = house['rooms'][rid]
    catalog = json.loads((bundle / 'catalog.json').read_text())
    templates = {e['template']['object_id']: e['template'] for e in catalog['entries']}
    assert len(plan['commands']) == plan['budget']['tool_calls'] == 9
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = dict(status='running', started_at=now(), phase=args.phase,
                  plan_sha256=sha256(args.plan), model_calls=0, training_updates=0,
                  tool_calls=0, observations=0, historical_trajectory=False,
                  professional_demonstration=False, training_admitted=False)
    print(now(), 'begin researcher reference', args.phase, flush=True)
    client = server = None
    try:
        if args.phase == 'preflight':
            state = copy.deepcopy(initial)
            for item in plan['commands']:
                command = item['command']; a = command['arguments']
                obj = copy.deepcopy(templates[a['asset_id']]); oid = item['planned_id']
                angle = math.radians(a['yaw'])
                obj.update(object_id=oid, transform={'translation': [a['x'], a['y'], 0],
                           'rotation_wxyz': [math.cos(angle/2), 0, 0, math.sin(angle/2)]})
                state['objects'][oid] = obj
            write(args.out / 'PROPOSED_STATE.json', state)
            env = {k: v for k, v in boxes(state).items()
                   if state['objects'][k]['object_type'] == 'furniture'}
            overlap = [[a,b] for (a,(al,ah)),(b,(bl,bh)) in itertools.combinations(env.items(),2)
                       if np.all(np.minimum(ah,bh)-np.maximum(al,bl)>1e-6)]
            outside = [k for k,(lo,hi) in env.items() if lo[0]<-3 or hi[0]>3 or lo[1]<-2.5 or hi[1]>2.5]
            measurement = measure(state, bundle / ('room_'+rid), bundle, plan['reference_measurement'])
            write(args.out / 'CONDITIONAL_ACCESS.json', measurement)
            result.update(status='reference_preflight_completed', overlaps=overlap,
                          outside=outside, all_declared_targets_reachable=all(
                              t['reachable_from_entry'] for s in measurement['scenarios'] for t in s['targets'].values()))
            assert not overlap and not outside, 'Proposed furniture boxes overlap or leave room'
            # Unreachable proxies are retained as evidence, not silently optimized away.
        else:
            preflight = json.loads(Path(plan['preflight_result']).read_text())
            assert preflight['status'] == 'reference_preflight_completed'
            assert preflight['plan_sha256'] == sha256(args.plan)
            for name in ['SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                         'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH']:
                os.environ.pop(name, None)
            os.environ.update(OPENAI_API_KEY='disabled-local-validation', OPENAI_BASE_URL='http://127.0.0.1:9/v1',
                OPENAI_AGENTS_DISABLE_TRACING='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                NO_PROXY='127.0.0.1,localhost', SCENE_RENDER_THREADS='4', SCENE_RENDER_SAMPLES='8', OPENSCENEAGENT_STUB_ASSETS='0')
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
            from fastapi.testclient import TestClient
            from catalog_project_session import create_app
            app, server = create_app(Path(plan['designer']), Path(plan['scenesmith']))
            client = TestClient(app); requests=[]
            def post(route, payload=None):
                reply = client.post(route, json=payload or {}); data=reply.json()
                requests.append(dict(route=route, payload=payload, status_code=reply.status_code, response=data))
                write(args.out/'HTTP_REQUESTS.json',requests)
                assert reply.status_code == 200, route
                return data
            post('/session/catalog',dict(bundle_directory=str(bundle), output_directory=str((args.out/'house').resolve())))
            session=server._state.session
            post('/session/stage',dict(stage='furniture',stub_assets=False,skip_render=False))
            def capture(label):
                state=session.scene.to_state_dict(); write(args.out/(label+'_state.json'),state)
                obs=post('/session/observation'); result['observations']+=1
                binding=json.loads(Path(obs['state_binding']).read_text())
                assert len(obs['image_paths'])==5 and binding['state_sha256']==canonical_hash(state)
                write(args.out/(label+'_observation.json'),obs)
                return state
            capture('initial'); steps=[]
            for index,item in enumerate(plan['commands'],1):
                command=item['command']; before=session.scene.to_state_dict()
                write(args.out/f'step_{index:02d}_before.json',before)
                result['tool_calls']+=1
                reply=post('/session/step',dict(tool_calls=[command])); output=reply['results'][0]['output']
                after=session.scene.to_state_dict(); write(args.out/f'step_{index:02d}_after.json',after)
                steps.append(dict(index=index, role=item['role'], command=command, reply=reply,
                                  before_sha256=canonical_hash(before), after_sha256=canonical_hash(after)))
                write(args.out/'STEPS.json',steps);write(args.out/'RESULT.json',result)
                assert isinstance(output,dict) and output.get('success') is True, 'Rejected action; no retry'
                oid=output['object_id']
                assert set(after['objects'])==set(before['objects'])|{oid} and oid not in before['objects']
                assert all(after['objects'][k]==v for k,v in before['objects'].items())
                assert session.current_context()['asset_bindings'][oid]['registry_id']==command['arguments']['asset_id']
                capture(f'step_{index:02d}')
                print(now(),'added',index,oid,flush=True)
            final=session.scene.to_state_dict()
            write(args.out/'final_state.json',final)
            write(args.out/'final_observation.json',json.loads((args.out/f"step_{len(plan['commands']):02d}_observation.json").read_text()))
            checkpoint=post('/session/checkpoint',dict(output_directory=str((args.out/'checkpoint').resolve())))
            write(args.out/'FINAL_CONTEXT.json',session.current_context())
            measurement=measure(final,session.scene.scene_dir,session.project.directory,plan['reference_measurement'])
            write(args.out/'CONDITIONAL_ACCESS.json',measurement)
            result.update(status='reference_constructed_pending_independent_audit', checkpoint=checkpoint,
                          final_state_sha256=canonical_hash(final), all_declared_targets_reachable=all(
                              t['reachable_from_entry'] for s in measurement['scenarios'] for t in s['targets'].values()))
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc));raise
    finally:
        if client is not None:
            try: client.post('/session/close',json={})
            finally: client.close()
        result['ended_at']=now();write(args.out/'RESULT.json',result)
        print(now(),result['status'],flush=True)


if __name__=='__main__':
    main()
