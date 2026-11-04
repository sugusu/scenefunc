"""Observe a complete declared reconstruction through the real native session.

One state query, one five-view observation, no design edits or model calls.
Envelope warnings are diagnostic measurements, not collision/utility scores.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

from existing_project_session import create_app, canonical_hash
from native_project import sha256


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--designer', type=Path, required=True)
    p.add_argument('--scenesmith', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(), 'begin native whole-room observation', flush=True)
    for key in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                'HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy'):
        os.environ.pop(key,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    requests, checks = [], []
    result = {'status':'running','model_calls':0,'training_updates':0,'design_edits':0,'checks':checks}
    server = client = None
    try:
        from fastapi.testclient import TestClient
        from PIL import Image
        app,server = create_app(a.designer,a.scenesmith)
        client = TestClient(app)
        def post(route,payload=None):
            start=time.monotonic()
            response=client.post(route,json=payload or {})
            data=response.json()
            requests.append({'route':route,'payload':payload,'status_code':response.status_code,
                             'elapsed_seconds':time.monotonic()-start,'response':data})
            dump(a.out/'HTTP_REQUESTS.json',requests)
            if response.status_code!=200:
                raise RuntimeError(route+': '+str(response.status_code)+' '+str(data))
            return data
        def check(name,condition):
            checks.append({'name':name,'passed':bool(condition)})
            if not condition:raise AssertionError(name)
        post('/session/existing',{'bundle_directory':str(a.bundle.resolve()),'output_directory':str((a.out/'house').resolve())})
        session=server._state.session
        scene=session.scene
        expected={row['object_id'] for row in json.loads((a.bundle/'provenance.json').read_text())['objects']}
        check('all_21_initial_furniture_ids_present',len(expected)==21 and expected<=set(scene.objects))
        check('six_structural_walls_and_five_openings',len(scene.room_geometry.walls)==6 and len(scene.room_geometry.openings)==5)
        initial_state=scene.to_state_dict()
        dump(a.out/'native_initial_state.json',initial_state)
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        observation=post('/session/observation')
        dump(a.out/'OBSERVATION.json',observation)
        check('five_current_state_images',len(observation['image_paths'])==5)
        for path in observation['image_paths']:
            with Image.open(path) as image:image.verify()
        check('observation_did_not_modify_state',canonical_hash(initial_state)==canonical_hash(scene.to_state_dict()))
        context=session.current_context()
        check('all_objects_and_assets_visible',set(context['objects'])==set(scene.objects)==set(context['asset_bindings']))
        check('raw_task_and_unknowns_visible',bool(context['design_request']) and bool(context['unknowns']))
        dump(a.out/'CURRENT_CONTEXT.json',context)
        post('/session/step',{'tool_calls':[{'name':'get_current_scene_state','arguments':{}}]})
        check('real_state_query_preserves_geometry',canonical_hash(initial_state)==canonical_hash(scene.to_state_dict()))
        checkpoint=post('/session/checkpoint',{'output_directory':str((a.out/'checkpoint').resolve())})
        check('checkpoint_manifest_saved',(a.out/'checkpoint/manifest.json').is_file())
        bounds={oid:scene.objects[oid].compute_world_bounds() for oid in expected}
        floor=scene.room_geometry
        inside_lo=np.array([-floor.length/2,-floor.width/2,0])
        inside_hi=np.array([floor.length/2,floor.width/2,floor.wall_height])
        boundary=[]
        for oid,(lo,hi) in sorted(bounds.items()):
            excursion=np.maximum(0,np.maximum(inside_lo-lo,hi-inside_hi))
            if excursion.max()>1e-6:boundary.append({'object_id':oid,'outside_room_envelope_m':excursion.tolist()})
        overlaps=[]
        for left,right in itertools.combinations(sorted(expected),2):
            if any(scene.objects[k].object_type.value=='thin_covering' for k in (left,right)):
                continue
            lo=np.maximum(bounds[left][0],bounds[right][0]);hi=np.minimum(bounds[left][1],bounds[right][1])
            ext=hi-lo
            if np.all(ext>1e-4):overlaps.append({'object_ids':[left,right],'aabb_overlap_extents_m':ext.tolist(),
                                             'aabb_overlap_volume_m3':float(np.prod(ext))})
        dump(a.out/'ENVELOPE_DIAGNOSTICS.json',{'boundary_excursions':boundary,'envelope_overlaps':overlaps,
            'thin_coverings_excluded_from_overlap_pairs':True,'collision_truth':False,
            'meaning':'AABB warnings are conservative and cannot establish collision-free geometry or activity suitability.'})
        result.update(status='native_observation_and_query_completed',furniture_objects=21,native_objects=len(scene.objects),
            observations=1,images=5,tool_calls=1,checkpoint=checkpoint,
            state_sha256=canonical_hash(scene.to_state_dict()),boundary_warning_count=len(boundary),
            envelope_overlap_pair_count=len(overlaps),visual_review='pending',whole_room_success='unknown',
            professional_sample_admitted=False)
    except BaseException as exc:
        result.update(status='failed',failure=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if server is not None:server._close_all()
        if client is not None:client.close()
        import psutil
        remaining=[]
        for process in psutil.Process().children(recursive=True):
            try:
                if os.getpgid(process.pid)==os.getpgrp():remaining.append(process);process.terminate()
            except (ProcessLookupError,psutil.Error):pass
        _,alive=psutil.wait_procs(remaining,timeout=5)
        for process in alive:
            try:process.kill()
            except psutil.Error:pass
        result['owned_descendants_cleaned']=[p.pid for p in remaining]
        result['ended_at']=datetime.now(timezone.utc).isoformat()
        dump(a.out/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'checks':len(checks),'failure':result.get('failure')}),flush=True)


if __name__=='__main__':
    main()
