"""Render one immutable native input observation, without policy or actions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from catalog_project_session import create_app, canonical_hash
from native_project import sha256


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for name,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Code hash differs: '+name)
    if sha256(Path(plan['bundle'])/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Input hash differs')
    for name in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH'):
        os.environ.pop(name,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    now=lambda:datetime.now(timezone.utc).isoformat()
    result={'status':'running','started_at':now(),'model_calls':0,'tool_actions':0,'training_updates':0,'views':[],'plan_sha256':sha256(a.plan)}
    public=Path(plan['public_directory']);client=None
    def progress():
        write(a.out/'RESULT.json',result)
        write(public/'STATUS.json',{'run_id':plan['run_id'],'status':result['status'],'updated_at':now(),'question':plan['question'],'run_directory':plan['runtime_directory'],'completed_planned':{'views':[len(result['views']),5]},'configuration_hashes':{'plan':sha256(a.plan),'input_manifest':plan['manifest_sha256']},'latest_verifiable_artifact':str(a.out/'RESULT.json'),'failure_reason':result.get('error'),'started_at':result['started_at'],'ended_at':result.get('ended_at'),'next_step':'Review actual five views and retain all nine activity requirements before generation.'})
        write(public.parent/'latest.json',{'run_id':plan['run_id'],'status':result['status'],'stage':'native_initial_observation','updated_at':now(),'run_directory':plan['runtime_directory']})
    print(now(),'begin native input observation',flush=True);progress()
    try:
        from fastapi.testclient import TestClient
        from PIL import Image
        app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
        def post(route,payload=None):
            resp=client.post(route,json=payload or {})
            if resp.status_code!=200:raise RuntimeError(route+': '+resp.text[:1000])
            return resp.json()
        post('/session/catalog',{'bundle_directory':plan['bundle'],'output_directory':str(a.out/'house')})
        session=server._state.session;before=session.scene.to_state_dict();write(a.out/'initial_state.json',before)
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        write(a.out/'current_context.json',session.current_context())
        observation=post('/session/observation');write(a.out/'observation.json',observation)
        bound=json.loads(Path(observation['state_binding']).read_text())
        assert len(observation['image_paths'])==5
        assert bound['state_sha256']==canonical_hash(before)==canonical_hash(session.scene.to_state_dict())
        assert [x['path'] for x in bound['images']]==observation['image_paths']
        for record in bound['images']:
            path=Path(record['path']);assert sha256(path)==record['sha256']
            with Image.open(path) as im:
                size=list(im.size);im.verify()
            result['views'].append({**record,'size':size})
        result.update(status='five_views_bound_to_unchanged_initial_state',state_sha256=bound['state_sha256'],catalog_entries=len(session.catalog['entries']),furniture_instances=len(before['objects']),observation=str(a.out/'observation.json'),formal_evaluation_ready=False)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if client is not None:
            try:client.post('/session/close',json={})
            finally:client.close()
        result['ended_at']=now();progress();print(now(),result['status'],flush=True)


if __name__=='__main__':main()
