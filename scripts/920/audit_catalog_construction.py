"""Exercise native catalog placement and cold-process checkpoint continuation.

Commands are researcher engineering inputs, not a model or professional layout.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from catalog_project_session import create_app, canonical_hash, SCOPE
from native_project import sha256


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','resume'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for filename,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/filename)!=digest:raise ValueError('Frozen code changed')
    for name in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                 'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(name,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    print(datetime.now(timezone.utc).isoformat(),'begin catalog',a.phase,flush=True)
    result={'status':'running','phase':a.phase,'checks':[],'tool_calls':0,'images':[],
        'model_calls':0,'training_updates':0,'whole_room_design_success':'not_evaluated_engineering_commands_only'}
    requests=[];client=server=None
    def check(name,condition):
        result['checks'].append({'name':name,'passed':bool(condition)})
        if not condition:raise ValueError(name)
    try:
        from fastapi.testclient import TestClient
        from PIL import Image
        app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
        def post(route,payload=None):
            response=client.post(route,json=payload or {});data=response.json()
            requests.append({'route':route,'payload':payload,'status_code':response.status_code,'response':data})
            dump(a.out/'HTTP_REQUESTS.json',requests)
            if response.status_code!=200:raise RuntimeError('HTTP route failed: '+route)
            return data
        source=Path(plan['bundle'] if a.phase=='prepare' else plan['prepared_checkpoint'])
        if a.phase=='prepare':check('frozen_input_manifest',sha256(source/'manifest.json')==plan['manifest_sha256'])
        loaded=post('/session/catalog',{'bundle_directory':str(source.resolve()),'output_directory':str((a.out/'house').resolve())})
        session=server._state.session;scene=session.scene
        check('explicit_construction_scope',loaded['scope']==SCOPE)
        check('catalog_reconstructed',len(session.catalog['entries'])==plan['catalog_entries'])
        before=scene.to_state_dict();dump(a.out/'initial_state.json',before)
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        check('native_registry_restored',session.active_environment.agent.furniture_tools.asset_manager.registry.size()==plan['catalog_entries'])
        check('native_addition_advertised','add_furniture_to_scene_tool' in session.active_environment.tool_map)
        context=session.current_context();dump(a.out/'initial_context.json',context)
        check('catalog_in_actual_context',len(context['asset_catalog'])==plan['catalog_entries'])
        if a.phase=='resume':
            previous=json.loads(Path(plan['prepare_result']).read_text())
            check('cold_process_restores_actual_ids',set(before['objects'])==set(previous['object_ids']))
            check('cold_process_retains_receipts',len(session.receipts)==3 and loaded['resumed_step_idx']==3)
            check('cold_process_retains_catalog',session.catalog_sha256==previous['catalog_sha256'])
            check('cold_process_retains_bindings',session.current_context()['asset_bindings']==previous['asset_bindings'])
            if 'expected_added_ids' in plan:
                check('cold_process_retains_retired_ids',session.reserved_object_ids==set(previous['reserved_object_ids']))
        calls=plan['prepare_calls'] if a.phase=='prepare' else plan['resume_calls']
        added=[]
        for index,command in enumerate(calls):
            old=scene.to_state_dict()
            reply=post('/session/step',{'tool_calls':[command]});result['tool_calls']+=1
            actual=scene.to_state_dict();output=reply['results'][0]['output']
            if command['name']=='add_furniture_to_scene_tool' and command['arguments']['asset_id']=='missing_catalog_asset':
                check('unknown_asset_rejected',isinstance(output,dict) and output.get('success') is False)
                check('rejection_preserves_state',actual==old)
            elif command['name']=='add_furniture_to_scene_tool':
                check('native_add_accepted_'+str(index),isinstance(output,dict) and output.get('success') is True)
                oid=output['object_id'];added.append(oid)
                check('native_new_instance_'+str(index),oid not in old['objects'] and set(actual['objects'])==set(old['objects'])|{oid})
                check('other_objects_preserved_'+str(index),all(actual['objects'][k]==v for k,v in old['objects'].items()))
                binding=session.current_context()['asset_bindings'][oid]
                check('actual_asset_identity_'+str(index),binding['registry_id']==command['arguments']['asset_id'])
            elif command['name']=='remove_furniture_tool':
                oid=command['arguments']['object_id']
                check('native_remove_accepted_'+str(index),isinstance(output,dict) and output.get('success') is True)
                check('only_requested_instance_removed_'+str(index),set(actual['objects'])==set(old['objects'])-{oid} and
                      all(actual['objects'][k]==old['objects'][k] for k in actual['objects']))
                check('removed_id_remains_reserved_'+str(index),oid in session.reserved_object_ids)
            else:
                check('query_preserves_state',actual==old)
        observation=post('/session/observation');state=scene.to_state_dict()
        bound=json.loads(Path(observation['state_binding']).read_text())
        check('five_views_actual_state',len(observation['image_paths'])==5 and bound['state_sha256']==canonical_hash(state))
        for filename in observation['image_paths']:
            with Image.open(filename) as image:image.verify()
        result['images']=observation['image_paths']
        dump(a.out/'observation.json',observation);dump(a.out/'final_state.json',state)
        checkpoint=post('/session/checkpoint',{'output_directory':str((a.out/'checkpoint').resolve())})
        check('complete_asset_bindings',set(state['objects'])==set(session.current_context()['asset_bindings']))
        result.update(status='native_catalog_'+a.phase+'_completed',object_ids=sorted(state['objects']),
            added_object_ids=added,catalog_sha256=session.catalog_sha256,asset_bindings=session.current_context()['asset_bindings'],
            checkpoint=checkpoint,step_idx=session.active_environment.step_idx)
        if 'expected_added_ids' in plan:
            check('deleted_ids_never_reused',added==plan['expected_added_ids'][a.phase])
            result['reserved_object_ids']=sorted(session.reserved_object_ids)
        if a.phase=='resume':check('continued_step_index',result['step_idx']==5)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        dump(a.out/'RESULT.json',result)
        if client is not None:
            try:client.post('/session/close',json={})
            finally:client.close()
        print(datetime.now(timezone.utc).isoformat(),result['status'],flush=True)


if __name__=='__main__':main()
