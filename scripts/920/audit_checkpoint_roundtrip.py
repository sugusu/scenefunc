"""Continue the saved third-step state, retaining explicit native round-off events."""
from __future__ import annotations
import argparse, copy, json, os, shutil, time
from datetime import datetime, timezone
from pathlib import Path
from existing_project_session import create_app
from native_checkpoint import (canonical_hash, load_session_record, rotation_roundtrip_differences,
                               runtime_fingerprint)
from native_project import import_project, read_json, sha256, validate_bundle


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--previous-protocol',type=Path,required=True)
    p.add_argument('--designer',type=Path,required=True)
    p.add_argument('--scenesmith',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(),'begin checkpoint round-trip continuation',flush=True)
    for name in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                 'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(name,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
                      OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1',NO_PROXY='127.0.0.1,localhost',OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    checks=[];requests=[];server=client=None
    result={'status':'running','pid':os.getpid(),'model_calls':0,'training_updates':0,'renders':0,
            'tool_calls':0,'edits_replayed':0,'checks':checks}
    def check(name,ok):
        checks.append({'name':name,'passed':bool(ok)})
        if not ok:raise AssertionError(name)
    try:
        manifest,state,runtime,task=validate_bundle(args.source)
        source_hash=sha256(args.source/'manifest.json')
        record=read_json(args.source/manifest['inputs']['session'])
        previous=read_json(args.previous_protocol)
        fingerprint=runtime_fingerprint(args.designer,args.scenesmith)
        allowed=['adapter/existing_project_session.py','adapter/native_checkpoint.py']
        changed=sorted(k for k in set(fingerprint)|set(record['runtime_fingerprint'])
                       if fingerprint.get(k)!=record['runtime_fingerprint'].get(k))
        check('only_frozen_recovery_helpers_changed',changed==allowed and all(
            record['runtime_fingerprint'][key]==previous['scripts']['scripts/920/'+key.split('/')[-1]] for key in allowed))
        migrated=args.out/'migrated_checkpoint';shutil.copytree(args.source,migrated)
        migrated_record=copy.deepcopy(record);migrated_record['runtime_fingerprint']=fingerprint
        session_file=migrated/manifest['inputs']['session'];dump(session_file,migrated_record)
        new_manifest=copy.deepcopy(manifest)
        new_manifest['files'][manifest['inputs']['session']]={'bytes':session_file.stat().st_size,'sha256':sha256(session_file)}
        migration={'source_manifest_sha256':source_hash,'changed_fields':['runtime_fingerprint'],
                   'reason':'Record strictly bounded quaternion restoration differences as non-tool transitions',
                   'runtime_code_changes':{key:{'before':record['runtime_fingerprint'][key],'after':fingerprint[key]} for key in allowed},
                   'tools_replayed':0,'scene_memory_receipts_counters_unchanged':True}
        new_manifest['restoration_migration']=migration
        dump(migrated/'manifest.json',new_manifest);dump(args.out/'MIGRATION.json',migration)
        validate_bundle(migrated)
        check('migration_preserves_every_other_input_file',all(sha256(args.source/f)==sha256(migrated/f)
                    for f in manifest['files'] if f!=manifest['inputs']['session']))
        for kind in ('translation','large_rotation','identity'):
            changed_state=copy.deepcopy(state['rooms']['store_b'])
            obj=changed_state['objects']['table_0']
            if kind=='translation':obj['transform']['translation'][0]+=1e-9
            elif kind=='large_rotation':obj['transform']['rotation_wxyz'][3]+=1e-6
            else:obj['object_id']='other_instance'
            rejected=False
            try:rotation_roundtrip_differences(state['rooms']['store_b'],changed_state)
            except ValueError:rejected=True
            check(f'{kind}_change_is_not_treated_as_roundoff',rejected)
        app,server=create_app(args.designer,args.scenesmith)
        from fastapi.testclient import TestClient
        client=TestClient(app)
        def post(route,payload=None):
            start=time.monotonic();response=client.post(route,json=payload or {});data=response.json()
            requests.append({'route':route,'request':payload,'response':data,'status_code':response.status_code,
                             'elapsed_seconds':time.monotonic()-start});dump(args.out/'HTTP_REQUESTS.json',requests)
            if response.status_code!=200:raise RuntimeError(f'{route}: {response.status_code}: {data}')
            return data
        loaded=post('/session/existing',{'bundle_directory':str(migrated.resolve()),'output_directory':str((args.out/'house').resolve())})
        session=server._state.session
        check('resume_begins_after_third_actual_tool',loaded['resumed_step_idx']==3 and session.receipts==record['receipts'])
        differences=rotation_roundtrip_differences(state['rooms']['store_b'],session.scene.to_state_dict())
        check('exact_quaternion_difference_recorded',len(differences)==1 and session.restoration_transitions[-1]['differences']==differences
              and session.restoration_transitions[-1]['before_sha256']==record['receipts'][-1]['after_sha256'])
        check('identity_task_and_surface_counter_retained',session.project.task==task
              and session.scene._surface_id_counter==runtime['rooms']['store_b']['next_surface_id']
              and set(session.scene.objects)==set(state['rooms']['store_b']['objects']))
        post('/session/stage',{'stage':'furniture'});env=server._state.env
        check('memory_and_step_restored',env.mem==record['memory'] and env.step_idx==3)
        before=canonical_hash(session.scene.to_state_dict())
        response=post('/session/step',{'tool_calls':[{'name':'get_current_scene_state','arguments':{}}]});result['tool_calls']+=1
        check('real_state_query_continues_without_edit',env.step_idx==4 and canonical_hash(session.scene.to_state_dict())==before
              and response['results'][0]['name']=='get_current_scene_state' and 'ERROR' not in str(response['results'][0]['output']))
        final=args.out/'checkpoint_after_query';post('/session/checkpoint',{'output_directory':str(final.resolve())})
        restored=import_project(final,args.out/'final_offline_restore');final_record=load_session_record(restored)
        check('query_checkpoint_reloads_with_continuous_history',final_record['step_idx']==4
              and final_record['receipts'][:3]==record['receipts'] and final_record['memory']==env.mem)
        check('restoration_is_recorded_separately_from_tools',len(final_record['receipts'])==4
              and len(final_record['restoration_transitions'])>=1)
        check('current_objects_and_assets_retained',set(restored.room.objects)==set(session.scene.objects)
              and restored.runtime['rooms']['store_b']['asset_bindings']==session.project.runtime['rooms']['store_b']['asset_bindings'])
        check('source_checkpoint_unchanged',sha256(args.source/'manifest.json')==source_hash)
        validate_bundle(args.source)
        dump(args.out/'RESTORED_SESSION_RECORD.json',final_record)
        dump(args.out/'FINAL_RESTORED_STATE.json',restored.house.to_state_dict())
        result.update(status='passed',surface_counter=restored.room._surface_id_counter,
                      restored_step_idx=4,recorded_restoration_transitions=final_record['restoration_transitions'])
    except BaseException as error:
        result.update(status='failed',failure=f'{type(error).__name__}: {error}');raise
    finally:
        if server is not None:server._close_all()
        if client is not None:client.close()
        import psutil
        remaining=[]
        for process in psutil.process_iter(['pid']):
            try:
                if process.pid!=os.getpid() and os.getpgid(process.pid)==os.getpgrp():remaining.append(process);process.terminate()
            except (ProcessLookupError,PermissionError,psutil.Error):pass
        _,alive=psutil.wait_procs(remaining,timeout=3)
        for process in alive:
            try:process.kill()
            except psutil.Error:pass
        result.update(extra_owned_children_cleaned=[p.pid for p in remaining],completed_at=datetime.now(timezone.utc).isoformat())
        dump(args.out/'RESULT.json',result);print(json.dumps({'status':result['status'],'checks':len(checks),'failure':result.get('failure')}),flush=True)


if __name__=='__main__':main()
