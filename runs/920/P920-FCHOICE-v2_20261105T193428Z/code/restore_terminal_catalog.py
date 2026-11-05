"""Load a terminal native catalog checkpoint in a fresh process, without edits."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from catalog_project_session import create_app
from native_checkpoint import canonical_hash,rotation_roundtrip_differences
from native_project import sha256,validate_bundle


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(exist_ok=False)
    for name,h in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=h:raise ValueError('Frozen code changed: '+name)
    source=Path(plan['bundle']);assert sha256(source/'manifest.json')==plan['manifest_sha256']
    for name in ['SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH']:os.environ.pop(name,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_PROXY='127.0.0.1,localhost',OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);now=lambda:datetime.now(timezone.utc).isoformat()
    record={'status':'running','started_at':now(),'model_calls':0,'tool_edits':0,'render_calls':0,'training_updates':0,'plan_sha256':sha256(a.plan)};client=None
    print(now(),'begin terminal native cold restore',flush=True)
    try:
        manifest,state,runtime,task=validate_bundle(source);rid=manifest['selected_room_id'];before=state['rooms'][rid]
        from fastapi.testclient import TestClient
        app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
        response=client.post('/session/catalog',json={'bundle_directory':str(source),'output_directory':str(a.out/'restored')});write(a.out/'LOAD_REPLY.json',{'status_code':response.status_code,'response':response.json()});assert response.status_code==200
        session=server._state.session;after=session.scene.to_state_dict();differences=rotation_roundtrip_differences(before,after)
        original=json.loads((source/'session_checkpoint.json').read_text());assert len(session.receipts)==original['step_idx']==plan['expected_tool_calls']
        assert session.receipts==original['receipts'] and session.reserved_object_ids==set(original['reserved_object_ids'])
        assert session.current_context()['asset_bindings']==runtime['rooms'][rid]['asset_bindings']
        assert len(session.catalog['entries'])==plan['expected_catalog_entries']
        assert session.active_environment is None and response.json()['backend_generation_started'] is False
        assert sha256(source/'manifest.json')==plan['manifest_sha256']
        write(a.out/'RESTORED_STATE.json',after);write(a.out/'RESTORED_CONTEXT.json',session.current_context())
        record.update(status='terminal_native_session_restored_without_edits',source_state_sha256=canonical_hash(before),restored_state_sha256=canonical_hash(after),quaternion_roundtrip_differences=differences,all_object_ids=sorted(after['objects']),furniture_ids=sorted(k for k,v in after['objects'].items() if v['object_type']=='furniture'),catalog_entries=len(session.catalog['entries']),receipt_count=len(session.receipts),reserved_ids_preserved=True,asset_bindings_exact=True,actual_edit_dispatch_tested=False,whole_room_success=False)
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        if client is not None:
            try:client.post('/session/close',json={})
            finally:client.close()
        record['ended_at']=now();write(a.out/'RESULT.json',record);print(now(),record['status'],flush=True)


if __name__=='__main__':main()
