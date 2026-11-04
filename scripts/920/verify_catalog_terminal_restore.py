"""Cold-load a completed native catalog session without any model/edit/query call."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from catalog_project_session import create_app,canonical_hash,TOOLS
from native_checkpoint import rotation_roundtrip_differences
from native_project import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(exist_ok=False)
    for name,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Frozen restore code changed')
    source=Path(plan['bundle'])
    if sha256(source/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Bundle changed')
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH',
                 'SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','GOOGLE_API_KEY'):
        os.environ.pop(name,None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0',
        TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    result={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','model_calls':0,'tool_calls':0,'scene_edits':0}
    print(result['started_at'],'begin terminal cold restoration',flush=True)
    from fastapi.testclient import TestClient
    app,server=create_app(Path(plan['designer']),Path(plan['scenesmith']));client=TestClient(app)
    def post(route,payload):
        r=client.post(route,json=payload)
        if r.status_code!=200:raise RuntimeError('Native restore route failed '+route+': '+r.text[:1000])
        return r.json()
    try:
        original=json.loads((source/'house_state.json').read_text())['rooms']['geneva_development']
        record=json.loads((source/'session_checkpoint.json').read_text())
        post('/session/catalog',{'bundle_directory':str(source),'output_directory':str(a.out/'house')})
        session=server._state.session
        post('/session/stage',{'stage':'furniture','stub_assets':False,'skip_render':False})
        state=session.scene.to_state_dict();differences=rotation_roundtrip_differences(original,state)
        restored=session.capture_runtime(session.active_environment)
        checks={'step_idx_exact':restored['step_idx']==record['step_idx']==24,
            'all_24_receipts_exact':restored['receipts']==record['receipts'] and len(restored['receipts'])==24,
            'native_memory_exact':restored['memory']==record['memory'],
            'reserved_instance_ids_exact':restored['reserved_object_ids']==record['reserved_object_ids'],
            'tool_scope_exact':restored['tool_names']==record['tool_names']==list(TOOLS),
            'catalog_exact':restored['catalog_sha256']==record['catalog_sha256'],
            'configuration_exact':restored['effective_config_sha256']==record['effective_config_sha256'],
            'runtime_code_exact':restored['runtime_fingerprint']==record['runtime_fingerprint'],
            'placement_noise_exact':restored['active_noise_profile']==record['active_noise_profile'],
            'current_ids_exact':set(state['objects'])==set(original['objects'])}
        result.update(checks=checks,rotation_roundtrip_differences=differences,
            original_state_sha256=canonical_hash(original),restored_state_sha256=canonical_hash(state),
            input_manifest_sha256=plan['manifest_sha256'],plan_sha256=sha256(a.plan),
            unavailable_instance_ids=session.current_context()['unavailable_instance_ids'])
        if not all(checks.values()):raise ValueError('Terminal restoration invariant failed')
        result['status']='passed_native_terminal_cold_restore_no_edits'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        try:client.post('/session/close',json={})
        finally:client.close()
        result['ended_at']=datetime.now(timezone.utc).isoformat()
        (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        print(result['ended_at'],result['status'],flush=True)


if __name__=='__main__':main()
