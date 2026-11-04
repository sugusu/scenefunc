"""Explicitly migrate one reviewed label-rendering change and restore the room.

No edits, renders or tool calls are replayed. All other code and input bytes
must match. The original checkpoint is retained unchanged.
"""
import argparse
import copy
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import shutil

from existing_project_session import create_app
from native_checkpoint import canonical_hash, runtime_fingerprint, rotation_roundtrip_differences
from native_project import sha256, validate_bundle


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source, designer, smith = (Path(plan[k]) for k in ('source', 'designer', 'scenesmith'))
    print(datetime.now(timezone.utc).isoformat(), 'begin explicit room label checkpoint migration', flush=True)
    faulthandler.dump_traceback_later(45, repeat=True)
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'scene_edits': 0, 'renders': 0, 'tool_calls': 0, 'model_calls': 0,
              'training_updates': 0, 'checks': []}
    def check(name, condition):
        result['checks'].append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)
    server = client = None
    try:
        check('frozen_source_manifest', sha256(source/'manifest.json')==plan['source_manifest_sha256'])
        manifest, state, runtime, task = validate_bundle(source)
        session_path = manifest['inputs']['session']
        record = json.loads((source/session_path).read_text())
        current = runtime_fingerprint(designer, smith)
        changed = {k for k in set(current)|set(record['runtime_fingerprint'])
                   if current.get(k)!=record['runtime_fingerprint'].get(k)}
        key = 'designer/src/render_observation.py'
        check('only_declared_renderer_change', changed=={key}
              and record['runtime_fingerprint'][key]==plan['renderer_before_sha256']
              and current[key]==plan['renderer_after_sha256'])
        migrated = args.out/'checkpoint'
        shutil.copytree(source, migrated)
        migrated_record = copy.deepcopy(record)
        migrated_record['runtime_fingerprint'] = current
        dump(migrated/session_path, migrated_record)
        migration = {'source_manifest_sha256': plan['source_manifest_sha256'],
            'reason': 'Reviewed compact native labels, with full tool IDs retained in the annotation map; no state/tool change.',
            'changed_session_fields': ['runtime_fingerprint'], 'runtime_code_change': {
                key: {'before': record['runtime_fingerprint'][key], 'after': current[key]}},
            'scene_memory_receipts_counters_unchanged': True, 'plan_sha256': sha256(args.plan)}
        new_manifest = copy.deepcopy(manifest)
        new_manifest['files'][session_path] = {'bytes': (migrated/session_path).stat().st_size,
                                              'sha256': sha256(migrated/session_path)}
        new_manifest['renderer_migration'] = migration
        dump(migrated/'manifest.json', new_manifest)
        dump(args.out/'MIGRATION.json', migration)
        validate_bundle(migrated)
        check('all_other_input_files_unchanged', all(sha256(source/f)==sha256(migrated/f)
              for f in manifest['files'] if f!=session_path))
        for key in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                    'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
            os.environ.pop(key, None)
        os.environ.update(OPENAI_API_KEY='disabled-local-validation', OPENAI_BASE_URL='http://127.0.0.1:9/v1',
            OPENAI_AGENTS_DISABLE_TRACING='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', NO_PROXY='127.0.0.1,localhost', OPENSCENEAGENT_STUB_ASSETS='0')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        app, server = create_app(designer, smith)
        from fastapi.testclient import TestClient
        client = TestClient(app)
        requests = []
        def post(route, payload):
            print(datetime.now(timezone.utc).isoformat(), 'request', route, flush=True)
            response = client.post(route, json=payload)
            requests.append({'route': route, 'payload': payload, 'status_code': response.status_code,
                             'response': response.json()})
            dump(args.out/'HTTP_REQUESTS.json', requests)
            if response.status_code!=200:
                raise RuntimeError(str(requests[-1]))
            return response.json()
        post('/session/existing', {'bundle_directory': str(migrated.resolve()),
                                  'output_directory': str((args.out/'restored').resolve())})
        session = server._state.session
        check('real_session_restores_saved_step_and_receipts', session.resume_record['step_idx']==1
              and session.receipts==record['receipts'])
        differences = rotation_roundtrip_differences(state['rooms'][record['room_id']], session.scene.to_state_dict())
        check('native_geometry_restored_without_pose_change', not differences)
        post('/session/stage', {'stage': 'furniture', 'skip_render': True, 'stub_assets': False})
        env = server._state.env
        check('native_memory_and_step_restored', env.mem==record['memory'] and env.step_idx==1)
        check('all_objects_and_asset_bindings_retained', len(session.scene.objects)==32
              and session.project.task==task
              and session.project.runtime==runtime)
        check('original_checkpoint_retained', sha256(source/'manifest.json')==plan['source_manifest_sha256'])
        result.update(status='migrated_checkpoint_restored', checkpoint=str(migrated),
            checkpoint_manifest_sha256=sha256(migrated/'manifest.json'),
            state_sha256=canonical_hash(session.scene.to_state_dict()), step_idx=env.step_idx)
    except BaseException as error:
        result.update(status='failed', failure=type(error).__name__+': '+str(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if server is not None:
            server._close_all()
        if client is not None:
            client.close()
        import psutil
        owned = []
        for process in psutil.Process().children(recursive=True):
            try:
                if os.getpgid(process.pid)==os.getpgrp():
                    owned.append(process)
                    process.terminate()
            except (ProcessLookupError, psutil.Error):
                pass
        _, alive = psutil.wait_procs(owned, timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        result['ended_at'] = datetime.now(timezone.utc).isoformat()
        dump(args.out/'RESULT.json', result)
        print(json.dumps({'status': result['status'], 'checks': len(result['checks']), 'failure': result.get('failure')}), flush=True)


if __name__=='__main__':
    main()
