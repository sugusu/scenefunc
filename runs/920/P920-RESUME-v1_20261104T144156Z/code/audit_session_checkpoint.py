"""Cold-process checkpoint/continuation audit of the real imported native session."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from existing_project_session import create_app, canonical_hash
from native_project import import_project, sha256, validate_bundle


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('prepare', 'resume'), required=True)
    p.add_argument('--designer', type=Path, required=True)
    p.add_argument('--scenesmith', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    out = args.run / args.phase
    out.mkdir(parents=True, exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(), 'begin native checkpoint phase', args.phase, flush=True)
    for name in ('SCENESMITH_LLM_BASE_URL', 'SCENESMITH_LLM_API_KEY', 'OPENAI_API_KEY',
                 'GOOGLE_API_KEY', 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'https_proxy', 'http_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation', OPENAI_BASE_URL='http://127.0.0.1:9/v1',
                      OPENAI_AGENTS_DISABLE_TRACING='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', NO_PROXY='127.0.0.1,localhost', SCENE_RENDER_THREADS='4',
                      SCENE_RENDER_SAMPLES='8', OPENSCENEAGENT_STUB_ASSETS='0')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    checks, requests = [], []
    result = {'phase': args.phase, 'status': 'running', 'pid': os.getpid(), 'model_calls': 0,
              'training_updates': 0, 'checks': checks, 'tool_calls': 0, 'observations': 0}
    app = server = client = None
    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)
    try:
        app, server = create_app(args.designer, args.scenesmith)
        from fastapi.testclient import TestClient
        client = TestClient(app)
        def post(route, payload=None, expected=200):
            start = time.monotonic()
            response = client.post(route, json=payload or {})
            data = response.json()
            requests.append({'route': route, 'request': payload, 'status_code': response.status_code,
                             'elapsed_seconds': time.monotonic() - start, 'response': data})
            dump(out / 'HTTP_REQUESTS.json', requests)
            if response.status_code != expected:
                raise RuntimeError(f'{route}: expected {expected}, got {response.status_code}: {data}')
            return data
        def load(source, name='house', expected=200):
            return post('/session/existing', {'bundle_directory': str(source.resolve()),
                         'output_directory': str((out/name).resolve())}, expected)
        checkpoint = args.run / 'checkpoint_after_prepare'
        if args.phase == 'prepare':
            original_manifest = sha256(args.bundle/'manifest.json')
            load(args.bundle)
            session = server._state.session
            check('summary_has_only_actual_wall', '"wall_count": 1' in session.room_summary and 'Wall B' not in session.room_summary)
            post('/session/stage', {'stage': 'furniture'})
            move = {'name': 'move_furniture_tool', 'arguments': {'object_id': 'table_0', 'x': -.4, 'y': .75, 'yaw': 20.}}
            moved = post('/session/step', {'tool_calls': [move]}); result['tool_calls'] += 1
            check('actual_move_before_checkpoint', moved['results'][0]['output']['success'] is True)
            state_after_move = session.scene.to_state_dict()
            rejected = post('/session/step', {'tool_calls': [dict(move, arguments=dict(move['arguments'], object_id='north_wall'))]})
            result['tool_calls'] += 1
            check('rejected_attempt_preserved', rejected['results'][0]['output']['success'] is False and session.scene.to_state_dict() == state_after_move)
            saved = post('/session/checkpoint', {'output_directory': str(checkpoint.resolve())})
            validate_bundle(checkpoint)
            check('checkpoint_is_native_valid_bundle', saved['step_idx'] == 2)
            expected = {'prepare_pid': os.getpid(), 'scene_state': session.scene.to_state_dict(),
                        'house_state': session.project.house.to_state_dict(), 'memory': server._state.env.mem,
                        'receipts': session.receipts, 'surface_counter': session.scene._surface_id_counter,
                        'original_request': session.project.task, 'manifest_sha256': sha256(checkpoint/'manifest.json')}
            dump(args.run/'EXPECTED.json', expected)
            post('/session/checkpoint', {'output_directory': str(checkpoint.resolve())}, expected=422)
            check('checkpoint_overwrite_rejected', sha256(checkpoint/'manifest.json') == expected['manifest_sha256'])
            check('original_bundle_unchanged', sha256(args.bundle/'manifest.json') == original_manifest)
        else:
            expected = json.loads((args.run/'EXPECTED.json').read_text())
            check('separate_process', os.getpid() != expected['prepare_pid'])
            # These deliberately invalid copies have fresh file hashes so the
            # semantic guards, not just byte corruption detection, are checked.
            for kind in ('code', 'history'):
                bad = out/f'invalid_{kind}'
                shutil.copytree(checkpoint, bad)
                record_path = bad/'session_checkpoint.json'
                record = json.loads(record_path.read_text())
                if kind == 'code':
                    record['runtime_fingerprint']['adapter/existing_project_session.py'] = '0'*64
                else:
                    record['receipts'][-1]['step_idx'] = 9
                dump(record_path, record)
                manifest = json.loads((bad/'manifest.json').read_text())
                manifest['files']['session_checkpoint.json'] = {'sha256': sha256(record_path), 'bytes': record_path.stat().st_size}
                dump(bad/'manifest.json', manifest)
                response = load(bad, name=f'rejected_{kind}', expected=422)
                check(f'{kind}_mismatch_rejected_without_active_session', server._state.session is None)
            loaded = load(checkpoint)
            session = server._state.session
            check('cold_load_preserves_house', session.project.house.to_state_dict() == expected['house_state'])
            check('cold_load_preserves_counter_and_task', session.scene._surface_id_counter == expected['surface_counter'] and session.project.task == expected['original_request'])
            check('checkpoint_reports_resume_step', loaded['resumed_step_idx'] == 2)
            post('/session/stage', {'stage': 'furniture'})
            env = server._state.env
            check('native_memory_and_step_restored', env.step_idx == 2 and env.mem == expected['memory'])
            check('actual_history_including_rejection_restored', session.receipts == expected['receipts'])
            observation = post('/session/observation'); result['observations'] += 1
            dump(out/'OBSERVATION.json', observation)
            from PIL import Image
            check('five_current_state_images', len(observation['image_paths']) == 5)
            for path in observation['image_paths']:
                with Image.open(path) as image:
                    image.verify()
            binding = json.loads(Path(observation['state_binding']).read_text())
            check('resumed_views_bind_to_checkpoint_state', binding['state_sha256'] == canonical_hash(expected['scene_state']))
            context = json.loads(observation['room_prompt'].split('CURRENT NATIVE STATE:\n', 1)[1])
            check('real_prior_receipts_visible', context['prior_tool_receipts'] == expected['receipts'] and context['total_tool_batches'] == 2)
            labels = json.loads(Path(observation['image_paths'][0]+'.annotations.json').read_text())
            check('topdown_uses_actual_wall_identity', [item[0] for item in labels['wall_items']] == ['north_wall'])
            moved = post('/session/step', {'tool_calls': [{'name': 'move_furniture_tool', 'arguments': {
                'object_id': 'table_0', 'x': .8, 'y': -.2, 'yaw': -15.}}]}); result['tool_calls'] += 1
            check('continued_tool_uses_same_instance', moved['results'][0]['output']['success'] is True
                  and set(session.scene.objects) == set(expected['scene_state']['objects']))
            check('continued_step_and_history_advance', env.step_idx == 3 and len(session.receipts) == 3
                  and session.receipts[:2] == expected['receipts'])
            final = args.run/'checkpoint_after_resume'
            post('/session/checkpoint', {'output_directory': str(final.resolve())})
            restored = import_project(final, out/'final_offline_restore')
            from native_checkpoint import load_session_record
            record = load_session_record(restored)
            check('continued_checkpoint_restores_latest_native_state', restored.house.to_state_dict() == session.project.house.to_state_dict()
                  and record['step_idx'] == 3 and record['memory'] == env.mem)
            check('first_checkpoint_unchanged', sha256(checkpoint/'manifest.json') == expected['manifest_sha256'])
            dump(out/'FINAL_STATE.json', session.project.house.to_state_dict())
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        if server is not None:
            server._close_all()
        if client is not None:
            client.close()
        import psutil
        remaining = []
        for process in psutil.process_iter(['pid']):
            try:
                if process.pid != os.getpid() and os.getpgid(process.pid) == os.getpgrp():
                    remaining.append(process); process.terminate()
            except (ProcessLookupError, PermissionError, psutil.Error):
                pass
        _, alive = psutil.wait_procs(remaining, timeout=3)
        for process in alive:
            try: process.kill()
            except psutil.Error: pass
        result.update(extra_owned_children_cleaned=[p.pid for p in remaining],
                      completed_at=datetime.now(timezone.utc).isoformat())
        dump(out/'RESULT.json', result)
        print(json.dumps({'phase': args.phase, 'status': result['status'], 'checks': len(checks),
                          'failure': result.get('failure')}), flush=True)


if __name__ == '__main__':
    main()
