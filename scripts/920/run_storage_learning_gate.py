"""Finite inference-only transfer of existing weights to one source-seen task.

No reference coordinates enter requests. Historical runs and weights are read-only.
The initial room and source knowledge are shared; only weights differ by arm.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from native_project import sha256
from native_checkpoint import canonical_hash, rotation_roundtrip_differences
from existing_project_session import create_app
from edit_request import build_request, validate_action
from live_edit_request import exchange
from shared_action_contract import schemas_from_observation, strict_loads
from measure_storage_consolidation import storage_measures
from preflight_sourced_trajectory import inspect


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def compact_measurement(m):
    return dict(
        fixed_access=[dict(width=s['square_body_width_m'], targets={k: dict(
            point=v['requested_point_m'], reachable=v['reachable_from_entry'])
            for k, v in s['targets'].items()}) for s in m['fixed_access']['scenarios']],
        storage_front=[dict(width=s['square_body_width_m'], targets={k: dict(
            point=v['requested_point_m'], reachable=v['reachable_from_entry'])
            for k, v in s['targets'].items()}) for s in m['storage_front_approaches']],
        living_region=m['living_region'],
        largest_empty_rectangle_m2=m['floor_rectangle']['largest_grid_empty_rectangle_m2'],
        limits=m['fixed_access']['limits'])


def score(initial, state, initial_m, m, plan):
    ids = set(plan['storage_ids'])
    same_ids = set(initial['objects']) == set(state['objects'])
    fixed_equal = all(state['objects'].get(k) == v for k, v in initial['objects'].items() if k not in ids)
    cabinet_equal = True
    for oid in ids:
        x, y = copy.deepcopy(initial['objects'][oid]), copy.deepcopy(state['objects'].get(oid, {}))
        x.pop('transform', None); y.pop('transform', None)
        cabinet_equal = cabinet_equal and x == y
    geometry = inspect(state)
    fixed = [sum(v['reachable_from_entry'] for v in s['targets'].values()) for s in m['fixed_access']['scenarios']]
    fronts = [sum(v['reachable_from_entry'] for v in s['targets'].values()) for s in m['storage_front_approaches']]
    gain = m['living_region']['unoccupied_envelope_m2'] - initial_m['living_region']['unoccupied_envelope_m2']
    passed = (same_ids and fixed_equal and cabinet_equal and fixed == [7, 7, 7] and fronts == [15, 15, 15]
              and gain > 1e-9 and not geometry['positive_3d_AABB_pairs'] and not geometry['outside_declared_wall_inner_faces'])
    return dict(conditional_geometric_rule_passed=bool(passed), fixed_access_counts=fixed,
                storage_front_counts=fronts, living_free_gain_m2=gain, geometry=geometry,
                all_object_ids_preserved=same_ids, non_target_records_preserved=fixed_equal,
                cabinet_identity_and_dimensions_preserved=cabinet_equal,
                whole_room_professional_quality='unknown', generalization_test=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase', choices=['prepare', 'rollout'], required=True)
    a = ap.parse_args(); p = read(a.plan)
    a.out.mkdir(parents=True, exist_ok=False)
    for name, digest in p['code_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Code changed: ' + name)
    for item in p['inputs'].values():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Input changed: ' + item['path'])
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
                'LD_LIBRARY_PATH', 'OPENAI_API_KEY', 'GOOGLE_API_KEY', 'SCENESMITH_LLM_API_KEY', 'SCENESMITH_LLM_BASE_URL'):
        os.environ.pop(key, None)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OPENAI_API_KEY='disabled-local-validation', OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1', NO_PROXY='127.0.0.1,localhost', SCENE_RENDER_THREADS='4',
        SCENE_RENDER_SAMPLES='8', OPENSCENEAGENT_STUB_ASSETS='0', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    public = Path(p['public_directory'])
    result = dict(status='running', phase=a.phase, started_at=now(), cell_id=p['cell_id'],
        model_calls=0, model_stage_attempts=0, native_calls=0, tool_attempts=0, training_updates=0,
        rounds=[], gpu_subprocess_seconds=0., plan_sha256=sha256(a.plan))
    def progress():
        write(a.out/'RESULT.json', result)
        write(public/(a.phase.upper()+'_RESULT.json'), result)
        write(public/'STATUS.json', dict(run_id=p['run_id'], question=p['question'], updated_at=now(),
            status=result['status'], stage=a.phase, run_directory=str(a.out),
            completed_planned=dict(model_calls=[result['model_calls'], p['budget']['model_calls']],
                                   native_calls=result['native_calls']),
            configuration_hashes=dict(plan=sha256(a.plan), inputs=p['inputs']),
            latest_verifiable_artifact=str(a.out/'RESULT.json'), failure_reason=result.get('error'),
            next_step='Finish frozen three-arm comparison without retries or training.',
            started_at=result['started_at'], ended_at=result.get('ended_at')))
    def model_call(request, directory, mode, turn):
        remaining = p['budget']['gpu_seconds_per_cell'] - result['gpu_subprocess_seconds']
        if mode == 'generate' and remaining < 15:
            result['stop_reason'] = 'gpu_budget_exhausted'; return None
        child = copy.deepcopy(p); child['inference']['seed'] += turn
        write(directory/'PLAN.json', child)
        timeout = min(p['budget']['seconds_per_model_call'], int(remaining)) if mode == 'generate' else 180
        command = [sys.executable, str(Path(__file__).with_name('run_stage.py')), '--receipt-dir', str(directory/'receipt'),
                   '--timeout', str(timeout), '--', p['model_python'], str(Path(__file__).with_name('edit_study_model_step.py')),
                   '--plan', str(directory/'PLAN.json'), '--request', str(request), '--out', str(directory/'model'), '--mode', mode]
        if mode == 'generate':
            result['model_stage_attempts'] += 1; progress()
        code = subprocess.call(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=p['gpu_uuid'] if mode == 'generate' else ''))
        receipt = read(directory/'receipt/receipt.json')
        record = read(directory/'model/RESULT.json') if (directory/'model/RESULT.json').exists() else {}
        if mode == 'generate':
            result['gpu_subprocess_seconds'] += receipt['elapsed_seconds']
            result['model_calls'] += record.get('model_calls', 0)
        progress()
        if code:
            raise RuntimeError(f'Model stage exit {code}; no retry')
        return record
    progress(); print(now(), 'begin', a.phase, p['cell_id'], flush=True)
    client = server = session = None; locks = []
    try:
        if a.phase == 'rollout':
            for path in p['gpu_lock_paths']:
                lock = Path(path).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
            values = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.used,memory.free',
                '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
            if values[0].strip() != p['gpu_uuid'] or int(values[1]) > 512 or int(values[2]) < 23000:
                raise RuntimeError('GPU0 unavailable')
        prepared = Path(p['prepared_directory'])
        bundle = Path(p['inputs']['manifest']['path']).parent if a.phase == 'prepare' else prepared/'checkpoint'
        from fastapi.testclient import TestClient
        app, server = create_app(Path(p['designer']), Path(p['scenesmith'])); client = TestClient(app)
        requests = []
        def post(route, payload=None):
            response = client.post(route, json=payload or {})
            body = response.json()
            requests.append(dict(at=now(), route=route, payload=payload, status_code=response.status_code, response=body))
            write(a.out/'HTTP_REQUESTS.json', requests)
            if response.status_code != 200:
                raise RuntimeError(f'{route}: {response.status_code}: {str(body)[:500]}')
            return body
        post('/session/existing', dict(bundle_directory=str(bundle), output_directory=str(a.out/'house')))
        session = server._state.session
        post('/session/stage', dict(stage='furniture', stub_assets=False, skip_render=False))
        initial = session.scene.to_state_dict()
        def measure_current():
            return storage_measures(session.scene.to_state_dict(), session.scene.scene_dir, session.project.directory, p)
        def capture(label):
            state = session.scene.to_state_dict(); obs = post('/session/observation')
            if len(obs['image_paths']) != 5 or canonical_hash(read(obs['state_binding'])['native_state']) != canonical_hash(state):
                raise ValueError('Five views are not bound to actual state')
            m = measure_current()
            write(a.out/(label+'_state.json'), state); write(a.out/(label+'_observation.json'), obs)
            write(a.out/(label+'_measurement.json'), m)
            return obs, m
        if a.phase == 'prepare':
            obs, initial_m = capture('initial')
            result['weight_verification'] = []
            for item in p['model_weight_files']:
                path = Path(p['model_directory'])/item['path']
                if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                    raise ValueError('Model weights changed')
                result['weight_verification'].append(dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=item['sha256']))
            result['checkpoint'] = post('/session/checkpoint', dict(output_directory=str(a.out/'checkpoint')))
            ref_state = read(p['inputs']['reference_state']['path'])
            ref_measure = read(p['inputs']['reference_measurement']['path'])
            # Check an existing reference, never execute or retune it.
            result['reference_gate'] = score(initial, ref_state, initial_m, ref_measure, p)
            if not result['reference_gate']['conditional_geometric_rule_passed']:
                raise ValueError('Existing reference does not satisfy frozen rule')
        else:
            prepare = read(prepared/'RESULT.json')
            if prepare['status'] != 'prepared_without_model':
                raise ValueError('Preparation not passed')
            for item in prepare['weight_verification']:
                st = Path(item['path']).stat()
                if [st.st_size, st.st_mtime_ns] != [item['size'], item['mtime_ns']]:
                    raise ValueError('Weights changed since hash verification')
            original = read(prepared/'initial_state.json')
            result['restore_differences'] = rotation_roundtrip_differences(original, initial)
            obs = read(prepared/'initial_observation.json'); initial_m = read(prepared/'initial_measurement.json')
            # The native loader can round rotations; all other fields must match.
            # This function raises on non-roundoff changes and returns a list.
        initial_m = copy.deepcopy(initial_m); m = initial_m
        tools = [x for x in schemas_from_observation(obs) if x['name'] == 'move_furniture_tool']
        if len(tools) != 1:
            raise ValueError('Missing native movement schema')
        knowledge = read(p['inputs']['knowledge']['path'])['records']
        def request_now():
            state = session.scene.to_state_dict()
            if not result['rounds'] and a.phase == 'rollout':
                # Reuse the exact common representation only after explicit
                # quaternion-roundoff equivalence validation above.
                state = read(prepared/'initial_state.json')
            objects = {oid: {k: obj[k] for k in ('object_id','name','object_type','bbox_min','bbox_max','transform','immutable')}
                       for oid, obj in state['objects'].items()}
            context = dict(design_request=session.project.task['design_request'], original_request=None,
                request_provenance='adapted', objects=objects, available_source_knowledge=knowledge,
                available_measurement=compact_measurement(m), geometry=inspect(state), unknowns=session.project.task['unknowns'])
            history = [exchange(x) for x in result['rounds'][-2:]]
            if len(json.dumps(history, ensure_ascii=False)) > 60000:
                raise ValueError('Complete two-turn history exceeds frozen limit')
            support = dict(remaining_calls=p['budget']['model_calls']-result['model_calls'],
                prior_proposals_and_actual_outcomes=history, omitted_older_turns=max(0, len(result['rounds'])-2),
                scope='Only move the three named storage instances; every other object must remain unchanged. No additions, deletion or scaling. No target layout is provided.',
                movable_storage_ids=p['storage_ids'])
            return build_request(context=context, tools=tools, scope=p['task_scope'], images=obs['image_paths'],
                                 additional_text=json.dumps(support, ensure_ascii=False))
        request = a.out/'initial_request.json'; write(request, request_now())
        result['initial_request_sha256'] = sha256(request)
        if a.phase == 'prepare':
            result['encoding'] = model_call(request, a.out/'encoding', 'encode', 0)['encoding']
            result['status'] = 'prepared_without_model'
        else:
            if sha256(request) != read(prepared/'RESULT.json')['initial_request_sha256']:
                raise ValueError('First actual model request differs from common preparation')
            result['stop_reason'] = 'call_budget_exhausted'
            for turn in range(p['budget']['model_calls']):
                rd = a.out/f'round_{turn:03d}'; req = rd/'request.json'; write(req, request_now())
                record = model_call(req, rd, 'generate', turn)
                if record is None: break
                raw = (rd/'model/raw_output.txt').read_text()
                row = dict(turn=turn, raw_output=raw, model=record)
                result['rounds'].append(row)
                blocks = re.findall(r'<tool_call>(.*?)</tool_call>', raw, re.S)
                if not blocks and '<done>' in raw and not record.get('output_rejected_reason'):
                    row['stop'] = True; result['stop_reason'] = 'model_done'; progress(); break
                result['tool_attempts'] += 1
                try:
                    if len(blocks) != 1 or raw.count('<tool_call>') != 1 or record.get('output_rejected_reason'):
                        raise ValueError('Expected one complete native tool call')
                    call = strict_loads(blocks[0]); validate_action(call, read(req))
                except Exception as exc:
                    # No output repair. Schema/identity failures consume a call.
                    from jsonschema.exceptions import ValidationError
                    if not isinstance(exc, (ValueError, ValidationError)):
                        raise
                    row['rejection'] = (type(exc).__name__ + ': ' + str(getattr(exc, 'message', exc)))[:512]
                    write(rd/'REJECTION.json', dict(error=str(exc))); progress(); continue
                row['before_state_sha256'] = canonical_hash(session.scene.to_state_dict())
                row['call'] = call
                row['reply'] = post('/session/step', dict(tool_calls=[call])); result['native_calls'] += 1
                row['after_state_sha256'] = canonical_hash(session.scene.to_state_dict())
                row['checkpoint'] = post('/session/checkpoint', dict(output_directory=str(rd/'checkpoint')))
                obs, m = capture(f'round_{turn:03d}')
                row['score'] = score(initial, session.scene.to_state_dict(), initial_m, m, p)
                progress()
            result['status'] = 'completed_pending_review'
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc)); raise
    finally:
        if session is not None and a.phase == 'rollout':
            try:
                result['final_checkpoint'] = post('/session/checkpoint', dict(output_directory=str(a.out/'final_checkpoint')))
                final = session.scene.to_state_dict(); write(a.out/'final_state.json', final)
                m = measure_current(); write(a.out/'final_measurement.json', m)
                write(a.out/'final_observation.json', obs)
                result['final_score'] = score(initial, final, initial_m, m, p)
            except BaseException as exc:
                result.update(status='failed', finalization_error=str(exc))
        if server is not None: server._close_all()
        if client is not None: client.close()
        for lock in locks: lock.close()
        result['ended_at'] = now(); progress(); print(now(), result['status'], flush=True)
    if result['status'] == 'failed': raise SystemExit(1)


if __name__ == '__main__':
    main()
