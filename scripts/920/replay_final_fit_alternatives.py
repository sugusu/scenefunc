"""Replay frozen, previously generated legal edits once from their own input states.

No model, training, coordinate repair or continuation. Duplicate state/action
pairs share one explicitly recorded execution, not independent repetitions.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from native_checkpoint import canonical_hash, rotation_roundtrip_differences
from native_project import sha256, validate_bundle
from conditional_room_metrics import measure_endpoint
from score_activity_transition import score_transition


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def check_inputs(plan):
    for path, digest in plan['input_hashes'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Frozen input changed: '+path)
    for name, digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise ValueError('Frozen code changed: '+name)
    raw = json.loads(Path(plan['raw_result']).read_text())['generations']
    for case in plan['cases']:
        manifest, house, _, _ = validate_bundle(Path(case['bundle']))
        state = house['rooms'][manifest['selected_room_id']]
        source = json.loads(Path(case['input_source']).read_text())
        records = source['context']['objects']
        if set(x['object_id'] for x in records) != set(state['objects']):
            raise ValueError('Input inventory mismatch')
        for obj in records:
            expected = {k: state['objects'][obj['object_id']][k] for k in obj}
            rotation_roundtrip_differences(obj, expected)
        for arm in case['arms']:
            row = next(g for g in raw if g['id'] == case['state_id'] and g['arm'] == arm)
            if row['parsed'] != {'kind': 'tool_call', 'value': case['command']}:
                raise ValueError('Proposal altered')
            if sha256(Path(row['raw_output'])) != row['raw_output_sha256']:
                raise ValueError('Raw generation changed')
        command = case['command']
        if command['name'] != 'move_furniture_tool' or command['arguments']['object_id'] not in plan['program']['movable_ids']:
            raise ValueError('Illegal edit included')
    if len(plan['cases']) != 6 or sum(len(c['arms']) for c in plan['cases']) != 7:
        raise ValueError('Frozen coverage mismatch')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args(); plan = json.loads(a.plan.read_text())
    check_inputs(plan)
    if a.validate_only:
        print('PASS: six unique proposals / seven arm rows; source states, raw actions, hashes and allowed objects checked')
        return
    for key in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH'):
        os.environ.pop(key, None)
    os.environ.update(OPENAI_API_KEY='disabled-local-validation', OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        NO_PROXY='127.0.0.1,localhost', SCENE_RENDER_THREADS='4', SCENE_RENDER_SAMPLES='8',
        OPENSCENEAGENT_STUB_ASSETS='0', PYTHONDONTWRITEBYTECODE='1')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    from existing_project_session import create_app
    from fastapi.testclient import TestClient
    from PIL import Image
    now = lambda: datetime.now(timezone.utc).isoformat()
    out = Path(plan['runtime_directory']); out.mkdir(parents=True, exist_ok=False)
    pub = Path(plan['public_directory'])
    result = dict(status='running', started_at=now(), model_calls=0, training_updates=0,
                  native_calls=0, completed=0, planned=6, rows=[], plan_sha256=sha256(a.plan))
    if plan.get('retained_result'):
        retained = json.loads(Path(plan['retained_result']).read_text())
        if retained['completed'] != 1 or retained['native_calls'] != 1 or len(retained['rows']) != 1:
            raise ValueError('Unexpected retained execution accounting')
        row = retained['rows'][0]
        if any(row[k] != v for k, v in plan['cases'][0].items()):
            raise ValueError('Retained case differs from planned case')
        result.update(rows=retained['rows'], completed=1, native_calls=1,
                      retained_native_calls=1, retained_result=plan['retained_result'])
    def progress():
        write(out/'RESULT.json', result)
        write(pub/'STATUS.json', dict(result, run_id=plan['run_id'], question=plan['question'],
            updated_at=now(), configuration_hashes={'plan': sha256(a.plan)},
            latest_verifiable_artifact=str(out/'RESULT.json'), failure_reason=result.get('error'),
            next_step='Review actual endpoint consequences; no automatic model continuation or training.'))
    print(now(), 'begin frozen existing-output native replay; zero model calls', flush=True)
    progress(); start = time.monotonic()
    im, ih, _, _ = validate_bundle(Path(plan['identity_reference_bundle']))
    reference = ih['rooms'][im['selected_room_id']]
    client = server = None
    try:
        app, server = create_app(Path(plan['designer']), Path(plan['scenesmith']))
        client = TestClient(app)
        for case in plan['cases'][result['completed']:]:
            cd = out/case['id']; cd.mkdir()
            requests = []
            try:
                def post(route, payload=None):
                    t = time.monotonic(); reply = client.post(route, json=payload or {})
                    requests.append(dict(route=route, payload=payload, status_code=reply.status_code,
                        elapsed_seconds=time.monotonic()-t, response=reply.json()))
                    write(cd/'HTTP_REQUESTS.json', requests)
                    if reply.status_code != 200: raise RuntimeError('Native HTTP error: '+route)
                    return reply.json()
                post('/session/existing', dict(bundle_directory=case['bundle'], output_directory=str(cd/'house')))
                post('/session/stage', dict(stage='furniture', stub_assets=False, skip_render=False))
                session = server._state.session; scene = session.scene
                before = scene.to_state_dict()
                manifest, house, _, _ = validate_bundle(Path(case['bundle']))
                drift = rotation_roundtrip_differences(house['rooms'][manifest['selected_room_id']], before)
                write(cd/'before_state.json', before)
                before_eval = measure_endpoint(before, reference, scene.scene_dir, session.project.directory, plan['program'], case['mode'])
                write(cd/'before_evaluation.json', before_eval)
                result['native_calls'] += 1; progress()
                reply = post('/session/step', {'tool_calls': [case['command']]})
                after = scene.to_state_dict(); write(cd/'after_state.json', after)
                after_eval = measure_endpoint(after, reference, scene.scene_dir, session.project.directory, plan['program'], case['mode'])
                write(cd/'after_evaluation.json', after_eval)
                changed = canonical_hash(before) != canonical_hash(after)
                score = score_transition(before_eval['review'], after_eval['review'], attempts=1, native_calls=1, state_changes=int(changed))
                write(cd/'TRANSITION.json', score)
                checkpoint = post('/session/checkpoint', {'output_directory': str(cd/'checkpoint')})
                obs = post('/session/observation')
                binding = json.loads(Path(obs['state_binding']).read_text())
                if len(obs['image_paths']) != 5 or canonical_hash(binding['native_state']) != canonical_hash(after):
                    raise ValueError('Final observation not bound to actual state')
                for path in obs['image_paths']:
                    with Image.open(path) as img: img.verify()
                write(cd/'observation.json', obs)
                row = dict(case, transition=score, native_reply=reply, checkpoint=checkpoint,
                    checkpoint_manifest_sha256=sha256(cd/'checkpoint/manifest.json'),
                    before_sha256=canonical_hash(before), after_sha256=canonical_hash(after),
                    restoration_roundoff=drift, image_paths=obs['image_paths'])
                result['rows'].append(row); result['completed'] += 1; progress()
                print(now(), case['id'], score['outcome'], score['final_failed'], flush=True)
            finally:
                if server is not None: server._close_all()
        result.update(status='completed', exit_code=0)
    except BaseException as exc:
        result.update(status='failed', exit_code=1, error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if client is not None: client.close()
        import psutil
        children = []
        for proc in psutil.Process().children(recursive=True):
            try:
                if os.getpgid(proc.pid) == os.getpgrp(): children.append(proc); proc.terminate()
            except (ProcessLookupError, psutil.Error): pass
        _, alive = psutil.wait_procs(children, timeout=5)
        for proc in alive:
            try: proc.kill()
            except psutil.Error: pass
        result.update(ended_at=now(), elapsed_seconds=time.monotonic()-start); progress()


if __name__ == '__main__':
    main()
