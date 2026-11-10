"""One ordinary isolated-geometry preview and pinned native memory replay.

All three saved proposals are retained. This does not simulate a counterfactual
model policy after rejection and does not call or train a model.
"""
import argparse
import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from cpu_continuation import read, dump, sha, execute, resource_snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    w, out = a.work_base.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    run = repo/'runs/autodl/20261110_bounded_base_execution/recovery_v1'
    ready = repo/'runs/autodl/20261110_bounded_base_readiness'
    old = w/'paris_local_run_results_20261104'
    native_python = '/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python'
    start = time.monotonic()
    hashes = read(w/'autodl_continuation_20261110/cpu_replay/source_hashes_before.json')
    assert all(sha(p) == h for p, h in hashes.items())
    lock = read(repo/'paris/config/upstream.lock.json')
    pins = {str(w/r['path']): r for r in lock['files']}
    memory_path = w/'professional_data_ingest_20261104/native_src/memory.py'
    rollout_path = w/'native_execution_inventory_20261104/source/Designer/agent_rollout.py'
    for path in [memory_path, rollout_path]:
        assert sha(path) == pins[str(path)]['sha256']
    spec = importlib.util.spec_from_file_location('pinned_native_memory', memory_path)
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    tree = ast.parse(rollout_path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_messages')
    scope = {'Path': Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(rollout_path), 'exec'), scope)
    dump(out/'source_receipt.json', {'memory_path': str(memory_path), 'memory_sha256': sha(memory_path),
        'memory_functions_executed': ['empty_memory', 'update_furniture', 'update_room_stage', '_track_recent', 'render_memory'],
        'rollout_path': str(rollout_path), 'rollout_sha256': sha(rollout_path),
        'extracted_function': {'name': fn.name, 'start': fn.lineno, 'end': fn.end_lineno, 'body_modified': False},
        'pin': 'a0565e50ed7d32c0e09132b9311c0b9b91165405', 'full_module_or_runtime_executed': False})

    def yaw(q):
        qw, qx, qy, qz = q
        return math.degrees(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))

    def exact_noop(obj, args):
        tf = obj['transform']
        angle = math.radians(args.get('yaw', 0))/2
        target_q = [math.cos(angle), 0., 0., math.sin(angle)]
        q = tf['rotation_wxyz']
        # Strict subset of native allclose: no new quality threshold introduced.
        return tf['translation'] == [args['x'], args['y'], 0.] and (q == target_q or q == [-x for x in target_q])

    cp = Path(read(run/'configuration.json')['source_checkpoint'])
    initial_sha = sha(cp)
    payload = read(cp)['payload']
    mem = memory.empty_memory('furniture')
    for obj in payload['placed_instances']:
        tf = obj['transform']
        mem['assets'][obj['object_id']] = {'name': obj['name'], 'placed': True,
            'pos': tuple(tf['translation'][:2]), 'rot': yaw(tf['rotation_wxyz']),
            'w': None, 'd': None, 'h': None}
    rows, memory_checks = [], []
    for i, turn in enumerate(sorted(run.glob('turn_*'))):
        req = read(turn/'request.json')
        before = read(cp)['payload']
        obs = json.loads(req[-1]['content'][-1]['text'].split('\nCurrent evidence:\n', 1)[1])
        calls = read(turn/'calls.json')
        call = calls[0]
        obj = next(o for o in before['placed_instances'] if o['object_id'] == call['arguments']['object_id'])
        native_text = memory.render_memory('furniture', mem)
        target = out/turn.name
        target.mkdir()
        (target/'native_memory_before.txt').write_text(native_text+'\n')
        dump(target/'native_memory_before.json', mem)
        # Same system, task, five images and exact raw evidence. Native text is
        # derived only from already supplied state/history; intrinsic sizes unknown.
        brief, raw_obs = req[-1]['content'][-1]['text'].split('\nCurrent evidence:\n', 1)
        image_paths = [Path(x['image']) for x in req[-1]['content'] if x['type'] == 'image']
        alternate = scope['build_messages'](req[0]['content'], brief,
            native_text+'\n\nExact original evidence (unchanged):\n'+raw_obs, image_paths)
        assert alternate[0] == req[0]
        assert alternate[-1]['content'][:-1] == req[-1]['content'][:-1]
        assert alternate[-1]['content'][-1]['text'].endswith(raw_obs)
        dump(target/'native_format_request.json', alternate)
        if exact_noop(obj, call['arguments']):
            row = {'turn': turn.name, 'ordinary_decision': 'reject_exact_unchanged_pose',
                'new_native_calls': 0, 'input_checkpoint': str(cp), 'input_sha256': sha(cp)}
        else:
            assert i == 0, 'Only the single predeclared shadow is authorized'
            snap = resource_snapshot(out)
            assert snap['disk_free_bytes'] > snap['reserve_bytes'] + 48*1024**2
            dump(target/'calls.json', calls)
            tool_cost = execute([native_python, str(old/'dispatch_corrected.py'), '--resume', str(cp),
                '--calls', str(target/'calls.json'), '--output', str(target/'shadow')], target, 'dispatch')
            audit_cost = execute([native_python, str(old/'audit_scene.py'), '--checkpoint',
                str(target/'shadow/checkpoint.json'), '--out', str(target/'shadow_audit.json')], target, 'audit')
            audit = read(target/'shadow_audit.json')
            baseline = read(ready/'initial_audit.json')
            new_pairs = sorted({tuple(sorted(p)) for p in audit['furniture_surface_intersection_pairs']} -
                               {tuple(sorted(p)) for p in baseline['furniture_surface_intersection_pairs']})
            shadow = read(target/'shadow/checkpoint.json')['payload']
            actual = read(turn/'tools/checkpoint.json')['payload']
            assert shadow['placed_instances'] == actual['placed_instances']
            assert shadow['frozen_shell'] == actual['frozen_shell']
            assert audit['furniture_surface_intersection_pairs'] == read(turn/'audit.json')['furniture_surface_intersection_pairs']
            assert new_pairs
            row = {'turn': turn.name, 'ordinary_decision': 'reject_new_furniture_intersection',
                'new_native_calls': 1, 'input_checkpoint': str(cp), 'input_sha256': sha(cp),
                'new_pairs': new_pairs, 'candidate_pairs': audit['furniture_surface_intersection_pairs'],
                'shadow_checkpoint': str(target/'shadow/checkpoint.json'),
                'dispatch_seconds': tool_cost['elapsed_seconds'], 'audit_seconds': audit_cost['elapsed_seconds'],
                'original_committed_checkpoint': str(cp), 'original_unchanged': sha(cp) == initial_sha}
        rows.append(row)
        output = read(turn/'tools/tool_responses.json')[0]['response']
        if i > 0:
            assert output['error_type'] == 'no_movement'
        memory.update_furniture(mem, call['name'], call['arguments'], output)
        after = read(turn/'tools/checkpoint.json')['payload']
        for item in after['placed_instances']:
            m = mem['assets'][item['object_id']]
            assert list(m['pos']) == item['transform']['translation'][:2]
            assert abs((m['rot']-yaw(item['transform']['rotation_wxyz'])+180)%360-180) < 1e-9
        memory_checks.append({'turn': turn.name, 'poses_match_actual_checkpoint': True,
            'failed_call': not output['success'],
            'recent_failed_entries': sum(bool(x.get('failed')) for x in mem['recent_tool_calls'])})
        (target/'native_memory_after.txt').write_text(memory.render_memory('furniture', mem)+'\n')
        cp = turn/'tools/checkpoint.json'
    assert all(sha(p) == h for p, h in hashes.items())
    dump(out/'result.json', {'status': 'completed', 'ordinary_proposal_checks': rows,
        'native_memory_replay': memory_checks, 'new_model_calls': 0, 'new_training_updates': 0,
        'new_paid_api_calls': 0, 'new_native_calls': sum(r['new_native_calls'] for r in rows),
        'new_complete_scene_generations': 0, 'original_sources_unchanged': True,
        'scene_success': None, 'elapsed_seconds': time.monotonic()-start,
        'scope': 'Post-hoc per-proposal ordinary diagnosis and CPU native formatting. Not a rerun of a model policy after rejection, or evidence of learned design gain.'})
    print(json.dumps(read(out/'result.json'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
