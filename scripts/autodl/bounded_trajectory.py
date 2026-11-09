"""One prospective base-model trajectory, with at most four model/tool attempts.

This is a furnished-scene editing diagnostic, not a learned consequence model
or full Scene-Designer reproduction. CPU preparation never loads model weights.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from cpu_continuation import dump, read, sha, execute, resource_snapshot


def make_request(w, checkpoint, audit, measured, joint, view_receipt, history):
    state = read(checkpoint)['payload']
    scene_sha = sha(checkpoint.parent / 'partial_scene.glb')
    assert audit['scene_sha256'] == scene_sha
    views = read(view_receipt)
    assert views.get('scene_hash_matches', views.get('source_sha256') == scene_sha)
    for img in views['views']:
        assert sha(img['path']) == img['sha256']
    system = '''You are editing a furnished room to meet the original brief.
Use the supplied current five views, object IDs and measured evidence. Preserve
all existing objects, asset geometry and architecture. Only floor furniture
translation/rotation through the following native tool is available in this
bounded diagnostic. Do not claim unavailable operations or unknown uses succeed.
Return at most ONE <tool_call>{"name":"move_furniture_tool","arguments":
{"object_id":"existing ID","x":0.0,"y":0.0,"yaw":0.0}}</tool_call> per turn.
Coordinates are world XY metres, Z is up, yaw is in DEGREES. Tool x/y follow the
native object transform, not necessarily world bounding-box centre. You may
include <think>reasoning</think>. If no available edit is appropriate, explain
remaining limitations without a tool call. No tool call ends the trajectory;
ending is not a claim of task success. Maximum four attempts. A rectangle-fit
witness verifies planar space only, not access, comfort, storage or preference.
'''
    observation = {
        'current_objects': [{k: o[k] for k in ['object_id', 'name', 'transform'] if k in o} for o in state['placed_instances']],
        'mesh_check': {k: audit[k] for k in ['objects', 'furniture_surface_intersection_pairs', 'architecture_surface_intersection_pairs', 'out_of_rectangular_floor_objects', 'floor_reference', 'not_checked']},
        'separate_planar_status': measured['results']['scene_section_free']['separate_use']['status'],
        'simultaneous_planar_status': joint['status'],
        'finite_search_scope': '0.01m centres and 0/90 degree rectangles; not_found is not proof of continuous infeasibility',
        'surfaces': [{k: s[k] for k in ['object_id', 'surface_m2', 'free_section_m2', 'blockers', 'uncertain_section_objects']} for s in measured['surfaces']],
        'previous_attempts': history,
    }
    content = [{'type': 'image', 'image': v['path']} for v in views['views']]
    content.append({'type': 'text', 'text': read(w / 'paris_A_inference_20261104/input_B.json')['brief'] + '\nCurrent evidence:\n' + json.dumps(observation, ensure_ascii=False, separators=(',', ':'))})
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--cpu-replay', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--phase', choices=['prepare', 'run'], required=True)
    args = ap.parse_args()
    w, root = args.work_base.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parent
    old = w / 'paris_local_run_results_20261104'
    stage = args.cpu_replay.resolve() / 'keep'
    cp = stage / 'tools/checkpoint.json'
    audit = read(stage / 'audit.json')
    measured = read(stage / 'planar_measurements.json')
    joint = read(stage / 'complete_finite_pool_joint.json')
    views = stage / 'views_receipt.json'
    model_python = str(w / 'v100_runtime_20261104/runtime/bin/python')
    renderer_python = '/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python'
    config = {'max_attempts': 4, 'max_actions_per_attempt': 1, 'model_subprocess_seconds': 200,
              'total_model_subprocess_seconds': 1200, 'model_input_tokens': 8192, 'model_output_tokens': 1024,
              'max_image_pixels': 65536, 'dtype': 'float16', 'training': False, 'paid_api_calls': 0,
              'source_checkpoint': str(cp), 'source_sha256': sha(cp), 'same_terminal_evaluated_under': ['A', 'B'],
              'scope': 'Base-model move-only editing diagnostic; no new generation or predictor training'}
    dump(root / 'config.json', config)
    history, gpu_seconds = [], 0.0
    outcome = {'status': 'started', 'model_calls': 0, 'tool_attempts': 0, 'full_scene_success': None}
    try:
        if args.phase == 'run':
            snap = resource_snapshot(root)
            dump(root / 'resources.json', snap)
            if not snap['nvidia_device_nodes'] or int(snap['memory_limit_bytes']) < 24 * 1024**3:
                outcome['status'] = 'resources_unavailable_no_model_started'
                return
        for step in range(4):
            turn = root / f'turn_{step:02d}'
            turn.mkdir()
            request = make_request(w, cp, audit, measured, joint, views, history)
            dump(turn / 'request.json', request)
            phase = 'encode' if args.phase == 'prepare' else 'generate'
            remaining = 1200 - gpu_seconds
            if remaining <= 0:
                outcome['status'] = 'model_time_budget_exhausted'
                break
            cost = execute([model_python, str(code / 'model_step.py'), '--phase', phase, '--work-base', str(w),
                            '--request', str(turn / 'request.json'), '--output', str(turn / 'model')], turn, 'model_process', min(200, remaining))
            if args.phase == 'prepare':
                outcome['status'] = 'processor_verified_no_weights_loaded'
                break
            gpu_seconds += cost['elapsed_seconds']
            outcome['model_calls'] += 1
            calls = read(turn / 'model/parsed_calls.json')
            raw = (turn / 'model/raw_output.txt').read_text()
            if not calls:
                outcome['status'] = 'upstream_no_tool_termination'
                break
            outcome['tool_attempts'] += 1
            if len(calls) != 1 or calls[0].get('name') != 'move_furniture_tool':
                feedback = {'success': False, 'error': 'Exactly one advertised native move allowed per attempt'}
                dump(turn / 'rejection.json', feedback)
                history.append({'raw_model_output': raw, 'feedback': feedback})
                continue
            dump(turn / 'calls.json', calls)
            execute([sys.executable, str(old / 'dispatch_corrected.py'), '--calls', str(turn / 'calls.json'),
                     '--resume', str(cp), '--output', str(turn / 'tools')], turn, 'dispatch')
            cp = turn / 'tools/checkpoint.json'
            history.append({'raw_model_output': raw, 'feedback': read(turn / 'tools/tool_responses.json')})
            execute([sys.executable, str(old / 'audit_scene.py'), '--checkpoint', str(cp), '--out', str(turn / 'audit.json')], turn, 'mesh_audit')
            audit = read(turn / 'audit.json')
            sys.path.insert(0, str(w / 'joint_use_feedback_20261104'))
            from evaluate import strong_measure, changes, payload
            from measure_joint_use import measure
            measured = measure(cp)
            joint = strong_measure(measured)
            dump(turn / 'measurements.json', measured)
            dump(turn / 'joint.json', joint)
            diff = changes(payload(Path(config['source_checkpoint'])), payload(cp))
            dump(turn / 'state_changes.json', diff)
            if diff['added'] or diff['removed']:
                raise RuntimeError('Move-only preservation contract violated')
            execute([renderer_python, str(old / 'render_state.py'), '--scene', str(cp.parent / 'partial_scene.glb'),
                     '--out', str(turn / 'views'), '--transient-blend'], turn, 'render')
            views = turn / 'views/receipt.json'
            outcome['status'] = 'attempt_budget_reached'
    except Exception as e:
        outcome.update(status='infrastructure_or_execution_failure_no_retry', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        # Include failed / timed-out model subprocesses in the actual cost.
        process_receipts = [read(p) for p in root.glob('turn_*/model_process_receipt.json')]
        if args.phase == 'run':
            gpu_seconds = sum(p['elapsed_seconds'] for p in process_receipts)
        outcome['model_subprocess_attempts'] = len(process_receipts) if args.phase == 'run' else 0
        outcome.update(terminal_checkpoint=str(cp), model_subprocess_seconds=gpu_seconds,
                       condition_A_separate_planar=measured['results']['scene_section_free']['separate_use']['status'],
                       condition_B_simultaneous_planar=joint['status'], remaining_mesh_pairs=audit['furniture_surface_intersection_pairs'],
                       unknowns=audit['not_checked'], paid_api_calls=0)
        dump(root / 'result.json', outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
