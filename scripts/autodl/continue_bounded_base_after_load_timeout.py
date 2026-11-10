"""Explicit one-time continuation of a pre-generation cold-load timeout.

The failed attempt remains in the four-attempt / 1200-second budget. Reading
locked files first warms the OS cache; it does not create another weight copy,
load tensors or open a CUDA context. No retries inside this continuation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from bounded_trajectory import make_request
from cpu_continuation import dump, read, sha, execute, resource_snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    w, parent, out = a.work_base.resolve(), a.parent.resolve(), a.output.resolve()
    prior = read(parent / 'result.json')
    cfg = read(parent / 'config.json')
    assert prior['status'] == 'infrastructure_or_execution_failure_no_retry'
    assert prior['model_subprocess_attempts'] == 1 and prior['model_calls'] == 0
    assert prior['tool_attempts'] == 0
    assert read(parent / 'turn_00/model_process_receipt.json')['timeout'] is True
    assert not list(parent.glob('turn_*/model/raw_output.txt'))
    assert cfg['max_attempts'] == 4 and cfg['total_model_subprocess_seconds'] == 1200
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    code = Path(__file__).resolve().parent
    repo = code.parent.parent
    old = w / 'paris_local_run_results_20261104'
    stage = w / 'autodl_continuation_20261110/cpu_replay/keep'
    cp = stage / 'tools/checkpoint.json'
    assert str(cp) == prior['terminal_checkpoint'] and sha(cp) == cfg['source_sha256']
    audit = read(stage / 'audit.json')
    measured = read(stage / 'planar_measurements.json')
    joint = read(stage / 'complete_finite_pool_joint.json')
    views = stage / 'views_receipt.json'
    assert make_request(w, cp, audit, measured, joint, views, []) == read(parent / 'turn_00/request.json')
    source_hashes = read(w / 'autodl_continuation_20261110/cpu_replay/source_hashes_before.json')
    assert {p: sha(p) for p in source_hashes} == source_hashes
    dump(out / 'configuration.json', dict(cfg, parent=str(parent),
        prior_model_attempts=1, prior_model_seconds=prior['model_subprocess_seconds'],
        max_new_attempts=3, source_execution_sha256=sha(Path(__file__)),
        scope='Explicit bounded recovery after cold load; no hidden retry or budget reset'))
    outcome = {'status': 'started', 'completed_model_outputs': 0, 'tool_attempts': 0,
        'paid_api_calls': 0, 'training_updates': 0, 'new_complete_scene_generations': 0,
        'whole_scene_success': None}
    history = []
    try:
        snap = resource_snapshot(out)
        dump(out / 'resources_before.json', snap)
        if not snap['nvidia_device_nodes'] or int(snap['memory_limit_bytes']) < 24*1024**3:
            raise RuntimeError('Required existing resources are no longer visible')
        if snap['disk_free_bytes'] < snap['reserve_bytes'] + 100*1024**2:
            raise RuntimeError('Insufficient output margin; preserve the 5 GiB reserve')

        lock = read(repo / 'paris/config/weights.lock.json')
        prewarm = {'purpose': 'Real sequential file-cache warmup; no tensors/GPU or duplicate weight files',
            'files': [], 'count_as_instance_wall_time': True, 'model_calls': 0}
        tick = time.monotonic()
        for row in lock['files']:
            p = w / 'critic_public_runtime_20261104/checkpoint' / row['path']
            t = time.monotonic()
            h = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(1024**2), b''):
                    if time.monotonic()-tick > 300:
                        raise TimeoutError('300-second cache warmup cap; no inference started')
                    h.update(chunk)
            actual = h.hexdigest()
            assert actual == row['sha256'] and p.stat().st_size == row['bytes']
            prewarm['files'].append({'path': str(p), 'bytes': row['bytes'], 'sha256': actual,
                                    'seconds': time.monotonic()-t})
            dump(out / 'cache_warmup.json', prewarm)
            print(json.dumps({'event': 'cache_file_read', 'file': p.name,
                              'seconds': prewarm['files'][-1]['seconds']}), flush=True)
        prewarm['seconds'] = time.monotonic()-tick
        prewarm['resources_after'] = resource_snapshot(out)
        dump(out / 'cache_warmup.json', prewarm)

        model_seconds = prior['model_subprocess_seconds']
        for step in range(3):
            remaining = 1200-model_seconds
            if remaining <= 0:
                outcome['status'] = 'inherited_time_budget_exhausted'
                break
            snap = resource_snapshot(out)
            if snap['disk_free_bytes'] < snap['reserve_bytes'] + 48*1024**2:
                raise RuntimeError('Insufficient margin for another terminal; no deletion or retry')
            turn = out / f'turn_{step+1:02d}'
            turn.mkdir()
            dump(turn / 'request.json', make_request(w, cp, audit, measured, joint, views, history))
            cost = execute([str(w/'v100_runtime_20261104/runtime/bin/python'),
                str(code/'model_step.py'), '--phase', 'generate', '--work-base', str(w),
                '--request', str(turn/'request.json'), '--output', str(turn/'model')],
                turn, 'model_process', min(200, remaining))
            model_seconds += cost['elapsed_seconds']
            outcome['completed_model_outputs'] += 1
            calls = read(turn/'model/parsed_calls.json')
            raw = (turn/'model/raw_output.txt').read_text()
            if not calls:
                outcome['status'] = 'upstream_no_tool_termination'
                break
            if len(calls) != 1 or calls[0].get('name') != 'move_furniture_tool':
                feedback = {'success': False, 'error': 'Exactly one advertised native move allowed'}
                dump(turn/'rejection.json', feedback)
                history.append({'raw_model_output': raw, 'feedback': feedback})
                outcome['status'] = 'new_attempt_budget_reached'
                continue
            outcome['tool_attempts'] += 1
            dump(turn/'calls.json', calls)
            execute([sys.executable, str(old/'dispatch_corrected.py'), '--calls', str(turn/'calls.json'),
                '--resume', str(cp), '--output', str(turn/'tools')], turn, 'dispatch')
            cp = turn/'tools/checkpoint.json'
            history.append({'raw_model_output': raw, 'feedback': read(turn/'tools/tool_responses.json')})
            execute([sys.executable, str(old/'audit_scene.py'), '--checkpoint', str(cp),
                '--out', str(turn/'audit.json')], turn, 'mesh_audit')
            audit = read(turn/'audit.json')
            sys.path.insert(0, str(w/'joint_use_feedback_20261104'))
            from evaluate import strong_measure, changes, payload
            from measure_joint_use import measure
            measured = measure(cp)
            joint = strong_measure(measured)
            dump(turn/'measurements.json', measured)
            dump(turn/'joint.json', joint)
            diff = changes(payload(Path(cfg['source_checkpoint'])), payload(cp))
            dump(turn/'state_changes.json', diff)
            if diff['added'] or diff['removed']:
                raise RuntimeError('Move-only object preservation violated')
            execute(['/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python',
                str(old/'render_state.py'), '--scene', str(cp.parent/'partial_scene.glb'),
                '--out', str(turn/'views'), '--transient-blend'], turn, 'render')
            views = turn/'views/receipt.json'
            outcome['status'] = 'new_attempt_budget_reached'
    except BaseException as exc:
        outcome.update(status='infrastructure_or_execution_failure_no_retry', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        costs = [read(p) for p in out.glob('turn_*/model_process_receipt.json')]
        outcome.update(new_model_subprocess_attempts=len(costs),
            cumulative_model_subprocess_attempts=1+len(costs),
            new_model_subprocess_seconds=sum(p['elapsed_seconds'] for p in costs),
            cumulative_model_subprocess_seconds=prior['model_subprocess_seconds']+sum(p['elapsed_seconds'] for p in costs),
            terminal_checkpoint=str(cp), condition_A_separate_planar=measured['results']['scene_section_free']['separate_use']['status'],
            condition_B_simultaneous_planar=joint['status'],
            remaining_mesh_pairs=audit['furniture_surface_intersection_pairs'], unknowns=audit['not_checked'],
            elapsed_seconds=time.monotonic()-started,
            original_sources_unchanged={p: sha(p) for p in source_hashes} == source_hashes)
        dump(out/'result.json', outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
