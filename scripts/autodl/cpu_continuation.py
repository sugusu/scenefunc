"""Replay two fixed reference candidates; never describe them as model outputs.

Uses the existing, read-only AutoDL native bridge and mesh measurements. No
weights are loaded, no new search is performed, and no threshold is adjusted.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def execute(command, folder, label, seconds=300):
    start = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (label + '.stdout')).open('w') as out, (folder / (label + '.stderr')).open('w') as err:
        try:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=seconds, check=False)
            record = {'exit_code': result.returncode, 'timeout': False}
        except subprocess.TimeoutExpired:
            record = {'exit_code': None, 'timeout': True}
    record.update(command=command, elapsed_seconds=time.monotonic() - start, timeout_seconds=seconds)
    dump(folder / (label + '_receipt.json'), record)
    if record['exit_code'] != 0:
        raise RuntimeError(f'{label} failed; retained logs; no automatic retry')
    return record


def resource_snapshot(path):
    def scalar(p):
        q = Path(p)
        return q.read_text().strip() if q.exists() else None
    return {
        'memory_limit_bytes': scalar('/sys/fs/cgroup/memory/memory.limit_in_bytes'),
        'memory_usage_bytes': scalar('/sys/fs/cgroup/memory/memory.usage_in_bytes'),
        'cpu_quota_us': scalar('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
        'cpu_period_us': scalar('/sys/fs/cgroup/cpu/cpu.cfs_period_us'),
        'nvidia_device_nodes': [str(p) for p in Path('/dev').glob('nvidia*')],
        'disk_free_bytes': shutil.disk_usage(path).free,
        'reserve_bytes': 5 * 1024**3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--finish-existing', action='store_true', help='Evaluate saved native terminals only; never rerun a tool')
    args = ap.parse_args()
    w = args.work_base.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.finish_existing)
    start = time.monotonic()
    old = w / 'paris_local_run_results_20261104'
    prior = w / 'joint_use_feedback_20261104/ordinary_control/B'
    witness = w / 'joint_use_witness_20261104'
    source = old / 'B_mesh_feedback_sdpa_kv/tools/checkpoint.json'
    sys.path.insert(0, str(w / 'joint_use_feedback_20261104'))
    from evaluate import payload, changes, strong_measure
    from measure_joint_use import measure

    snapshot = resource_snapshot(out)
    dump(out / ('resources_at_evaluation_resume.json' if args.finish_existing else 'resources.json'), snapshot)
    if snapshot['disk_free_bytes'] < snapshot['reserve_bytes'] + 100 * 1024**2:
        raise RuntimeError('Insufficient margin for two complete GLBs above the 5 GiB reserve')
    baseline = payload(source)
    sources = [source, source.parent / 'partial_scene.glb',
               old / 'dispatch_corrected.py', old / 'audit_scene.py',
               witness / 'measure_joint_use.py', witness / 'strong_pair_control.py',
               witness / 'PROTOCOL.json', prior / 'parsed_tool_calls.json',
               Path(baseline['frozen_shell']['path'])]
    sources += [Path(p) for p in baseline['geometry_hashes']]
    sources += [w / f'paris_A_inference_20261104/input_{c}.json' for c in ['A', 'B']]
    preserved = {str(p): sha(p) for p in sources}
    if args.finish_existing:
        assert read(out / 'source_hashes_before.json') == preserved
    else:
        dump(out / 'source_hashes_before.json', preserved)
    cases = [('keep', [], old / 'B_after_feedback_views/receipt.json'),
             ('ordinary_reference', read(prior / 'parsed_tool_calls.json'), prior / 'views/receipt.json')]
    protocol = {
        'scene': 'Existing furnished Paris development state, not a new generated room',
        'choice': 'Waterloo native checkpoint/SDF package unavailable locally; Paris fallback',
        'conditions': {c: read(w / f'paris_A_inference_20261104/input_{c}.json') for c in ['A', 'B']},
        'candidates': [{'name': n, 'calls': calls} for n, calls, _ in cases],
        'candidate_origin': 'Previously selected ordinary candidate; post-hoc development reference, not prospective method comparison',
        'labels': 'Researcher-constructed use variants; no independent human preference labels',
        'full_scene_success': None,
        'paid_api_calls': 0, 'model_calls': 0, 'training': False,
        'finite_planar_checks': 'Inherited 0.01 m centers, 0/90 degree rectangles; exact pair check over that pool; no continuous infeasibility claim',
    }
    if args.finish_existing:
        assert read(out / 'protocol.json') == protocol
    else:
        dump(out / 'protocol.json', protocol)
    rows = []
    base_audit = None
    for name, calls, view_path in cases:
        stage = out / name
        if args.finish_existing:
            assert read(stage / 'calls.json') == calls
            assert read(stage / 'tools/receipt.json')['resumed_from'] == str(source)
            if (stage / 'result.json').exists():
                rows.append(read(stage / 'result.json'))
                base_audit = read(stage / 'audit.json')
                continue
            # The dispatcher saved its own completed receipt before the outer
            # process exited. Do not replay or invent missing wrapper timings.
            dispatch = read(stage / 'tools/receipt.json')
            assert dispatch['status'] == 'bounded_dispatch_completed'
            audit_cost = {'elapsed_seconds': read(stage / 'audit.json')['elapsed_seconds']}
        else:
            stage.mkdir()
            dump(stage / 'calls.json', calls)
            print(json.dumps({'stage': name, 'event': 'native_replay_started'}), flush=True)
            dispatch = execute([sys.executable, str(old / 'dispatch_corrected.py'), '--calls', str(stage / 'calls.json'),
                                '--resume', str(source), '--output', str(stage / 'tools')], stage, 'dispatch')
        cp = stage / 'tools/checkpoint.json'
        if not args.finish_existing:
            audit_cost = execute([sys.executable, str(old / 'audit_scene.py'), '--checkpoint', str(cp),
                                  '--out', str(stage / 'audit.json')], stage, 'mesh_audit')
        audit = read(stage / 'audit.json')
        if base_audit is None:
            base_audit = audit
        tick = time.monotonic()
        measured = measure(cp)
        joint = strong_measure(measured)
        diff = changes(baseline, payload(cp))
        dump(stage / 'planar_measurements.json', measured)
        dump(stage / 'complete_finite_pool_joint.json', joint)
        dump(stage / 'state_changes.json', diff)
        view_receipt = read(view_path)
        views = {'source_receipt': str(view_path), 'fresh_render': False, 'scene_hash_matches': view_receipt['source_sha256'] == audit['scene_sha256'], 'views': []}
        if views['scene_hash_matches']:
            for view in view_receipt['views']:
                assert sha(view['path']) == view['sha256']
                dest = stage / 'views' / Path(view['path']).name
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(view['path'], dest)
                views['views'].append({'path': str(dest), 'sha256': sha(dest)})
        dump(stage / 'views_receipt.json', views)
        before_pairs = {tuple(sorted(p)) for p in base_audit['furniture_surface_intersection_pairs']}
        after_pairs = {tuple(sorted(p)) for p in audit['furniture_surface_intersection_pairs']}
        row = {
            'candidate': name,
            'scene_sha256': audit['scene_sha256'],
            'condition_A_separate_planar_status': measured['results']['scene_section_free']['separate_use']['status'],
            'condition_B_simultaneous_planar_status': joint['status'],
            'furniture_count': audit['placed_furniture'],
            'added': len(diff['added']), 'removed': len(diff['removed']), 'changed': len(diff['changed']),
            'architecture_preserved': diff['shell_unchanged'],
            'intersection_pairs': len(after_pairs), 'new_intersection_pairs': sorted(after_pairs - before_pairs),
            'eliminated_intersection_pairs': sorted(before_pairs - after_pairs),
            'outside_floor_objects': audit['out_of_rectangular_floor_objects'],
            'wall_door_window_intersections': audit['architecture_surface_intersection_pairs']['walls_doors_windows'],
            'floor_reference': audit['floor_reference'],
            'unknowns': ['human access and posture', 'storage suitable for putting work away in A', 'sleep usability', 'window view', 'materials and aesthetics', 'continuous feasibility if no finite witness'],
            'full_scene_success': None,
            'dispatch_seconds': dispatch['elapsed_seconds'], 'mesh_audit_seconds': audit_cost['elapsed_seconds'],
            'measure_and_bind_seconds': time.monotonic() - tick,
            'timing_scope': 'Internal native dispatch / audit timers; original wrapper interrupted' if args.finish_existing else 'Subprocess wall timers',
            'model_calls': 0, 'paid_api_calls': 0,
        }
        rows.append(row)
        dump(stage / 'result.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    after = {p: sha(p) for p in preserved}
    assert after == preserved
    dump(out / 'source_preservation.json', {'all_unchanged': True, 'file_count': len(after), 'hashes': after})
    dump(out / 'results.json', {'rows': rows, 'this_evaluation_process_seconds': time.monotonic() - start, 'resumed_evaluation_only': args.finish_existing, 'new_generated_rooms': 0,
                              'fresh_native_replays': len(rows), 'model_calls': 0, 'paid_api_calls': 0,
                              'conclusion': 'Ordinary placement and known geometry explain this reference result; no learned consequence predictor or Agent advantage tested'})
    fields = ['candidate', 'furniture_count', 'changed', 'intersection_pairs', 'condition_A_separate_planar_status',
              'condition_B_simultaneous_planar_status', 'dispatch_seconds', 'mesh_audit_seconds', 'measure_and_bind_seconds', 'full_scene_success']
    with (out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
