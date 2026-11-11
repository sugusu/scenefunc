"""Run a pinned public candidate detector and real native deletion replays.

This is an offline audit of already generated development trajectories, not an
agent rollout, training run, original OTA corpus audit, or full CTC execution.
All native actions are copied from the saved model calls. No API/model imports.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

W = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
R = W / 'autodl_continuation_20261110/repository'
PILOT = R / 'runs/autodl/20261111_furnished_pair_pilot'
START = W / 'autodl_continuation_20261110/cpu_replay/keep/tools/checkpoint.json'
SOURCE = Path('/dev/shm/scenedesigner_training_review_20261111')
ARMS = ('B0', 'B1_repaired')
EXPECTED_DETECTOR = '82f3f703d8e9c19822cb86d1b39b877c53d9b90ea38916d77027b580d8eabb3a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def partial_stage_check(episode):
    """Execute unchanged public input validation, without a fake other stage."""
    source = SOURCE / 'trajectory_compression/engine.py'
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in ('ValidationError', 'validate_input')]
    assert len(selected) == 2
    model = load_module(SOURCE / 'trajectory_compression/model.py', 'ctc_stages')
    scope = {'STAGES': model.STAGES}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    try:
        scope['validate_input']({'furniture': episode})
    except scope['ValidationError'] as exc:
        return {'accepted': False, 'error': str(exc), 'body_modified': False,
                'invoked': 'validate_input only; full Compressor/reviewer not invoked'}
    raise AssertionError('An isolated furniture trace unexpectedly passed four-stage validation')


def source_history(arm):
    path = PILOT / ('model_arms_terminal.json' if arm == 'B0'
                    else 'model_arms_terminal_B1_repaired.json')
    value = read(path)[arm]
    assert len(value['history']) == 4
    return path, value['history']


def projection(arm):
    path, history = source_history(arm)
    turns, images, inputs = [], [], []
    for i, h in enumerate(history):
        turn = PILOT / arm / f'turn_{i:02d}'
        receipt = read(turn / 'model_process_receipt.json')
        assert receipt['status'] == 'terminal' and receipt['exit_code'] == 0
        feedback = h['feedback']
        assert isinstance(feedback, list) and len(feedback) == 1
        calls = [f['call'] for f in feedback]
        assert calls == read(turn / 'calls.json')
        assert all(c['name'] == 'move_furniture_tool' for c in calls)
        request = read(turn / 'request.json')
        view_paths = [c['image'] for m in request
                      if isinstance(m.get('content'), list) for c in m['content']
                      if c.get('type') == 'image']
        image_rows = [{'path': x, 'sha256': sha(x)} for x in view_paths]
        assert len(image_rows) == 5
        images.append(image_rows)
        turns.append({'role': 'assistant', 'content': h['raw_model_output'],
                      'tool_calls': calls, 'images': view_paths,
                      'original_model_turn': i,
                      'raw_model_output_sha256': hashlib.sha256(h['raw_model_output'].encode()).hexdigest()})
        turns.append({'role': 'tool_results',
                      'results': [{'name': f['call']['name'], 'output': f['response']}
                                  for f in feedback]})
        inputs.extend([turn / x for x in ('request.json', 'calls.json',
                       'checkpoint.json', 'tool_responses.json', 'audit.json',
                       'joint.json', 'model_process_receipt.json', 'views_receipt.json')])
        inputs.extend(Path(v['path']) for v in image_rows)
    return {'stage': 'furniture', 'turns': turns,
            'projection_scope': 'call/result/image projection for candidate extraction only; not SFT'}, images, [path, *inputs]


def selected_state(payload):
    # Deliberately omit tool_history/call counts; those change under deletion.
    # Compare all object fields, template/asset identity, shell, geometry hashes
    # and frame/surface metadata exactly, without an adjustable tolerance.
    keys = ('format', 'room_id', 'room_type', 'text_description',
            'surface_id_counter', 'frozen_shell', 'asset_templates',
            'placed_instances', 'geometry_hashes', 'asset_records', 'physics_ready')
    return {k: payload[k] for k in keys}


def native_owner(ramout):
    import numpy as np
    sys.path.insert(0, str(W / 'native_room_bridge_20261104'))
    sys.path.insert(0, str(W / 'paris_native_editing_20261104'))
    sys.path.insert(0, str(W / 'paris_continuation_state_20261104'))
    import native_kernels as native
    import native_editing
    import state_io
    frozen = W / 'paris_A_inference_20261104/frozen_initial.json'
    floor = next(o for o in read(frozen)['room_structure'] if o['name'] == 'floor')
    G = native.G
    geo = G['RoomGeometry'](sdf_tree=ET.ElementTree(ET.Element('sdf')),
                           sdf_path=ramout / 'not_physics_ready.sdf',
                           length=floor['dimensions'][0], width=floor['dimensions'][2],
                           wall_height=2.5, wall_thickness=.02)
    scene = G['RoomScene'](room_geometry=geo, scene_dir=ramout,
                          room_id='paris_19_remote_studio',
                          text_description='Frozen Paris development shell',
                          action_log_path=ramout / 'native_action_log.json')
    registry = G['AssetRegistry']()
    assets, history = state_io.restore(START, scene, registry, frozen, G)
    state_io.install_path_helper(G)
    tools = native_editing.make_tools(scene, registry, native)
    assert all(np.isfinite(o.transform.GetAsMatrix4()).all() for o in scene.objects.values())
    return native, scene, registry, tools, assets, history, state_io, frozen


def live_state(base, scene, registry):
    value = copy.deepcopy(base)
    value['placed_instances'] = [o.to_dict() for o in scene.objects.values()]
    value['asset_templates'] = [o.to_dict() for o in registry.list_all()]
    value['surface_id_counter'] = scene._surface_id_counter
    return value


def furniture_collisions(scene):
    import trimesh
    cm = trimesh.collision.CollisionManager()
    for oid, obj in scene.objects.items():
        mesh = trimesh.load(obj.geometry_path, force='mesh')
        mesh.apply_transform(obj.transform.GetAsMatrix4())
        cm.add_object(str(oid), mesh)
    _, pairs = cm.in_collision_internal(return_names=True)
    return [list(p) for p in sorted(pairs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ram-output', type=Path, required=True)
    args = ap.parse_args()
    protocol = read(args.output / 'PROTOCOL.json')
    assert protocol['max_native_actions'] == 24 and protocol['new_model_calls'] == 0
    assert not (args.output / 'replay_results.json').exists()
    args.ram_output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    assert sha(SOURCE / 'trajectory_compression/candidates.py') == EXPECTED_DETECTOR
    detector = load_module(SOURCE / 'trajectory_compression/candidates.py', 'official_ctc_candidates')
    initial = read(START)['payload']
    base = selected_state(initial)
    snapshots, rows, candidate_rows, checks = {}, [], [], []
    inputs = {START}
    total_actions = 0
    for arm in ARMS:
        episode, views, source_inputs = projection(arm)
        inputs.update(source_inputs)
        segments = detector.candidates('furniture', episode)
        dump(args.output / f'{arm}_projected_episode.json', episode)
        dump(args.output / f'{arm}_official_candidates.json', segments)
        # This is what the detector actually returns, not a reviewer decision.
        allowed = sorted(set().union(*(detector.deletable_turns(s) for s in segments)))
        original_allowed = [i // 2 for i in allowed]
        candidate_rows.append({'arm': arm, 'segments': len(segments),
                               'allowed_projected_turns': allowed,
                               'allowed_original_model_turns': original_allowed,
                               'repair_confirmed': [s['repair_confirmed'] for s in segments],
                               'full_compressor_input_check': partial_stage_check(episode)})
        original = [read(PILOT / arm / f'turn_{i:02d}/checkpoint.json')['payload']
                    for i in range(4)]
        snapshots[arm] = {'initial_world': base,
                          'original_world_after_each_action': [selected_state(x) for x in original],
                          'model_input_image_hashes': views}
        _, history = source_history(arm)
        masks = [set(c) for n in range(len(original_allowed) + 1)
                 for c in itertools.combinations(original_allowed, n)]
        for number, drops in enumerate(masks):
            run = args.ram_output / arm / f'candidate_{number:02d}'
            run.mkdir(parents=True)
            native, scene, registry, tools, assets, prior, state_io, frozen = native_owner(run)
            steps, events = [], []
            for i, h in enumerate(history):
                if i in drops:
                    continue
                before = live_state(base, scene, registry)
                expected_before = base if i == 0 else selected_state(original[i-1])
                event = h['feedback'][0]
                call = event['call']
                response = json.loads(tools[call['name']](**call['arguments']))
                total_actions += 1
                assert total_actions <= protocol['max_native_actions']
                after = live_state(base, scene, registry)
                steps.append({'source_model_turn': i, 'call': call, 'response': response,
                              'original_response': event['response'],
                              'input_world_matches_original': before == expected_before,
                              'response_matches_original': response == event['response'],
                              'output_world_matches_original': after == selected_state(original[i]),
                              'before_world_sha256': digest(before),
                              'after_world_sha256': digest(after),
                              'original_input_images_verified': views[i]})
                events.append({'call': call, 'response': response})
            final = live_state(base, scene, registry)
            assert final == selected_state(original[-1]), 'Endpoint changed; preserve failure, do not tune tolerance'
            if not drops:
                assert all(s['input_world_matches_original'] and s['response_matches_original']
                           and s['output_world_matches_original'] for s in steps), 'Baseline replay differs'
                pairs = furniture_collisions(scene)
                recorded_pairs = read(PILOT / arm / 'turn_03/audit.json')['furniture_surface_intersection_pairs']
                assert pairs == recorded_pairs
                checks.append({'arm': arm, 'new_native_mesh_intersection_pairs': pairs,
                               'matches_saved_audit': True,
                               'planar_simultaneous_use_from_saved_identical_endpoint':
                                   read(PILOT / arm / 'turn_03/joint.json'),
                               'full_scene_success': None})
            state_io.save(run / 'checkpoint.json', scene, registry, assets, prior + events, frozen, native.G)
            row = {'arm': arm, 'candidate_index': number, 'dropped_original_model_turns': sorted(drops),
                   'retained_actions': len(steps), 'final_world_preserved_exactly': True,
                   'all_retained_input_worlds_preserved': all(s['input_world_matches_original'] for s in steps),
                   'all_retained_tool_responses_preserved': all(s['response_matches_original'] for s in steps),
                   'world_and_response_reusable': all(s['input_world_matches_original'] and s['response_matches_original'] for s in steps),
                   'requires_rebuilt_policy_history': bool(drops),
                   'old_reasoning_admitted': not bool(drops),
                   'training_admitted': False,
                   'non_admission': ['Furniture-only development trace, not four-stage source',
                                     'Endpoint fails declared simultaneous-use proxy; no full-room success label',
                                     'Any compressed policy memory/reasoning must be rebuilt, not copied'],
                   'steps': steps, 'final_world_sha256': digest(final)}
            rows.append(row)
    assert total_actions == 24 and len(rows) == 8
    dump(args.output / 'state_and_observation_evidence.json', snapshots)
    dump(args.output / 'source_input_hashes.json', [{'path': str(p), 'sha256': sha(p)} for p in sorted(inputs)])
    dump(args.output / 'candidate_detection.json', candidate_rows)
    dump(args.output / 'endpoint_mesh_recheck.json', checks)
    dump(args.output / 'replay_results.json', {'status': 'completed', 'rows': rows,
        'elapsed_seconds': time.perf_counter() - started, 'native_actions': total_actions,
        'deletion_replays': len(rows), 'new_model_calls': 0, 'training_steps': 0,
        'new_API_calls': 0, 'new_scenes_generated': 0,
        'full_official_CTC_runs': 0, 'full_physics_runs': 0})
    print(canonical({'status': 'completed', 'replays': len(rows), 'native_actions': total_actions,
                     'world_preserved': sum(r['final_world_preserved_exactly'] for r in rows),
                     'world_and_response_reusable': sum(r['world_and_response_reusable'] for r in rows)}))


if __name__ == '__main__':
    main()
