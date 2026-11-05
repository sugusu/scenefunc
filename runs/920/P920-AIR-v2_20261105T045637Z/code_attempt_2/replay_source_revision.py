"""One frozen corrective move from a source-guided native checkpoint. Reuse verified initial views."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from catalog_project_session import create_app, canonical_hash
from native_project import sha256, validate_bundle
from preflight_sourced_trajectory import inspect
from measure_room_access import measure
from measure_free_floor import rigid_sweep_diagnostic
from native_checkpoint import rotation_roundtrip_differences
import shutil
from inspect_native_mesh_contacts import world_transform


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan = json.loads(a.plan.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(), 'begin one frozen source-guided corrective move', flush=True)
    result = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'tool_calls': 0,
              'observations': 0, 'model_calls': 0, 'training_updates': 0, 'checks': []}
    client = server = None; requests = []
    def check(name, condition):
        result['checks'].append({'name': name, 'passed': bool(condition)})
        if not condition: raise ValueError(name)
    try:
        for name, digest in plan['scripts'].items():
            check('code_'+name, sha256(Path(__file__).parent/name) == digest)
        check('frozen_program', sha256(Path(plan['program']['path'])) == plan['program']['sha256'])
        frozen = json.loads(Path(plan['program']['path']).read_text())
        command = frozen['single_program'][0]
        check('single_call_budget', len(frozen['single_program']) == 1)
        bundle = Path(plan['bundle'])
        check('input_hash', sha256(bundle/'manifest.json') == plan['manifest_sha256'])
        manifest, house, _, _ = validate_bundle(bundle); rid = manifest['selected_room_id']
        initial = house['rooms'][rid]
        oid = frozen['target_object_id']; args = command['arguments']
        check('single_move_contract', command['name'] == 'move_furniture_tool' and args['object_id'] == oid)
        check('preserved_source_endpoint', canonical_hash(initial) == frozen['initial_native_state_sha256'])
        proposed = copy.deepcopy(initial); yaw = math.radians(args['yaw'])
        proposed['objects'][oid]['transform'] = {'translation': [args['x'], args['y'], 0.0],
            'rotation_wxyz': [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]}
        geometry = {'initial': inspect(initial), 'proposed': inspect(proposed)}
        geometry['planned_sweep'] = rigid_sweep_diagnostic(initial, proposed, oid, bundle/('room_'+rid), bundle)
        dump(a.out/'PREFLIGHT.json', geometry)
        check('initial_and_proposed_geometry', all(not geometry[k]['outside_declared_wall_inner_faces'] and
            not geometry[k]['positive_3d_AABB_pairs'] for k in ('initial','proposed')))
        check('planned_sweep_clear', geometry['planned_sweep']['clear_under_declared_sweep_model'])
        for name in ('SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY','OPENAI_API_KEY','GOOGLE_API_KEY',
                     'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
            os.environ.pop(name, None)
        os.environ.update(OPENAI_API_KEY='disabled-local-validation', OPENAI_BASE_URL='http://127.0.0.1:9/v1',
            OPENAI_AGENTS_DISABLE_TRACING='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            NO_PROXY='127.0.0.1,localhost', SCENE_RENDER_THREADS='4', SCENE_RENDER_SAMPLES='8', OPENSCENEAGENT_STUB_ASSETS='0')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
        from fastapi.testclient import TestClient
        from PIL import Image
        app, server = create_app(Path(plan['designer']), Path(plan['scenesmith'])); client = TestClient(app)
        def post(route, payload=None):
            start = time.monotonic(); reply = client.post(route, json=payload or {}); data = reply.json()
            requests.append({'route': route, 'payload': payload, 'status_code': reply.status_code,
                             'elapsed_seconds': time.monotonic()-start, 'response': data})
            dump(a.out/'HTTP_REQUESTS.json', requests)
            if reply.status_code != 200: raise RuntimeError('Native route failed: '+route)
            return data
        restored = post('/session/catalog', {'bundle_directory': str(bundle), 'output_directory': str((a.out/'house').resolve())})
        check('restored_step', restored['resumed_step_idx'] == frozen['initial_step_idx'])
        session = server._state.session; scene = session.scene
        post('/session/stage', {'stage': 'furniture', 'stub_assets': False, 'skip_render': False})
        check('expected_initial_inventory', len(scene.objects) == frozen['expected_native_objects_before'])
        check('actual_catalog', len(session.current_context()['asset_catalog']) == frozen['catalog_entries'])
        def capture(label):
            state = scene.to_state_dict(); context = session.current_context()
            dump(a.out/(label+'_state.json'), state); dump(a.out/(label+'_context.json'), context)
            obs = post('/session/observation'); result['observations'] += 1
            bound = json.loads(Path(obs['state_binding']).read_text())
            check(label+'_state_context_bound', len(obs['image_paths']) == 5 and
                  bound['state_sha256'] == canonical_hash(state) and bound['current_context_sha256'] == canonical_hash(context))
            for image in obs['image_paths']:
                with Image.open(image) as im: im.verify()
            dump(a.out/(label+'_observation.json'), obs)
            metrics = measure(state, scene.scene_dir, session.project.directory, frozen['measurement'])
            dump(a.out/(label+'_measurement.json'), metrics)
            return state, metrics
        # The source endpoint has already been rendered. Reuse its exact bound views only after cold restoration matches.
        before = scene.to_state_dict(); reuse = frozen['reuse_initial_evidence']
        for label, ref in reuse.items():
            check('reuse_hash_'+label, sha256(Path(ref['path'])) == ref['sha256'])
        original = json.loads(Path(reuse['state']['path']).read_text())
        context = json.loads(Path(reuse['context']['path']).read_text())
        obs = json.loads(Path(reuse['observation']['path']).read_text())
        binding = json.loads(Path(obs['state_binding']).read_text())
        actual_context = session.current_context()
        dump(a.out/'actual_restored_state.json', before); dump(a.out/'actual_restored_context.json', actual_context)
        state_differences = rotation_roundtrip_differences(original, before)
        context_differences = rotation_roundtrip_differences(context, actual_context)
        dump(a.out/'RESTORATION.json', {'source_state_sha256': canonical_hash(original),
            'restored_state_sha256': canonical_hash(before), 'state_differences': state_differences,
            'context_differences': context_differences, 'runtime_transitions': session.restoration_transitions,
            'view_reuse_scope': 'Existing source-endpoint views retained under strictly quaternion-only <=1e-12 numerical equivalence; not byte-identical actual restored state or new rendering.'})
        check('cold_restore_quaternion_only_equivalence', canonical_hash(original) == binding['state_sha256'])
        check('cold_restore_context_quaternion_only_equivalence', canonical_hash(context) == binding['current_context_sha256'])
        check('reuse_five_views', len(obs['image_paths']) == len(binding['images']) == 5)
        for image in binding['images']:
            check('reuse_view_'+Path(image['path']).stem, sha256(Path(image['path'])) == image['sha256'])
        for label, ref in reuse.items():
            if label in ('state','context','observation','measurement'):
                shutil.copyfile(ref['path'], a.out/('initial_'+label+'.json'))
        before_metrics = json.loads(Path(reuse['measurement']['path']).read_text())
        result['reused_observations'] = 1
        result['tool_calls'] += 1
        reply = post('/session/step', {'tool_calls': [command]})
        dump(a.out/'STEP.json', {'command': command, 'reply': reply, 'before_sha256': canonical_hash(before)})
        output = reply['results'][0]['output']
        check('native_move_accepted', output.get('success') is True)
        actual = scene.to_state_dict()
        check('all_ids_retained', set(actual['objects']) == set(before['objects']))
        check('all_non_target_objects_unchanged', all(actual['objects'][key] == value for key, value in before['objects'].items() if key != oid))
        check('only_target_transform_changed', {k:v for k,v in actual['objects'][oid].items() if k != 'transform'} ==
              {k:v for k,v in before['objects'][oid].items() if k != 'transform'})
        check('asset_identity', session.current_context()['asset_bindings'][oid]['registry_id'] == frozen['registry_id'])
        actual_sweep = rigid_sweep_diagnostic(before, actual, oid, scene.scene_dir, session.project.directory)
        dump(a.out/'ACTUAL_SWEEP.json', actual_sweep)
        check('actual_sweep_clear', actual_sweep['clear_under_declared_sweep_model'])
        endpoint = inspect(actual); dump(a.out/'ACTUAL_GEOMETRY.json', endpoint)
        check('actual_endpoint_geometry', not endpoint['outside_declared_wall_inner_faces'] and not endpoint['positive_3d_AABB_pairs'])
        after, after_metrics = capture('after')
        checkpoint = post('/session/checkpoint', {'output_directory': str((a.out/'checkpoint').resolve())})
        rail = after['objects'][oid]; definition = frozen['rail_approach_definition']; strip_results = []
        for body in frozen['measurement']['square_body_widths_m']:
            radius = body/2; lo, hi = np.array(rail['bbox_min']), np.array(rail['bbox_max'])
            start, end = lo[0]+radius, hi[0]-radius
            if start > end: start = end = (lo[0]+hi[0])/2
            points = {}
            for side in definition['local_sides']:
                y = hi[1]+radius+definition['clearance_m'] if side == '+Y' else lo[1]-radius-definition['clearance_m']
                for i, x in enumerate(np.linspace(start, end, definition['line_samples'])):
                    points[side+'_'+str(i)] = list(map(float, (world_transform(rail) @ np.array([x,y,0,1]))[:2]))
            settings = {**frozen['measurement'], 'square_body_widths_m': [body], 'approach_points_xy_m': points}
            measured = measure(after, scene.scene_dir, session.project.directory, settings)['scenarios'][0]
            strip_results.append(measured)
        dims = np.array(rail['bbox_max'])-rail['bbox_min']
        summary = {'moved_object_id': oid, 'registry_id': frozen['registry_id'], 'before_objects': len(before['objects']),
            'after_objects': len(after['objects']), 'fixed_point_access': [], 'rail_approaches': strip_results,
            'unscaled_dimensions_m': dims.tolist(), 'retained_floor_AABB_m2': float(dims[0]*dims[1]), 'added_floor_AABB_m2': 0.0,
            'source_supported_strategy': 'Open clothing provision for two short-stay visitors.',
            'geometry_only': True, 'historical_trajectory': False, 'training_admitted': False,
            'unknowns': frozen['assumptions']}
        for bm, am in zip(before_metrics['scenarios'], after_metrics['scenarios']):
            summary['fixed_point_access'].append({'body_width_m': bm['square_body_width_m'],
                'before': sum(t['reachable_from_entry'] for t in bm['targets'].values()),
                'after': sum(t['reachable_from_entry'] for t in am['targets'].values()),
                'restored': [n for n in bm['targets'] if not bm['targets'][n]['reachable_from_entry'] and am['targets'][n]['reachable_from_entry']],
                'total': len(bm['targets']), 'lost': [n for n in bm['targets'] if bm['targets'][n]['reachable_from_entry'] and not am['targets'][n]['reachable_from_entry']],
                'before_reachable_floor_center_area_m2': bm['reachable_floor_center_area_m2'],
                'after_reachable_floor_center_area_m2': am['reachable_floor_center_area_m2']})
        summary['before_rail_approaches'] = json.loads(Path(reuse['summary']['path']).read_text())['rail_approaches']
        summary['acceptance'] = {
            'all_seven_fixed_points_reachable_at_all_widths': all(x['after'] == x['total'] == 7 for x in summary['fixed_point_access']),
            'at_least_one_complete_rail_side_reachable_per_width': all(any(all(s['targets'][side+'_'+str(i)]['reachable_from_entry']
                for i in range(definition['line_samples'])) for side in definition['local_sides']) for s in strip_results)}
        summary['conditional_objective_pass'] = all(summary['acceptance'].values())
        summary['professional_preference'] = 'unknown'
        summary['repair_origin'] = 'Researcher geometric correction informed by retained v1 failure; not a designer-authored relocation or model output.'
        dump(a.out/'SUMMARY.json', summary)
        step = json.loads((a.out/'STEP.json').read_text()); step.update(after_sha256=canonical_hash(after), object_id=oid, checkpoint=checkpoint)
        dump(a.out/'STEP.json', step)
        result.update(status='source_guided_native_revision_completed', checkpoint=checkpoint, moved_object_id=oid,
                      conditional_objective_pass=summary['conditional_objective_pass'],
                      whole_room_usability='unknown', professional_preference_label=False, training_admitted=False)
    except BaseException as exc:
        result.update(status='failed', failure=type(exc).__name__+': '+str(exc)); raise
    finally:
        if server is not None: server._close_all()
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
        result['ended_at'] = datetime.now(timezone.utc).isoformat(); dump(a.out/'RESULT.json', result)
        print(result['ended_at'], result['status'], flush=True)


if __name__ == '__main__': main()
