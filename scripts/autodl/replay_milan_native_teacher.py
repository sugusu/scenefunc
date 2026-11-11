"""Execute a saved ordinary completion through pinned native placement closures.

No model call, generative service, SDK bootstrap, training, or geometry search.
The two ordered actions are one solver-teacher decision, not two independently
observed model decisions. Tool success and geometry validity are kept separate.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import resource
import shutil
import socket
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh

W = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
R = Path(__file__).resolve().parents[2]
BRIDGE = W/'native_room_bridge_20261104'
NATIVE = R/'runs/autodl/20261110_native_state_runtime/source'
HELPERS = R/'runs/autodl/20261110_public_function_search/original_evaluator'
OLD = R/'runs/autodl/20261111_milan_completion'
SCHEMA = W/'professional_data_ingest_20261104/native_src/stage_schemas.py'
SFT = W/'autodl_continuation_20261110/training_source_review_20261110/upstream/src/sft_dataset.py'


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--scratch', type=Path, required=True)
    a = ap.parse_args()
    root, bundle, scratch = a.root.resolve(), a.bundle.resolve(), a.scratch.resolve()
    if root.exists() or scratch.exists():
        raise FileExistsError('Fresh destinations required; no implicit replay retry')
    if shutil.disk_usage(R).free < 5*1024**3 + 3*1024**2:
        raise RuntimeError('Keep five GiB data reserve; large intermediates go to RAM')
    root.mkdir(parents=True); scratch.mkdir(parents=True)
    started = time.monotonic()
    network_attempts = []
    def offline(*_, **__):
        network_attempts.append(True)
        raise RuntimeError('This CPU replay forbids network')
    socket.socket.connect = offline
    socket.create_connection = offline
    sys.path[:0] = [str(NATIVE), str(HELPERS)]
    from native_project import validate_bundle, import_project
    from scenesmith.agent_utils import room as core
    from inspect_native_mesh_contacts import sdf_visuals, world_transform
    from trimesh.collision import mesh_to_BVH
    import fcl

    manifest, house, runtime, task = validate_bundle(bundle)
    before = {str(p):sha(p) for p in bundle.rglob('*') if p.is_file()}
    rid = manifest['selected_room_id']
    initial = house['rooms'][rid]
    catalog = read(bundle/'catalog.json')['entries']
    expected = read(OLD/'completed_room_state.json')
    assert read(OLD/'completion_result.json')['selected_candidate_id'] == 173
    assert len(initial['objects']) == 30 and len(catalog) == 16
    dump(root/'protocol.json', dict(
        native_revision='e888297a8b0a2f3acbb8634edf9def59cceab836',
        source_manifest_sha256=sha(bundle/'manifest.json'),
        source_completion_result_sha256=sha(OLD/'completion_result.json'),
        decision_origin='Already executed ordinary finite CSP, selected candidate173; no new proposal/model/search',
        batch='One decision containing two ordered add calls; no invented intermediate observation',
        wrapper_omission='Only SDK function_tool decorator and unused service constructors; unchanged native closure body',
        noise='Explicit deterministic zero noise, not a reproduction of stochastic author settings',
        equality_tolerance_m_and_matrix=1e-12,
        controls='One invalid catalog ID, one out-of-bounds call, one deliberate overlapping placement; not model errors or scene benchmark examples',
        observation='Frozen five initial native views, unchanged; no new final view claimed',
        paid_API_calls=0, model_calls=0, training_updates=0, new_scene_generations=0,
        source_family='Previously exposed Milan furnished development room, not held out',
        semantic_scope='Declared geometric proxies only; raw-brief interpretation and human acceptance unknown'))

    n = load('native_teacher_kernels', BRIDGE/'native_kernels.py')
    # Same source hashes are necessary, but distinct Enum identities also matter.
    # Bind the extracted functions to the actual native restoration classes.
    for name in ['SceneObject', 'RoomScene', 'UniqueID', 'ObjectType',
                 'PlacementInfo', 'SupportSurface', 'copy_scene_object_with_new_pose']:
        n.G[name] = getattr(core, name)
    for rel in ['agent_utils/room.py', 'agent_utils/house.py', 'utils/sdf_utils.py']:
        assert sha(BRIDGE/'source/scenesmith'/rel) == sha(NATIVE/'scenesmith'/rel)
    project = import_project(bundle, scratch/'native_import')
    scene = project.room
    assert scene.to_state_dict() == initial
    registry = n.G['AssetRegistry']()
    for e in catalog:
        registry.register(core.SceneObject.from_dict(e['template'], scene_dir=scene.scene_dir))
    assert registry.size() == 16
    registry.save_to_file(root/'asset_registry.json')
    restored_registry = n.G['AssetRegistry']()
    restored_registry.load_from_file(root/'asset_registry.json')
    assert [str(x.object_id) for x in restored_registry.list_all()] == [str(x.object_id) for x in registry.list_all()]
    for asset in registry.list_all():
        assert asset.to_dict(scene_dir=scene.scene_dir) == restored_registry.get(asset.object_id).to_dict(scene_dir=scene.scene_dir)
    tools = n.make_tools(scene, registry, {})
    validator = load('native_teacher_schema', SCHEMA)
    states = [copy.deepcopy(initial)]
    log, mapping, actions = [], {}, []
    for logical in ['ordinary_work_table_0', 'ordinary_work_chair_0']:
        target = expected['objects'][logical]
        entry = next(e for e in catalog if e['template']['name'] == target['name'])
        q = target['transform']['rotation_wxyz']
        yaw = math.degrees(2*math.atan2(q[3], q[0]))
        args = dict(asset_id=entry['template']['object_id'], x=target['transform']['translation'][0],
                    y=target['transform']['translation'][1], yaw=yaw)
        errors = validator.validate_tool_call('furniture', 'add_furniture_to_scene_tool', args)
        assert not errors
        action = dict(name='add_furniture_to_scene_tool', arguments=args)
        raw = tools[action['name']](**args)
        receipt = json.loads(raw)
        assert receipt['success']
        oid = receipt['object_id']
        assert oid not in initial['objects'] and oid not in mapping.values()
        actual = scene.to_state_dict()
        assert len(actual['objects']) == len(states[-1]['objects'])+1
        assert all(actual['objects'][i] == obj for i, obj in states[-1]['objects'].items())
        mapping[logical] = oid
        o = actual['objects'][oid]
        assert np.allclose(world_transform(o), world_transform(target), rtol=0, atol=1e-12)
        assert all(o[k] == target[k] for k in target if k not in ['object_id', 'transform'])
        runtime['rooms'][rid]['asset_bindings'][oid] = {**entry['binding'],
            'registry_id':entry['template']['object_id'],
            'identity_provenance':'native_kernel_replay_of_existing_ordinary_completion'}
        log.append(dict(index=len(log), action=action, raw_receipt=raw, receipt=receipt,
                        logical_teacher_id=logical, actual_native_id=oid,
                        added_ids=[oid], unchanged_prior_object_count=len(states[-1]['objects'])))
        actions.append(action); states.append(actual)
    for i, state in enumerate(states):
        dump(root/f'state_{i:03d}.json', state)
    dump(root/'actual_tool_trace.json', log)
    dump(root/'logical_to_native_ids.json', mapping)

    # Native receipts perform registry lookup and center bounds checks, not mesh validation.
    controls = []
    initial_final = copy.deepcopy(scene.to_state_dict())
    for label, args in [
        ('unknown_asset', dict(asset_id='catalog_missing_control', x=0., y=0., yaw=0.)),
        ('center_out_of_bounds', dict(asset_id=actions[0]['arguments']['asset_id'], x=100., y=0., yaw=0.))]:
        assert not validator.validate_tool_call('furniture', 'add_furniture_to_scene_tool', args)
        receipt = json.loads(tools['add_furniture_to_scene_tool'](**args))
        assert not receipt['success'] and scene.to_state_dict() == initial_final
        controls.append(dict(control=label, synthetic=True, action=args, receipt=receipt,
                             state_unchanged=True))

    # Single synthetic overlapping clone: exact same real table geometry and pose.
    args = copy.deepcopy(actions[0]['arguments'])
    receipt = json.loads(tools['add_furniture_to_scene_tool'](**args))
    assert receipt['success']
    overlap_id = receipt['object_id']
    overlap_state = scene.to_state_dict()
    room_dir = bundle/('room_'+rid)
    mesh_cache = {}
    def mesh_of(obj):
        m = trimesh.util.concatenate([m for _,m in sdf_visuals(room_dir/obj['sdf_path'], bundle, mesh_cache)])
        m.apply_transform(world_transform(obj))
        return m
    left, right = mapping['ordinary_work_table_0'], overlap_id
    lmesh, rmesh = [mesh_of(overlap_state['objects'][i]) for i in [left, right]]
    res = fcl.CollisionResult()
    fcl.collide(fcl.CollisionObject(mesh_to_BVH(lmesh)), fcl.CollisionObject(mesh_to_BVH(rmesh)),
                fcl.CollisionRequest(), res)
    assert res.is_collision
    scene.remove_object(core.UniqueID(overlap_id))
    assert scene.to_state_dict() == initial_final
    controls.append(dict(control='identical_table_overlap', synthetic=True, action=args,
        receipt=receipt, triangle_surface_intersection=True, object_ids=[left,right],
        restored_after_diagnostic=True,
        conclusion='Native tool success is execution acceptance, not geometric reward or semantic truth'))
    dump(root/'tool_acceptance_controls.json', controls)

    final = scene.to_state_dict()
    final_bundle = scratch/'final_bundle'
    shutil.copytree(bundle, final_bundle)
    house['rooms'][rid] = final
    dump(final_bundle/'house_state.json', house)
    dump(final_bundle/'runtime.json', runtime)
    final_manifest = copy.deepcopy(manifest)
    for p in ['house_state.json', 'runtime.json']:
        final_manifest['files'][p] = dict(bytes=(final_bundle/p).stat().st_size, sha256=sha(final_bundle/p))
    final_manifest['checkpoint_parent'] = dict(manifest_sha256=sha(bundle/'manifest.json'))
    dump(final_bundle/'manifest.json', final_manifest)
    validate_bundle(final_bundle)
    restored = import_project(final_bundle, scratch/'roundtrip_import')
    assert restored.room.to_state_dict() == final
    for logical, oid in mapping.items():
        assert np.allclose(world_transform(final['objects'][oid]), world_transform(expected['objects'][logical]), rtol=0, atol=1e-12)
    # Actual native removal on the completed scene restores initial state exactly.
    for oid in reversed(list(mapping.values())):
        scene.remove_object(core.UniqueID(oid))
    assert scene.to_state_dict() == initial
    dump(root/'final_bundle_metadata.json', dict(manifest=final_manifest, runtime=runtime,
        persistent_state=str(root/'state_002.json'), original_bundle=str(bundle),
        replay_bundle_in_RAM=str(final_bundle), new_geometry_files=0,
        reproduction='Copy original bundle; replace selected room state, runtime, manifest with recorded values. All source resources unchanged.'))

    # Re-execute the unchanged object-relative evaluator on the tool-produced state.
    # Its module harness alone is relocated; predicates and thresholds are unchanged.
    ray_src = OLD/'source/scripts/920/measure_family_choice.py'
    ray_node = next(x for x in ast.parse(ray_src.read_text()).body if isinstance(x, ast.FunctionDef) and x.name=='hit_segment')
    env = {'np':np}; exec(compile(ast.Module(body=[ray_node], type_ignores=[]), str(ray_src), 'exec'), env)
    import types
    stub = types.ModuleType('measure_family_choice'); stub.hit_segment = env['hit_segment']
    sys.modules['measure_family_choice'] = stub
    evaluator = load('native_teacher_evaluator', OLD/'source/scripts/920/evaluate_milan_policy.py')
    plan = read(OLD/'relocated_evaluation_plan.json'); plan['initial_bundle'] = str(bundle)
    dump(root/'relocated_evaluation_plan.json', plan)
    oldargv = sys.argv
    sys.argv = ['native_teacher_evaluator', '--plan', str(root/'relocated_evaluation_plan.json'),
                '--bundle', str(final_bundle), '--out', str(root/'original_evaluator_result.json')]
    try:
        evaluator.main()
    finally:
        sys.argv = oldargv
    ev = read(root/'original_evaluator_result.json')
    assert ev['dedicated_work_geometry_found']
    assert all(ev['retained_original_ids'].values()) and all(ev['identity_assets_dimensions_preserved'].values())

    # Fresh FCL verification on every furniture pair and every structural visual.
    furniture = {k:mesh_of(o) for k,o in final['objects'].items() if o['object_type']=='furniture'}
    structures = {name:m for name,m in sdf_visuals(room_dir/final['room_geometry']['sdf_path'], bundle, mesh_cache)
                  if name not in ('floor','ceiling')}
    collision_objects = {k:fcl.CollisionObject(mesh_to_BVH(m)) for k,m in {**furniture, **structures}.items()}
    checked = 0; pairs = []
    for l, r in [*itertools.combinations(furniture, 2), *itertools.product(furniture, structures)]:
        ml = furniture[l]; mr = furniture.get(r, structures.get(r))
        if not np.all(np.minimum(ml.bounds[1],mr.bounds[1])-np.maximum(ml.bounds[0],mr.bounds[0])>1e-4):
            continue
        checked += 1; res = fcl.CollisionResult()
        fcl.collide(collision_objects[l],collision_objects[r],fcl.CollisionRequest(),res)
        if res.is_collision:pairs.append([l,r])
    assert not pairs
    dump(root/'fresh_mesh_contacts.json', dict(furniture=len(furniture), structures=len(structures),
        broad_phase_positive_pairs=checked, intersecting_pairs=pairs,
        fcl_version=fcl.__version__, trimesh_version=trimesh.__version__,
        limits='Visual triangle intersections; not solid containment, dynamics, usability, articulated door clearance or professional acceptance'))

    # A batch label contains no intermediate response in its input and no invented thought.
    views = read(R/'runs/autodl/20261111_activity_resource_program/view_manifest.json')
    for v in views:
        assert sha(v['local_path']) == v['sha256']
    fn = next(x for x in ast.parse(SFT.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_build_messages')
    env = {'Any':Any, 'StepSample':SimpleNamespace}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),str(SFT),'exec'),env)
    memory = dict(state=initial, asset_catalog=catalog, prior_tool_receipts=[],
        coordinates=dict(units='metres', up='+Z', frame='native_room_local', yaw='degrees'),
        design_status='Research transfer; dimensions, use proxies and asset suitability have documented unknowns')
    system = ('Complete the furnished room using the supplied catalog and preserve existing objects. '
        'Use add_furniture_to_scene_tool(asset_id: str, x: float, y: float, yaw: float=0). '
        'Coordinates are room-local metres; yaw is degrees. The tool creates new native IDs. '
        'An ordered batch may contain two calls. Tool success does not certify collision freedom or use. '
        'Return tool calls only; do not invent a rationale. This is a restricted offline catalog interface, '
        'not the complete original Actor system prompt.')
    sample = SimpleNamespace(stage='furniture', system=system, task_context=task['design_request'],
        memory_str=json.dumps(memory,ensure_ascii=False,allow_nan=False),
        scene_images=[v['local_path'] for v in views], image_path=None,
        assistant_content='', tool_calls=actions)
    full, prompt = env['_build_messages'](sample,no_thinking=True)
    assert full[-1]['content']=='' and len(full[-1]['tool_calls'])==2 and full[:-1]==prompt
    assert memory['prior_tool_receipts'] == []
    assert not any(i.startswith('ordinary_work_') for i in memory['state']['objects'])
    dump(root/'sft_full_messages.json',full);dump(root/'sft_prompt_messages.json',prompt)
    dump(root/'sft_sample_provenance.json',dict(
        decision_units=1, ordered_tool_calls=2, images=views, teacher='ordinary finite CSP selected173',
        chain_of_thought_present=False, model_generated=False, source_family_count=1,
        split='exposed development only', objective='Ordinary no-thinking conditional CE, not a new loss',
        prefix_has_intermediate_receipt=False, native_serializer_sha256=sha(SFT),
        author_original_system_prompt=False, schema_source_sha256=sha(SCHEMA),
        individual_ergonomic_or_subjective_quality_label=False,
        unconditional_training_admitted=False,
        admission_scope='Single replay-verified geometry-qualified ordinary-teacher batch for pipeline development only'))
    dump(root/'source_reading.json',dict(kernel_loader=dict(path=str(BRIDGE/'native_kernels.py'),sha256=sha(BRIDGE/'native_kernels.py')),
        native_definitions=n.READ, native_core_classes=[dict(path=str(NATIVE/'scenesmith'/rel),sha256=sha(NATIVE/'scenesmith'/rel))
        for rel in ['agent_utils/room.py','agent_utils/house.py','utils/sdf_utils.py']],
        schema=dict(path=str(SCHEMA),sha256=sha(SCHEMA)), sft_serializer=dict(path=str(SFT),sha256=sha(SFT))))
    assert before == {p:sha(p) for p in before}
    dump(root/'source_before_after.json',dict(files=len(before),all_unchanged=True,sha256=before))
    result = dict(status='native_closure_batch_and_geometry_replay_completed',
        ordinary_selected_candidate=173, added_native_ids=list(mapping.values()),
        teacher_decision_units=1, positive_tool_calls=2, diagnostic_tool_calls=3,
        original_objects_exactly_preserved=30, final_objects=32, final_furniture=15,
        native_state_roundtrip_exact=True, rollback_exact=True, issued_ID_persistence_across_deletion=False,
        ordinary_target_poses_match_atol_1e_12=True, final_triangle_intersections=0,
        declared_workspace_proxy_pass=True, source_resources_unchanged=True,
        full_scene_generation_success=None, HTTP_or_SDK_replay=False,
        model_calls=0, model_forwards=0, training_updates=0, paid_API_calls=0,
        network_attempts=len(network_attempts), wall_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(root/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
