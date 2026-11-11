"""Bounded CPU comparison on the frozen Waterloo development reconstruction.

This is not an Agent, learned case transfer, or new whole-scene generation.
Two independently evaluated arms share a task-complete ordinary solver. The
case-derived join/separate hint is intentionally not allowed extra coordinates,
objects, constraints, or hidden output examples. If its compiled constraints
are identical, that equivalence is a result, not a new algorithm.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
from shapely.geometry import Polygon


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, allow_nan=False).encode()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def xy(obj):
    return np.array(obj['transform']['translation'][:2])


def move(state, oid, point):
    # Source permissions preserve Z, rotation, scale, identity and support.
    out = copy.deepcopy(state)
    if out['objects'][oid]['immutable']:
        raise ValueError('Immutable object: '+oid)
    out['objects'][oid]['transform']['translation'][:2] = [float(x) for x in point]
    return out


def cheap_modes(state, task, transform, footprint):
    a, b = [state['objects'][i] for i in task['table_ids']]
    ma, mb = transform(a), transform(b)
    rel = np.linalg.inv(ma) @ np.r_[mb[:3, 3], 1]
    distance = footprint(a).distance(footprint(b))
    joined = bool(distance <= task['joined_max_gap_m'] and abs(rel[1]) <= .03
                  and np.dot(ma[:2, 0], mb[:2, 0]) > .999
                  and abs(ma[2, 3]+a['bbox_max'][2]-mb[2, 3]-b['bbox_max'][2]) <= .01)
    align, places = {}, []
    for seat in task['seat_assignments']:
        t, c = [state['objects'][seat[k]] for k in ['table_id', 'chair_id']]
        tm = transform(t)
        local = np.linalg.inv(tm) @ np.r_[transform(c)[:3, 3], 1]
        lo, hi = t['bbox_min'], t['bbox_max']
        side = seat['side_y']
        edge = hi[1] if side > 0 else lo[1]
        gap = side*(local[1]-edge)-(c['bbox_max'][1]-c['bbox_min'][1])/2
        w, d = task['place_rectangle_width_depth_m']
        x, y = float(local[0]), edge-side*d/2
        corners = [[x-w/2, y-d/2], [x+w/2, y-d/2],
                   [x+w/2, y+d/2], [x-w/2, y+d/2]]
        within = all(lo[0]-1e-6 <= p[0] <= hi[0]+1e-6 and
                     lo[1]-1e-6 <= p[1] <= hi[1]+1e-6 for p in corners)
        align[seat['chair_id']] = bool(within and .05 <= gap <= .4)
        places.append(Polygon([(tm @ np.array([*p, 0, 1]))[:2] for p in corners]))
    overlap = any(x.intersection(y).area > 1e-8 for i, x in enumerate(places) for y in places[i+1:])
    work = bool(distance >= task['separate_min_gap_m'] and all(align[i] for i in task['work_chair_ids']))
    host = bool(joined and all(align.values()) and not overlap)
    return dict(joined=joined, seat_alignment=align, place_overlap=overlap,
                separate_work=work, joined_four_place_hosting=host,
                failure_count=int(not joined)+sum(not x for x in align.values())+int(overlap),
                table_gap_m=float(distance))


def candidates(initial, task):
    # Deliberately do not read the source FROZEN_PROGRAM.tool_calls or poses.
    aid, bid = task['table_ids']
    cb = task['work_chair_ids'][1]
    a, b, chair = [initial['objects'][i] for i in [aid, bid, cb]]
    if any(o['transform']['rotation_wxyz'] != [1., 0., 0., 0.] for o in [a, b, chair]):
        raise ValueError('This finite proposal contract requires frozen axis-aligned substitutes')
    seat_offset = xy(chair)-xy(b)
    half_width_sum = (a['bbox_max'][0]-a['bbox_min'][0]+b['bbox_max'][0]-b['bbox_min'][0])/2
    rows = [dict(candidate_id=0, actions=[], kind='stay')]
    for side, gap, dy, cx in itertools.product([-1, 1], [.01, .03, .05], [-.02, 0., .02], [-.15, 0., .15]):
        tb = xy(a)+[side*(half_width_sum+gap), dy]
        ch = tb+seat_offset+[cx, 0.]
        rows.append(dict(candidate_id=len(rows), kind='joint_translation', actions=[
            dict(object_id=bid, xy=tb.tolist()), dict(object_id=cb, xy=ch.tolist())]))
    return rows


def apply_candidate(initial, row):
    s = initial
    for action in row['actions']:
        s = move(s, action['object_id'], action['xy'])
    return s


def access_pass(measurement):
    return all(s['entry_cell_clear'] and all(t['reachable_from_entry'] for t in s['targets'].values())
               for s in measurement['scenarios'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--native-source', type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve(); bundle = a.bundle.resolve()
    if (root/'protocol.json').exists():
        raise FileExistsError('No automatic rerun or overwrite')
    start = time.monotonic()
    orig_eval = Path('runs/autodl/20261110_public_function_search/original_evaluator').resolve()
    sys.path.insert(0, str(orig_eval))
    sys.path.insert(0, str(a.native_source.resolve()))
    from inspect_native_mesh_contacts import sdf_visuals, world_transform
    from measure_free_floor import footprint, sweep_diagnostic
    from measure_room_access import measure
    from native_project import import_project, validate_bundle
    from pydrake.all import RigidTransform
    import fcl
    import trimesh
    from trimesh.collision import mesh_to_BVH
    pinned_modes = module('frozen_waterloo_modes', root/'source/code/measure_table_modes.py')
    _, state, _, taskfile = validate_bundle(bundle)
    rid = 'waterloo_table_transfer'; roomdir = bundle/('room_'+rid)
    initial = state['rooms'][rid]
    # The source metadata file is a scoring convention, never a plan input.
    scoring = read('runs/autodl/20261110_cpu_continuation/waterloo_availability/FROZEN_PROGRAM.json')
    fields = ['table_ids', 'work_chair_ids', 'seat_assignments', 'joined_max_gap_m',
              'separate_min_gap_m', 'place_rectangle_width_depth_m', 'access_settings']
    task = {k: scoring[k] for k in fields}
    # These seat assignments are exposed to *both* methods, and labeled
    # researcher-authored task inputs rather than automatically discovered.
    dump(root/'common_task.json', task)
    before_paths = [p for p in root.joinpath('source').rglob('*') if p.is_file()]
    before_paths += [Path('runs/autodl/20261110_cpu_continuation/waterloo_availability/initial_state.json'),
                     Path('runs/autodl/20261110_cpu_continuation/waterloo_availability/FROZEN_PROGRAM.json')]
    before = {str(p.resolve()): sha(p) for p in before_paths}
    protocol = dict(project='waterloo_table_transfer', source_split='train_source_candidate/development',
        independent_test=False, original_house_reconstructed=False,
        original_tool_actions_used_as_proposals=False, automatic_task_extraction=False,
        researcher_task_binding_visible_to_both=True,
        arms=['B0_task_aware_ordinary_CSP', 'B1_case_compiled_CSP_diagnostic'],
        semantics='B1 is a hand-compiled case strategy, not an Agent or learned mechanism.',
        common_budget=dict(candidate_cap=55, committed_single_object_moves_cap=4,
                           local_model_calls=0, paid_API_calls=0, training_updates=0),
        candidate_space=dict(table_sides=[-1, 1], seam_gaps_m=[.01, .03, .05],
                             table_y_offsets_m=[-.02, 0., .02], chair_x_offsets_m=[-.15, 0., .15]),
        common_order='task violation count; squared displacement; candidate id',
        early_stop='first candidate satisfying all frozen endpoint, access and conservative sweep checks',
        outcome='conditional geometry, not human comfort/physics/full scene success',
        decision_rule='Any factor already entailed by task constraints is deduplicated in both arms.',
        thresholds_from_frozen_source=True, no_quality_dependent_retry=True,
        source_reviewed_before_run=True)
    dump(root/'protocol.json', protocol)
    dump(root/'inputs_before.json', before)
    old = read('runs/autodl/20261110_cpu_continuation/waterloo_availability/initial_state.json')
    # Compare every value, using canonical native rotation matrices separately
    # if serialization picks a sign-equivalent quaternion. No tolerances relaxed.
    from load_paper_native_state import difference
    differences = difference(old, initial)
    dump(root/'cold_restore_comparison.json', dict(object_ids_exact=set(old['objects']) == set(initial['objects']),
        raw_value_differences=differences, raw_exact=not differences))
    assert set(old['objects']) == set(initial['objects'])
    for oid, obj in initial['objects'].items():
        assert np.max(np.abs(world_transform(obj)-world_transform(old['objects'][oid]))) <= 1e-12
        for k in ['bbox_min', 'bbox_max', 'immutable', 'scale_factor', 'geometry_path']:
            assert obj[k] == old['objects'][oid][k], (oid, k)
    native = import_project(bundle, bundle.parent/'search_native_import')
    assert len(native.room.objects) == 27
    dump(root/'native_restore.json', dict(native_class=type(native.house).__module__+'.'+type(native.house).__name__,
        object_records=27, raw_state_exact=not differences, native_pose_max_tolerance_m=1e-12,
        source_geometry_identical=True, new_scene_generation=False))
    # Full geometry is shared: visual SDFs include real sofa/table/stool meshes
    # and explicit proxy furnishings; no individual difficult object is dropped.
    meshes = {}; cache = {}
    for oid, obj in initial['objects'].items():
        if obj['sdf_path']:
            parts = [m for _, m in sdf_visuals(roomdir/obj['sdf_path'], bundle, cache)]
            meshes[oid] = trimesh.util.concatenate(parts)
    structure = []
    for name, mesh in sdf_visuals(roomdir/initial['room_geometry']['sdf_path'], bundle, cache):
        if name not in ('floor', 'ceiling') and 'door' not in name:
            structure.append((name, mesh))
    shapes = {oid: mesh_to_BVH(mesh) for oid, mesh in meshes.items()}
    structures = {name: fcl.CollisionObject(mesh_to_BVH(mesh)) for name, mesh in structure}
    def contacts(s):
        bodies = {oid: fcl.CollisionObject(shapes[oid], fcl.Transform(world_transform(o)[:3, :3], world_transform(o)[:3, 3]))
                  for oid, o in s['objects'].items() if oid in shapes}
        pairs = []
        # Only edited-vs-other checks are needed to count new pairs; include all
        # baseline pairs separately so unchanged errors cannot be hidden.
        bounds = {oid: trimesh.transform_points(np.array(list(itertools.product(*zip(o['bbox_min'], o['bbox_max'])))),
                                               world_transform(o)) for oid, o in s['objects'].items() if oid in bodies}
        bb = {i: np.array([p.min(0), p.max(0)]) for i, p in bounds.items()}
        for left, right in itertools.combinations(sorted(bodies), 2):
            overlap = np.minimum(bb[left][1], bb[right][1])-np.maximum(bb[left][0], bb[right][0])
            if not np.all(overlap > 1e-4):
                continue
            result = fcl.CollisionResult(); fcl.collide(bodies[left], bodies[right], fcl.CollisionRequest(), result)
            if result.is_collision:
                pairs.append([left, right])
        wall_pairs = []
        for oid, body in bodies.items():
            for name, wall in structures.items():
                wb = next(m.bounds for n, m in structure if n == name)
                if not np.all(np.minimum(bb[oid][1], wb[1])-np.maximum(bb[oid][0], wb[0]) > 1e-4):
                    continue
                result = fcl.CollisionResult(); fcl.collide(body, wall, fcl.CollisionRequest(), result)
                if result.is_collision:
                    wall_pairs.append([oid, name])
        return dict(furniture_pairs=pairs, structure_pairs=wall_pairs)
    baseline_contacts = contacts(initial)
    dump(root/'initial_contacts.json', baseline_contacts)
    initial_measure = pinned_modes.table_modes(initial, roomdir, bundle, task)
    dump(root/'initial_measurements.json', initial_measure)
    pool = candidates(initial, task)
    for row in pool:
        s = apply_candidate(initial, row)
        m = cheap_modes(s, task, world_transform, footprint)
        displacement = sum(float(np.sum((xy(s['objects'][oid])-xy(obj))**2))
                           for oid, obj in initial['objects'].items())
        row.update(cheap=m, order_key=[m['failure_count'], displacement, row['candidate_id']])
    pool.sort(key=lambda r: r['order_key'])
    dump(root/'common_candidates.json', pool)
    assert len(pool) == 55
    # Both ways of writing the constraint graph reduce to the same factors.
    factors = ['table_seam', 'parallel', 'top_height', 'seat_table_assignment',
               'place_disjointness', 'object_preservation', 'room_containment',
               'fixed_activity_access', 'seat_approach', 'no_new_intersection', 'straight_sweep']
    dump(root/'compiled_factor_equivalence.json', dict(B0=factors, B1=factors,
        factor_sets_equal=True, candidate_order_equal=True,
        reason='Join/separate tables and seating are explicit in the common task; the public case adds no constraint.',
        novelty_claim=False, learned_agent_run=False))
    results = []
    for arm in protocol['arms']:
        armstart = time.monotonic(); rows = []; selected = None
        for rank, row in enumerate(pool):
            t = time.monotonic(); s = apply_candidate(initial, row)
            record = dict(candidate_id=row['candidate_id'], rank=rank, cheap=row['cheap'],
                          exact_measurement_run=False, accepted=False)
            if row['cheap']['joined_four_place_hosting']:
                measure_result = pinned_modes.table_modes(s, roomdir, bundle, task)
                geom = contacts(s)
                newf = [p for p in geom['furniture_pairs'] if p not in baseline_contacts['furniture_pairs']]
                neww = [p for p in geom['structure_pairs'] if p not in baseline_contacts['structure_pairs']]
                inside = True
                for obj in s['objects'].values():
                    if not obj['sdf_path']:
                        continue
                    pts = trimesh.transform_points(np.array(list(itertools.product(*zip(obj['bbox_min'], obj['bbox_max'])))), world_transform(obj))
                    inside &= bool(np.all(pts[:, :2] >= [-4.-1e-6, -3.5-1e-6]) and
                                   np.all(pts[:, :2] <= [4.+1e-6, 3.5+1e-6]))
                # Compare both joint-action orders; no intermediate endpoint
                # must improve, but both paths and fixed activity routes must
                # remain valid. Max two moves, common budget to both arms.
                orders = []
                for actions in itertools.permutations(row['actions']):
                    current = initial; sweeps = []; intermediates = []
                    for action in actions:
                        after = move(current, action['object_id'], action['xy'])
                        sweeps.append(sweep_diagnostic(current, after, action['object_id'], roomdir, bundle))
                        intermediates.append(access_pass(measure(after, roomdir, bundle, task['access_settings'])))
                        current = after
                    orders.append(dict(object_order=[x['object_id'] for x in actions], sweeps=sweeps,
                        intermediate_access=intermediates,
                        valid=all(x['clear_under_declared_sweep_model'] for x in sweeps) and all(intermediates)))
                order = next((x for x in orders if x['valid']), None)
                accepted = (measure_result['mode_geometry']['joined_four_place_hosting'] and
                    access_pass(measure_result['fixed_access']) and
                    all(access_pass({'scenarios': [x]}) for x in measure_result['seat_approaches']) and
                    not newf and not neww and inside and order is not None)
                record.update(exact_measurement_run=True, accepted=bool(accepted),
                    new_furniture_pairs=newf, new_structure_pairs=neww, inside=inside,
                    fixed_access_pass=access_pass(measure_result['fixed_access']),
                    seat_approach_pass=all(access_pass({'scenarios': [x]}) for x in measure_result['seat_approaches']),
                    order_diagnostics=orders)
                dump(root/arm/f'candidate_{row["candidate_id"]:03d}_measurements.json', measure_result)
                if accepted:
                    selected = dict(row=row, state=s, order=order, measurements=measure_result, contacts=geom)
            record['cpu_wall_seconds'] = time.monotonic()-t
            rows.append(record)
            dump(root/arm/'evaluated_candidates.json', rows)
            if selected:
                break
        result = dict(arm=arm, evaluated_candidates=len(rows), proposed_candidates=len(pool),
            exact_measurements=sum(x['exact_measurement_run'] for x in rows),
            first_satisfying_rank=next((x['rank'] for x in rows if x['accepted']), None),
            committed_single_object_moves=0, selected_candidate_id=None,
            hosting_conditional_pass=False, return_state_exact=None,
            full_scene_generation_success=None, agent_model_calls=0)
        if selected:
            final = selected['state']; result['selected_candidate_id'] = selected['row']['candidate_id']
            # Actually modify fresh native objects, preserving immutable records.
            native_state = copy.deepcopy(initial)
            transitions = []
            for oid in selected['order']['object_order']:
                target = final['objects'][oid]['transform']['translation']
                native.room.objects[oid].transform = RigidTransform(np.array(target))
                native_state['objects'][oid]['transform']['translation'] = list(target)
                transitions.append(dict(object_id=oid, translation=target))
            host_state = native.room.to_state_dict()
            for oid in initial['objects']:
                assert np.max(np.abs(world_transform(host_state['objects'][oid])-world_transform(final['objects'][oid]))) <= 1e-12
            dump(root/arm/'hosting_state.json', host_state)
            dump(root/arm/'hosting_contacts.json', selected['contacts'])
            # Recover initial *from the current native state*, reverse validated
            # paths, no source FROZEN_PROGRAM endpoints used.
            for oid in reversed(selected['order']['object_order']):
                target = initial['objects'][oid]['transform']['translation']
                native.room.objects[oid].transform = RigidTransform(np.array(target))
                transitions.append(dict(object_id=oid, translation=target))
            returned = native.room.to_state_dict()
            assert not difference(initial, returned)
            dump(root/arm/'returned_state.json', returned)
            dump(root/arm/'committed_native_transforms.json', transitions)
            result.update(hosting_conditional_pass=True, return_state_exact=True,
                committed_single_object_moves=len(transitions),
                hosted_state_sha256=digest(host_state), return_state_sha256=digest(returned),
                preserved_objects=len(host_state['objects']),
                untouched_objects=25,
                remaining_furniture_intersections=len(selected['contacts']['furniture_pairs']),
                remaining_structure_intersections=len(selected['contacts']['structure_pairs']))
        result['cpu_wall_seconds'] = time.monotonic()-armstart
        results.append(result); dump(root/arm/'result.json', result)
        print(json.dumps(result), flush=True)
    after = {p: sha(p) for p in before}
    dump(root/'input_hashes_before_after.json', dict(before=before, after=after, unchanged=before == after))
    assert before == after
    dump(root/'summary.json', dict(arms=results, wall_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        same_selected_state=all(x.get('hosted_state_sha256') == results[0].get('hosted_state_sha256') for x in results),
        baseline_contacts=baseline_contacts, all_inputs_unchanged=True,
        new_model_calls=0, paid_API_calls=0, training_updates=0, new_scene_generations=0,
        independent_projects=1, scope='Second development project, same source family already used in research.',
        full_scene_generation_success=None))
    with (root/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)


if __name__ == '__main__':
    main()
