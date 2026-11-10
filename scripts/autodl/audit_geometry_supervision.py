"""Audit actual view provenance and extract current-state geometry targets.

This does not reconstruct, infer, render, execute actions, or train. Existing
policy observations and action/decision labels remain untouched. Extracted
poses describe the recorded state, not a good design or future action effect.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import time


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def rotation(q):
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        raise ValueError('Invalid recorded quaternion')
    norm = math.sqrt(sum(v * v for v in q))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError('Recorded quaternion not normalized')
    w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]


def geometry(obj):
    transform = obj['transform']
    p = transform['translation']; q = transform['rotation_wxyz']
    if len(p) != 3 or not all(math.isfinite(v) for v in p):
        raise ValueError('Invalid recorded translation')
    rot = rotation(q)
    lo, hi = obj.get('bbox_min'), obj.get('bbox_max')
    corners = None
    size = None
    if lo is not None and hi is not None:
        if len(lo) != 3 or len(hi) != 3 or any(a > b for a, b in zip(lo, hi)):
            raise ValueError('Invalid local bounding box')
        size = [b-a for a, b in zip(lo, hi)]
        corners = [[p[j] + sum(rot[j][k] * c[k] for k in range(3)) for j in range(3)]
                   for c in itertools.product(*zip(lo, hi))]
    return {'object_id': obj['object_id'], 'object_type': obj['object_type'],
            'position_m': p, 'rotation_wxyz': q, 'local_bbox_extent_m': size,
            'world_bbox_corners_m': corners, 'scale_factor': obj.get('scale_factor'),
            'immutable': obj['immutable'], 'semantic_front_verified': False,
            'local_bbox_scale_convention': 'Recorded bounds used directly; no second scale_factor multiplication',
            'geometry_target_mask': {'pose': True, 'local_extent': size is not None,
                                     'strict_positive_extent': size is not None and min(size) > 0},
            'label_meaning': 'Current engine state, not expert design quality or future outcome'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(); start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    material = REPO / 'runs/autodl/20261110_storage_observation_recovery'
    inputs = [material / 'joint_multimodal_v4/summary.json',
              material / 'material_v4/observation_bindings.json',
              REPO / 'runs/autodl/20261110_corpus_state_recovery/state_audit_v2/units.json']
    hashes = {str(p): sha(p) for p in inputs}
    encoding = read(inputs[0])
    bindings = {r['unit_id']: r for r in read(inputs[1])}
    states = {r['unit_id']: r for r in read(inputs[2])}
    view_rows = []; unit_rows = []; targets = []; sidecar_hashes = {}
    for unit in encoding['records']:
        uid = unit['unit_id']; binding = bindings.get(uid)
        current_views = []; changed_topdown = 0; cut_views = 0
        annotation = None
        if binding:
            annpath = resolve(binding['annotation_receipt'])
            assert sha(annpath) == binding['annotation_receipt_sha256']
            sidecar_hashes[str(annpath)] = sha(annpath)
            annotation = read(annpath)
        for index, image in enumerate(unit.get('images', [])):
            p = resolve(image['path']); actual = sha(p)
            assert actual == image['sha256'], uid
            with p.open('rb') as f:
                header = f.read(24)
            assert header.startswith(b'\x89PNG\r\n\x1a\n')
            width, height = struct.unpack('>II', header[16:24])
            original_sha = None
            if annotation:
                original_sha = next((value for key, value in annotation.get('unannotated_sha256', {}).items()
                                     if resolve(key) == p), None)
            overlay_changed = original_sha is not None and original_sha != actual
            raw_unchanged = original_sha is not None and original_sha == actual
            visibility_path = Path(str(p) + '.visibility.json')
            visibility = None
            if visibility_path.exists():
                visibility = read(visibility_path)
                sidecar_hashes[str(visibility_path)] = sha(visibility_path)
            meta_path = Path(str(p) + '.meta.json')
            metadata = None
            if meta_path.exists():
                metadata = read(meta_path); sidecar_hashes[str(meta_path)] = sha(meta_path)
            hidden = visibility.get('native_cut_ids', []) if visibility else []
            calibrated = False
            # These names are an explicit contract, not an attempt to infer
            # calibration from a filename or partially specified camera.
            camera_records = [v.get('camera', v) for v in (metadata, visibility) if isinstance(v, dict)]
            for camera in camera_records:
                if (any(k in camera for k in ('intrinsics', 'K'))
                        and any(k in camera for k in ('world_to_camera', 'camera_to_world', 'extrinsics'))):
                    calibrated = True
            row = {'unit_id': uid, 'index': index, 'path': str(p), 'sha256': actual,
                   'width': width, 'height': height,
                   'raw_rgb_unchanged_by_annotation_receipt': raw_unchanged,
                   'overlay_change_proved_by_hash': overlay_changed,
                   'unannotated_sha256_if_recorded': original_sha,
                   'annotation_provenance_known': original_sha is not None,
                   'visibility_receipt': str(visibility_path) if visibility else None,
                   'camera_location_recorded': bool(visibility and 'camera_location' in visibility),
                   'explicit_full_intrinsics_and_extrinsics': calibrated,
                   'cutaway_object_ids': hidden,
                   'per_view_visibility_mask_available': False,
                   'photometric_training_admitted': False,
                   'why_not_admitted': 'No complete raw/calibration/per-view visibility contract; do not infer nonexistence of a reconstructable source camera'}
            changed_topdown += int(overlay_changed); cut_views += int(bool(hidden))
            view_rows.append(row); current_views.append(row)
        target = {'unit_id': uid, 'training_admitted': False,
                  'original_actions_and_decisions_unchanged': True,
                  'image_set_complete': unit['status'] == 'encoded',
                  'current_state_geometry': None}
        if uid in states:
            state_info = states[uid]; p = resolve(state_info['local_state']); state = read(p)
            canonical = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            assert canonical == state_info['state_canonical_sha256']
            sidecar_hashes[str(p)] = sha(p)
            if binding:
                assert binding['source_state_sha256'] == canonical
                assert annotation['source_room_sha256'] == canonical
            target.update(state_source=str(p), state_canonical_sha256=canonical,
                          source_to_image_binding_verified=binding is not None,
                          current_state_geometry=[geometry(obj) for obj in state['objects'].values()],
                          frame={'up_axis': '+Z', 'distance_unit': 'metres', 'quaternion_order': 'wxyz',
                                 'source': 'Existing native state/observation contract; no Y-up remapping'})
        else:
            target.update(source_to_image_binding_verified=False,
                          reason='No original full state exported in this particular recovery ledger; not a claim that no state exists elsewhere')
        targets.append(target)
        unit_rows.append({'unit_id': uid, 'observation_status': unit['status'],
                          'observation_scope': unit.get('observation_scope'),
                          'available_views': len(current_views),
                          'views_with_proved_overlay_change': changed_topdown,
                          'views_with_recorded_cutaway': cut_views,
                          'explicitly_calibrated_views': sum(v['explicit_full_intrinsics_and_extrinsics'] for v in current_views),
                          'current_state_objects_extracted': len(target['current_state_geometry'] or []),
                          'source_to_image_binding_verified': target['source_to_image_binding_verified'],
                          'photometric_training_admitted': False})
    assert len(targets) == 24 and len(view_rows) == 19 * 5
    assert all(sha(Path(p)) == expected for p, expected in hashes.items())
    assert all(sha(Path(p)) == expected for p, expected in sidecar_hashes.items())
    def write(name, value):
        (args.out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    write('view_provenance.json', view_rows)
    write('unit_admission.json', unit_rows)
    with (args.out / 'current_state_geometry.jsonl').open('w') as f:
        for target in targets:
            f.write(json.dumps(target, ensure_ascii=False) + '\n')
    with (args.out / 'unit_admission.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(unit_rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(unit_rows)
    write('input_hashes.json', {'inputs': hashes, 'sidecars': sidecar_hashes})
    summary = {'status': 'actual_current_material_geometry_provenance_extracted',
               'fixed_units': len(unit_rows), 'available_image_units': 19, 'missing_image_units': 5,
               'existing_images_read': len(view_rows),
               'views_with_annotation_hash_change': sum(v['overlay_change_proved_by_hash'] for v in view_rows),
               'views_with_recorded_cutaway': sum(bool(v['cutaway_object_ids']) for v in view_rows),
               'views_with_unchanged_raw_rgb_receipt': sum(v['raw_rgb_unchanged_by_annotation_receipt'] for v in view_rows),
               'explicit_full_camera_records': sum(v['explicit_full_intrinsics_and_extrinsics'] for v in view_rows),
               'recorded_current_state_units': sum(t['current_state_geometry'] is not None for t in targets),
               'recorded_current_state_objects': sum(len(t['current_state_geometry'] or []) for t in targets),
               'state_and_image_bound_units': sum(t['source_to_image_binding_verified'] for t in targets),
               'photometric_training_admitted_units': 0, 'current_geometry_training_admitted_units': 0,
               'original_supervision_modified': False, 'new_labels_of_good_design': 0,
               'new_model_calls': 0, 'new_renders': 0, 'new_training_updates': 0,
               'new_complete_scenes': 0, 'sources_unchanged': True,
               'scope': 'Input qualification and optional auxiliary current-state targets; not a trained geometric representation or full scene result',
               'seconds': time.monotonic()-start}
    write('summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
