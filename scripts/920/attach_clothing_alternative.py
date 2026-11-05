"""Attach one inspected closed-wardrobe alternative; retain all initial scene records."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from build_reference_room import add, box, numbers, prepare_mesh, save_sdf, sdf_root, write
from native_project import sha256, validate_bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan = json.loads(a.plan.read_text())
    print('begin source-specific catalog attachment; no scene edits', flush=True)
    for ref in plan['inputs'].values():
        if sha256(Path(ref['path'])) != ref['sha256']: raise ValueError('Changed input')
    sys.path.insert(0, plan['scenesmith'])
    from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix
    from scenesmith.agent_utils.room import SceneObject, ObjectType, UniqueID
    base = Path(plan['inputs']['base_manifest']['path']).parent
    manifest, state, _, _ = validate_bundle(base)
    a.out.mkdir(parents=True, exist_ok=False); bundle = a.out/'bundle'
    shutil.copytree(base, bundle)
    rid = manifest['selected_room_id']; roomdir = bundle/('room_'+rid)
    source = Path(plan['inputs']['reuse_manifest']['path']).parent
    oldmanifest, _, _, _ = validate_bundle(source)
    oldroom = source/('room_'+oldmanifest['selected_room_id'])
    oldcatalog = json.loads((source/oldmanifest['inputs']['asset_catalog']).read_text())
    entries = []
    for registry_id in plan['reuse_registry_ids']:
        entry = copy.deepcopy(next(e for e in oldcatalog['entries'] if e['template']['object_id'] == registry_id))
        for key in ('sdf_path', 'geometry_path'):
            relative = Path(entry['template'][key]).parent
            shutil.copytree(oldroom/relative, roomdir/relative, dirs_exist_ok=True)
        entries.append(entry)
    download = json.loads(Path(plan['inputs']['download']['path']).read_text())
    assert download['status'] == 'verified'
    groups = {item['role']: item for item in download['files']}
    assert len(groups) == len(download['files']) == 2
    for item in groups.values(): assert sha256(Path(item['local_path'])) == item['sha256']
    mesh = groups['mesh']; aid = mesh['asset_id']; reg = 'catalog_'+aid
    spec = {'asset_id': aid, 'source_path': mesh['local_path'], 'sha256': mesh['sha256'],
            'source_to_native_rotation': plan['source_to_native_rotation']}
    visual, bounds, gltf, rotation, shift = prepare_mesh(spec, roomdir/'assets'/aid)
    objdir = roomdir/'objects'/aid; objdir.mkdir(parents=True)
    xml, link = sdf_root(reg)
    node = add(link, 'visual', name='source_asset')
    rpy = RollPitchYaw(RotationMatrix(rotation)).vector()
    add(node, 'pose', numbers([*shift, *rpy]))
    add(add(add(node, 'geometry'), 'mesh'), 'uri', os.path.relpath(gltf, objdir))
    box(link, 'conservative_envelope', bounds.mean(axis=0), bounds[1]-bounds[0], [1,1,1,1], visual=False)
    sdf = objdir/'object.sdf'; save_sdf(sdf, xml)
    obj = SceneObject(UniqueID(reg), ObjectType.FURNITURE, 'closed_wardrobe',
        'Unscaled static closed wardrobe substitute; visually inspected door panels, no native articulation or internal capacity validation.',
        RigidTransform(), geometry_path=visual, sdf_path=sdf, bbox_min=bounds[0], bbox_max=bounds[1],
        metadata={'catalog_registry_id': reg, 'asset_source': 'hssd_catalog',
                  'source_to_native_rotation': plan['source_to_native_rotation']})
    entries.append({'template': obj.to_dict(scene_dir=roomdir),
                    'binding': {'asset_id': aid, 'identity_provenance': 'catalog_substitute_not_original_project_object'},
                    'limitations': ['Closed wardrobe is a researcher alternative, not a source-designer rejected candidate.',
                                    'Door swing, internal storage, garment handling, load and usability remain unknown.']})
    assert len(entries) == plan['catalog_entries']
    write(bundle/'catalog.json', {'schema': 'scenefunc-native-catalog-v1', 'entries': entries})
    manifest['inputs']['asset_catalog'] = 'catalog.json'
    provenance = json.loads((bundle/'provenance.json').read_text())
    provenance['catalog_attachment'] = {'plan_sha256': sha256(a.plan), 'old_room_poses_reused': False,
        'objects_added': 0, 'reused_templates': plan['reuse_registry_ids'], 'new_asset_id': aid}
    write(bundle/'provenance.json', provenance)
    manifest['files'] = {str(p.relative_to(bundle)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
        for p in sorted(bundle.rglob('*')) if p.is_file() and p != bundle/'manifest.json'}
    write(bundle/'manifest.json', manifest)
    _, after, _, _ = validate_bundle(bundle)
    assert state == after
    size = sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    assert size <= plan['max_bundle_bytes']
    write(a.out/'RESULT.json', {'status': 'furnished_input_catalog_attached', 'initial_state_unchanged': True,
        'objects_added': 0, 'catalog_entries': len(entries), 'bundle_bytes': size,
        'manifest_sha256': sha256(bundle/'manifest.json'), 'plan_sha256': sha256(a.plan)})
    print('catalog attached', len(entries), 'entries; scene state unchanged', flush=True)


if __name__ == '__main__': main()
