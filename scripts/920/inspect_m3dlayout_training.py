"""Inspect official train splits and stage deterministic, unbound layout records.

No mesh retrieval, native scene construction, resident-brief synthesis, or training.
Sizes remain half extents in the upstream coordinate system.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

EXPECTED_SHA = '10c0b11d43f980cd03f214cbd6bf377192257c7da58b56c94125d20375ef39e2'
MEMBERS = {
    '3dfront': 'M3DLayout_json/3dfront_train.json',
    'infinigen': 'M3DLayout_json/infinigen_train_merged.json',
    'mp3d': 'M3DLayout_json/mp3d_train.json',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def geometry_valid(obj):
    try:
        return (isinstance(obj['category'], str) and bool(obj['category'])
                and len(obj['location']) == len(obj['size']) == 3
                and all(math.isfinite(float(v)) for v in obj['location'] + obj['size'])
                and all(float(v) > 0 for v in obj['size'])
                and math.isfinite(float(obj['rotation'])))
    except (KeyError, TypeError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--local-output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    print('Verify fixed archive; inspect exactly three train members; stage up to 12 per source.', flush=True)
    archive_hash = digest(a.archive.read_bytes())
    if archive_hash != EXPECTED_SHA:
        raise ValueError('Archive SHA256 differs from official LFS oid')
    a.local_output.mkdir(parents=True, exist_ok=False)
    a.report.mkdir(parents=True, exist_ok=False)
    reports, staged = [], []
    with zipfile.ZipFile(a.archive) as z:
        members = [dict(name=i.filename, bytes=i.file_size,
                        payload_opened=i.filename in MEMBERS.values()) for i in z.infolist()]
        for source, member in MEMBERS.items():
            if z.getinfo(member).file_size > 200000000:
                raise ValueError('Train member exceeds inspection memory limit')
            raw = z.read(member)  # CRC verified for opened train payload only.
            data = json.loads(raw)
            scenes = data['scenes']
            ids = [s['scene_id'] for s in scenes]
            id_counts = Counter(ids)
            duplicate_hashes = {}
            for scene in scenes:
                if id_counts[scene['scene_id']] > 1:
                    duplicate_hashes.setdefault(scene['scene_id'], set()).add(digest(canonical(scene)))
            categories, fields = Counter(), Counter()
            invalid, missing_assets, object_count, described, shells = 0, 0, 0, 0, 0
            valid = []
            for scene in scenes:
                described += bool(scene.get('description'))
                shells += any(k in scene for k in ('floor_plan', 'room_polygon', 'walls', 'doors'))
                ok = True
                for obj in scene['objects']:
                    object_count += 1
                    fields.update(obj.keys())
                    categories.update([obj.get('category', '<missing>')])
                    good = geometry_valid(obj)
                    invalid += not good
                    ok = ok and good
                    missing_assets += not any(obj.get(k) for k in ('asset_id', 'model_id', 'jid', 'mesh_path'))
                if ok and len(scene['objects']) >= 2 and id_counts[scene['scene_id']] == 1:
                    valid.append(scene)
            ordered = sorted(valid, key=lambda s: digest((source + '/' + s['scene_id']).encode()))[:12]
            for index, scene in enumerate(ordered):
                record = {
                    'source': source, 'official_split': 'train', 'source_member': member,
                    'scene_id': scene['scene_id'], 'source_record_sha256': digest(canonical(scene)),
                    'provenance': 'procedurally_generated_layout' if source == 'infinigen' else
                                  'professional_CAD_layout_without_decision_rationale' if source == '3dfront' else
                                  'real_world_scan_layout_without_resident_brief',
                    'allocation': 'development_intake_exposed_not_training_admitted',
                    'source_layout': scene,
                    'coordinate_convention': {'vertical_axis': 'y', 'ground_plane': 'xz',
                                              'size': 'half_extents', 'rotation': 'radians'},
                    'resident_brief': None, 'professional_reason': None,
                    'native_asset_bindings': None, 'native_actions': None,
                    'room_shell': None, 'whole_room_usable': None,
                    'description_role': 'completed_layout_description_not_pre_design_need',
                    'training_admitted': False,
                }
                local = a.local_output / f'{source}_{index:02d}.json'
                local.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
                staged.append(dict(source=source, scene_id=scene['scene_id'], local_path=str(local),
                                   sha256=digest(local.read_bytes()), source_record_sha256=record['source_record_sha256'],
                                   objects=len(scene['objects'])))
            report = dict(source=source, member=member, member_sha256=digest(raw),
                          scenes=len(scenes), unique_scene_ids=len(id_counts),
                          repeated_id_groups=len(duplicate_hashes),
                          identical_record_duplicate_groups=sum(len(v) == 1 for v in duplicate_hashes.values()),
                          differing_record_duplicate_groups=sum(len(v) > 1 for v in duplicate_hashes.values()),
                          declared_scene_count=data.get('scene_count'),
                          objects=object_count, invalid_geometry_objects=invalid,
                          scenes_with_nonempty_description=described,
                          scenes_with_explicit_shell_fields=shells,
                          objects_without_asset_reference=missing_assets,
                          object_fields=dict(sorted(fields.items())),
                          category_counts=dict(sorted(categories.items())), staged=len(ordered))
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k not in ('category_counts','object_fields')}), flush=True)
    result = dict(archive=str(a.archive), archive_sha256=archive_hash, archive_members=members,
                  train_splits=reports, staged_records=staged,
                  totals={k:sum(r[k] for r in reports) for k in ('scenes','objects','staged','invalid_geometry_objects')},
                  held_out_payloads_opened=0, native_actions=0, model_calls=0, training_updates=0,
                  new_professional_decision_records=0,
                  scope='Train-file statistics and 36 deterministic local source-layout records; no executable Scene-Designer samples.')
    (a.report / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['totals']), flush=True)


if __name__ == '__main__':
    main()
