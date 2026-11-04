"""Read two generated references; never export them as professional actions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(datetime.now(timezone.utc).isoformat(), 'begin source-decision reference audit', flush=True)
    protocol = json.loads(a.protocol.read_text())
    for item in protocol['inputs'].values():
        if digest(Path(item['path'])) != item['sha256']:
            raise ValueError('Input hash mismatch: ' + item['path'])
    folders = json.loads(Path(protocol['inputs']['embedded']['path']).read_text())['folders']
    lock = json.loads(Path(protocol['inputs']['assets']['path']).read_text())
    selected = [r for r in lock['rows'] if r.get('selected_records')]
    verified = []
    for row in selected:
        path = Path(protocol['asset_root']) / row['path']
        if digest(path) != row['sha256']:
            raise ValueError('Local asset hash mismatch: ' + row['id'])
        verified.append(row['id'])
    a.out.mkdir(parents=True, exist_ok=False)
    cases = []
    inventory = []
    for case in protocol['cases']:
        files = folders[case['reference_key']]['files']
        initial, final = files['initial.json'], files['final.json']
        def objects(state):
            result = {obj['name']: obj for obj in state['furniture']}
            if len(result) != len(state['furniture']):
                raise ValueError('Duplicate furniture name in ' + case['reference_key'])
            return result
        before, after = objects(initial), objects(final)
        name = case['target_name']
        old, new = before[name], after[name]
        vector = [float(b) - float(a) for a, b in zip(old['location'], new['location'])]
        rotation = ((new.get('facing', 0) - old.get('facing', 0) + 180) % 360) - 180
        categories = set(case['target_asset_categories'])
        candidates = [r for r in lock['rows'] if categories.intersection(r['categories'])]
        available = [r for r in candidates if r['id'] in verified]
        for obj in initial['furniture']:
            inventory.append({'decision_id': case['decision_id'], 'reference_name': obj['name'],
                              'source_type': 'upstream_generated_proxy_not_survey',
                              'proxy_record': obj, 'native_object_id': None, 'native_asset_id': None,
                              'support_relation_verified': False})
        cases.append({
            'decision_id': case['decision_id'], 'reference_key': case['reference_key'],
            'input_schema': 'Function2Scene generated layout, not native SceneSmith',
            'initial_snapshot_canonical_sha256': canonical(initial),
            'final_snapshot_canonical_sha256': canonical(final),
            'initial_furniture_count': len(before), 'final_furniture_count': len(after),
            'new_names_in_final': sorted(after.keys() - before.keys()),
            'missing_names_in_final': sorted(before.keys() - after.keys()),
            'target': {
                'reference_name': name,
                'initial_record_canonical_sha256': canonical(old),
                'final_record_canonical_sha256': canonical(new),
                'displacement_m': math.sqrt(sum(v * v for v in vector)),
                'delta_xyz_y_up_m': vector, 'facing_delta_degrees': rotation,
                'dimensions_changed': old['dimensions'] != new['dimensions'],
                'color_changed': old.get('color') != new.get('color'),
                'native_id_present_in_either_snapshot': any(
                    any(k in ob for k in ('object_id', 'asset_id', 'instance_id')) for ob in (old, new)),
                'name_match_proves_instance_preservation': False,
            },
            'asset_mapping': {
                'category_filter': sorted(categories),
                'locked_catalog_candidates': len(candidates),
                'hash_verified_local_candidates': len(available),
                'local_candidate_ids': [r['id'] for r in available],
                'category_match_is_semantic_or_size_acceptance': False,
                'selected_asset_id': None,
            },
            'authentic_tool_trace_present': False,
            'activity_transition_states_identified': False,
            'professional_action_target_admitted': False,
        })
    report = {'schema': 'scenefunc-decision-reference-audit-v1',
              'protocol_sha256': digest(a.protocol), 'cases': cases,
              'completed': {'cases': len(cases), 'initial_furniture_records': len(inventory),
                            'local_asset_hashes': len(verified)},
              'model_calls': 0, 'engine_edits': 0, 'training_updates': 0,
              'qualified_professional_samples': 0,
              'interpretation': 'Generated layout deltas are not sourced professional action traces. '
                                'A declared reconstruction can assign proxy IDs without historical UUIDs; '
                                'it still needs source-to-action review and actual execution evidence.'}
    write(a.out / 'RESULT.json', report)
    write(a.out / 'INITIAL_PROXY_INVENTORY.json', inventory)
    print(json.dumps(report['completed']), flush=True)


if __name__ == '__main__':
    main()
