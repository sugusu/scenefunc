"""Independently check the finite condition matrix against pinned source inputs."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
from pathlib import Path

from audit_material_routing import CORPUS, GitInputs, write
from audit_condition_coverage import object_signature
from audit_declared_stop_gate import decide, extract_payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.audit.read_text())
    src = GitInputs(args.repo)
    for path, expected in data['source_sha256'].items():
        assert hashlib.sha256(src.blob(path)).hexdigest() == expected, path
    corpus = src.json(CORPUS)
    assert [u['id'] for u in corpus['rows']] == [r['unit_id'] for r in data['rows']]
    original = {}
    for unit, row in zip(corpus['rows'], data['rows']):
        inp, _ = src.reference(unit['input'])
        request, _ = src.reference(unit['request'])
        objs = inp['context']['objects']
        if isinstance(objs, dict):
            objs = list(objs.values())
        original[unit['id']] = {o['object_id']: {
            key: o.get(key) for key in ('object_type', 'bbox_min', 'bbox_max', 'transform')}
            for o in objs}
        assert row['object_table_sha256'] is not None
        assert decide(extract_payload(request)) == row['gate']
    pairs = []
    rows = data['rows']
    for a, b in itertools.combinations(rows, 2):
        if (a['family_id'] == b['family_id'] and a['task_value'] is not None
                and b['task_value'] is not None and a['task_value'] != b['task_value']
                and original[a['unit_id']] == original[b['unit_id']]):
            pairs.append([a['unit_id'], b['unit_id']])
    assert pairs == [p['ids'] for p in data['same_geometry_pairs']]
    supported = [r for r in rows if r['gate_agrees'] is not None]
    for name, control in data['controls'].items():
        if name == 'ordinary_declared_gate':
            continue
        cells = defaultdict(Counter)
        for r in supported:
            cells[tuple(r[k] for k in control['fields'])][r['edit_stop_reference']] += 1
        assert control['correct_at_most'] == sum(max(c.values()) for c in cells.values())
    # The first implementation failed on dict-form object tables. Check this
    # actual representation boundary, missing-state handling, identity and
    # tiny transform changes; no tolerance may create a fake matched pair.
    example = {'object_id': 'x', 'object_type': 'furniture', 'bbox_min': [0, 0, 0],
               'bbox_max': [1, 1, 1], 'transform': {'translation': [0, 0, 0]}}
    listed = {'objects': [example]}
    mapped = {'objects': {'x': example}}
    assert object_signature(listed) == object_signature(mapped)
    assert object_signature({}) is None
    shifted = copy.deepcopy(listed)
    shifted['objects'][0]['transform']['translation'][0] = 1e-9
    assert object_signature(shifted) != object_signature(listed)
    renamed = copy.deepcopy(listed)
    renamed['objects'][0]['object_id'] = 'another_instance'
    assert object_signature(renamed) != object_signature(listed)
    out = {'status': 'passed', 'source_files_rehashed': len(data['source_sha256']),
           'source_units_checked': len(rows), 'object_table_pairs_checked': pairs,
           'task_state_oracles_independently_recounted': True,
           'ordinary_rule_recomputed_from_original_requests': True,
           'schema_identity_and_exact_transform_checks': 4,
           'new_model_calls': 0, 'new_scene_edits': 0, 'training_updates': 0,
           'audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(),
           'scope': 'Document/hash and finite-record verification, not a native physics replay or model evaluation.'}
    write(args.output, out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
