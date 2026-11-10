"""Restore a recorded pose state against its unchanged asset/room bundle.

This is observation reconstruction, not a proposed edit or a model decision.
The independent state ledger must bind the requested unit to the exact state.
"""
import copy
import hashlib
import json
from pathlib import Path


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def apply_recorded_state(house, room_id, *, ledger, unit):
    ledger = Path(ledger).resolve()
    records = json.loads(ledger.read_text())
    matches = [r for r in records if r['unit_id'] == unit]
    if len(matches) != 1 or matches[0]['status'] != 'resolved':
        raise ValueError('Exactly one resolved frozen state is required')
    record = matches[0]
    path = Path(record['local_state']).resolve()
    state = json.loads(path.read_text())
    if canonical(state) != record['state_canonical_sha256']:
        raise ValueError('Frozen state does not match its independent binding')
    before = house['rooms'][room_id]
    a, b = copy.deepcopy(before), copy.deepcopy(state)
    if set(a['objects']) != set(b['objects']):
        raise ValueError('This recovered asset bundle does not cover changed object IDs')
    changed = []
    for oid in a['objects']:
        pa = a['objects'][oid].pop('transform')
        pb = b['objects'][oid].pop('transform')
        if pa != pb:
            changed.append({'object_id': oid, 'bundle_pose': pa, 'recorded_pose': pb})
    if a != b:
        raise ValueError('Non-pose state differs; asset/metadata equivalence not established')
    out = copy.deepcopy(house)
    out['rooms'][room_id] = state
    proof = {'unit_id': unit, 'ledger': str(ledger),
             'ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest(),
             'state_path': str(path), 'state_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'state_source': record['state_source'],
             'state_canonical_sha256': canonical(state), 'changed_from_bundle': changed,
             'asset_and_nonpose_fields_equal': True,
             'scope': 'Recorded state reconstruction only; no proposed action or quality label'}
    return out, proof
