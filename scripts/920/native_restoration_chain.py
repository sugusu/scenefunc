"""Validate explicit non-tool native restoration bridges, without rewriting hashes."""
import math
import re


def validate_chain(history, transitions, current_state_sha256):
    def sha(value):
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('Invalid restoration state hash')
    previous_step = -1
    for event in transitions:
        if set(event) != {'after_step', 'reason', 'before_sha256', 'after_sha256',
                          'source_manifest_sha256', 'differences'}:
            raise ValueError('Unexpected restoration event fields')
        if (type(event['after_step']) is not int or event['after_step'] < previous_step
                or event['reason'] != 'native_quaternion_roundtrip'):
            raise ValueError('Invalid restoration event order or reason')
        previous_step = event['after_step']
        for key in ('before_sha256', 'after_sha256', 'source_manifest_sha256'):
            sha(event[key])
        if event['before_sha256'] == event['after_sha256'] or not event['differences']:
            raise ValueError('Restoration must record an actual numerical difference')
        seen = set()
        for diff in event['differences']:
            if set(diff) != {'path', 'before', 'after'}:
                raise ValueError('Unexpected numerical difference fields')
            path = diff['path']
            if (not isinstance(path, list) or len(path) < 3
                    or path[-3:-1] != ['transform', 'rotation_wxyz']
                    or type(path[-1]) is not int or not 0 <= path[-1] < 4
                    or repr(path) in seen):
                raise ValueError('Restoration changed a non-quaternion field')
            seen.add(repr(path))
            a, b = diff['before'], diff['after']
            if (type(a) is not float or type(b) is not float
                    or not math.isfinite(a) or not math.isfinite(b)
                    or not 0 < abs(a-b) <= 1e-12):
                raise ValueError('Restoration exceeds native roundoff allowance')
    if current_state_sha256 is None:
        raise ValueError('Current state binding required')
    sha(current_state_sha256)
    if not history:
        if transitions:
            raise ValueError('Restoration history needs an initial recorded state')
        return
    cursor = 0
    def bridge(start, end):
        nonlocal cursor
        while start != end:
            if cursor >= len(transitions):
                raise ValueError('History is not bound to the restored/current state')
            event = transitions[cursor]
            if event['before_sha256'] != start:
                raise ValueError('Discontinuous restoration bridge')
            start = event['after_sha256']
            cursor += 1
    for old, new in zip(history, history[1:]):
        if new['attempt'] != old['attempt'] + 1:
            raise ValueError('Incomplete history around native restoration')
        bridge(old['after_sha256'], new['before_sha256'])
    bridge(history[-1]['after_sha256'], current_state_sha256)
    if cursor != len(transitions):
        raise ValueError('Unused or duplicated restoration transition')
