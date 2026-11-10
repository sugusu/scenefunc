"""Exact native quaternion serialization audit; no pose tolerance or edits.

Source and runtime hashes remain distinct. Only object rotation serialization
is recomputed using the same Drake Quaternion -> RotationMatrix -> Quaternion
path used by RoomScene. Every other field must retain its exact JSON value.
"""
import copy
import hashlib
import json


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False)


def verify(source, runtime):
    from pydrake.common.eigen_geometry import Quaternion
    from pydrake.math import RotationMatrix
    expected = copy.deepcopy(source)
    differences = []
    for oid, obj in expected['objects'].items():
        original = obj['transform']['rotation_wxyz']
        serialized = RotationMatrix(Quaternion(wxyz=original)).ToQuaternion().wxyz().tolist()
        obj['transform']['rotation_wxyz'] = serialized
        if canonical(original) != canonical(serialized):
            differences.append({'object_id': oid, 'source': original,
                                'native_serialization': serialized,
                                'max_component_difference': max(abs(a-b) for a,b in zip(original, serialized))})
    if canonical(expected) != canonical(runtime):
        raise ValueError('Runtime differs from exact native quaternion roundtrip; no tolerance allowed')
    return {'status': 'exact_native_serialization', 'source_sha256': hashlib.sha256(canonical(source).encode()).hexdigest(),
            'runtime_sha256': hashlib.sha256(canonical(runtime).encode()).hexdigest(),
            'raw_json_identical': canonical(source) == canonical(runtime),
            'rotation_representation_changes': differences,
            'all_other_fields_exact': True, 'scene_edits': 0,
            'scope': 'Serialization identity, not geometric or functional quality'}
