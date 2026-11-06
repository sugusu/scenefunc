"""Check evidence integrity and the new finite-ray primitive; no model or edit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh

p = Path(__file__).resolve().parent
repo = p.parents[2]
sys.path.insert(0, str(repo/'scripts/920'))
from storage_partition_metrics import segment_hits

read = lambda f: json.loads(Path(f).read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
mesh = trimesh.creation.box(extents=[.2, 2., 2.])
ray_checks = {
    'crosses_actual_box': bool(segment_hits(mesh, np.array([-2., 0., 0.]), np.array([2., 0., 0.]))),
    'finite_endpoint_before_box': not segment_hits(mesh, np.array([-2., 0., 0.]), np.array([-1., 0., 0.])),
    'misses_box': not segment_hits(mesh, np.array([-2., 3., 0.]), np.array([2., 3., 0.])),
}
assert all(ray_checks.values())
copies = read(p/'COPY_MANIFEST.json')['files']
for x in copies:
    assert sha(p/x['destination']) == x['sha256'] == sha(x['source'])
    assert (p/x['destination']).stat().st_size == x['bytes']
steps = read(p/'evidence/native/STEPS.json')
for i in range(4):
    binding = read(p/f'evidence/bindings/step_{i:03d}.state_binding.json')
    state = read(p/('evidence/native/initial_state.json' if i == 0 else f'evidence/native/step_{i:02d}_state.json'))
    assert binding['state_sha256'] == canonical(state) == canonical(binding['native_state'])
    if i:
        assert steps[i-1]['after_sha256'] == canonical(state)
    for image in binding['images']:
        assert sha(image['path']) == image['sha256']
        with Image.open(image['path']) as im:
            im.verify()
for n in ['INITIAL_AUDIT.json', 'CANDIDATE_AUDIT.json']:
    audit = read(p/n); rays = audit['activity']['visibility']['rays']
    assert len(rays) == 27
    assert sum(x['clear'] for x in rays) == (27 if n.startswith('INITIAL') else 18)
    assert len(audit['inventory']) == 8
    assert all(audit['immutable_records_unchanged'].values())
    for scenario in audit['activity']['scenarios']:
        assert len(scenario['role_access']) == 9 and all(scenario['role_access'].values())
        assert scenario['two_dining_place_proxy_pass'] and scenario['all_required_ids_preserved']
    for ray in rays:
        length = np.linalg.norm(np.array(ray['target_m'])-ray['origin_m'])
        for blocker in ray['blockers']:
            assert 0 < blocker['first_hit']['distance_m'] < length
hashes = 0
for planfile in p.rglob('*.json'):
    d = read(planfile)
    if not isinstance(d, dict):
        continue
    for name, digest in d.get('code_sha256', d.get('code_hashes', {})).items():
        path = Path(name)
        if not path.is_absolute():
            path = repo/path if '/' in name else repo/'scripts/920'/name
        assert sha(path) == digest, str(planfile)+': '+name
        hashes += 1
assert read(p/'FEASIBILITY_GATE.json')['model_rollout_allowed'] is False
assert all(read(p/'evaluation'/c/'STATUS.json')['model_calls'] == 0 for c in ['screen', 'open'])
assert read(p/'evidence/restore/RESULT.json')['quaternion_roundtrip_differences'] == []
result = dict(status='passed_evidence_checks_not_design_acceptance', checked_at=datetime.now(timezone.utc).isoformat(),
              finite_ray_checks=ray_checks, archive_copies_checked=len(copies), code_hash_references_checked=hashes,
              state_image_bindings=4, images_checked=20, actual_model_calls=0, new_scene_edits=0,
              caveat='No professional evaluation or new live model loop test; frozen reference still fails.')
with (p/'VALIDATION.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps(result), flush=True)
