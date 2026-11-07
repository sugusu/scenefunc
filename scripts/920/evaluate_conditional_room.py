"""Read-only condition-aware audit of a newly produced portable checkpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from conditional_room_metrics import measure_endpoint
from native_checkpoint import canonical_hash
from native_project import read_json, sha256, validate_bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan = read_json(a.plan)
    dependencies = {'evaluate_conditional_room.py', 'conditional_room_metrics.py',
        'build_activity_completion_reviews.py', 'inspect_native_mesh_contacts.py',
        'measure_free_floor.py', 'measure_room_access.py', 'preflight_sourced_trajectory.py',
        'native_project.py', 'native_checkpoint.py'}
    if not dependencies <= plan['code_sha256'].keys():
        raise ValueError('Incomplete frozen evaluation dependencies')
    for name, digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise ValueError('Frozen evaluation code changed: '+name)
    initial_root = Path(plan['initial_bundle'])
    if sha256(initial_root/'manifest.json') != plan['initial_manifest_sha256']:
        raise ValueError('Frozen initial bundle changed')
    im, initial, _, _ = validate_bundle(initial_root)
    root = a.bundle.resolve(); manifest, state, _, _ = validate_bundle(root)
    rid = manifest['selected_room_id']
    if rid != im['selected_room_id'] or manifest['project'] != im['project']:
        raise ValueError('Endpoint belongs to another room/project')
    print(datetime.now(timezone.utc).isoformat(), 'measure actual checkpoint; no model or edit', flush=True)
    native = state['rooms'][rid]
    result = measure_endpoint(native, initial['rooms'][rid], root/('room_'+rid), root,
        plan['program'], plan['requested_mode'])
    result.update(status='completed_native_checkpoint_measurement', evaluated_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha256(a.plan), input_manifest_sha256=sha256(root/'manifest.json'),
        state_sha256=canonical_hash(native), initial_state_sha256=canonical_hash(initial['rooms'][rid]),
        metric_version='conditional-endpoint-v1', model_calls=0, scene_edits=0, training_updates=0,
        limits=plan['unknowns'])
    with a.out.open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['review']['measured_requirements_status'],
        requirement_count=len(result['review']['checks']))), flush=True)


if __name__ == '__main__':
    main()
