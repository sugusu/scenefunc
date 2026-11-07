"""Read-only live-checkpoint evaluator for the isolated nesting-task family."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from native_checkpoint import canonical_hash
from native_project import read_json, sha256, validate_bundle
from nesting_room_metrics import measure_nesting


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan = read_json(a.plan)
    required = {'evaluate_nesting_room.py','nesting_room_metrics.py','conditional_room_metrics.py',
        'build_activity_completion_reviews.py','inspect_native_mesh_contacts.py','measure_free_floor.py',
        'measure_room_access.py','preflight_sourced_trajectory.py','native_project.py','native_checkpoint.py'}
    if not required <= plan['code_sha256'].keys(): raise ValueError('Missing evaluator dependency hashes')
    for name, digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name) != digest: raise ValueError('Evaluation code changed: '+name)
    initial_root = Path(plan['initial_bundle'])
    if sha256(initial_root/'manifest.json') != plan['initial_manifest_sha256']:
        raise ValueError('Frozen initial bundle changed')
    im, ih, _, _ = validate_bundle(initial_root)
    root = a.bundle.resolve(); manifest, house, _, _ = validate_bundle(root)
    rid = manifest['selected_room_id']
    if rid != im['selected_room_id'] or manifest['project'] != im['project']:
        raise ValueError('Endpoint belongs to another room/project')
    state, initial = house['rooms'][rid], ih['rooms'][rid]
    result = measure_nesting(state, initial, root/('room_'+rid), root, plan['program'], plan['requested_mode'])
    result.update(status='completed_native_checkpoint_measurement', evaluated_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha256(a.plan), input_manifest_sha256=sha256(root/'manifest.json'),
        state_sha256=canonical_hash(state), initial_state_sha256=canonical_hash(initial),
        metric_version='nesting-endpoint-v1', limits=plan['unknowns'], model_calls=0, scene_edits=0, training_updates=0)
    with a.out.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(activity=result['review']['requested_activity_status'],
        all_requirements=result['review']['measured_requirements_status'],requirements=len(result['review']['checks']))),flush=True)


if __name__ == '__main__': main()
