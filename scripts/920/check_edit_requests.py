"""CPU-only candidate input migration of the completed study; never calls a model."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from jsonschema.exceptions import ValidationError

from edit_request import (from_episode_inputs, from_recorded_request,
                          normalize_context, validate_action)
from shared_action_contract import context_from_request, digest, schemas_from_observation


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() or a.runtime.exists():
        raise ValueError('Use fresh output paths; do not overwrite results')
    a.runtime.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    print(started, 'begin CPU input migration: 44 episodes and 144 recorded requests', flush=True)
    dataset = read(a.study / 'DATASET.json')
    first = read(dataset['rows'][0]['documents']['action_only']['path'])
    native = schemas_from_observation(dict(system=first['system']))
    training, deployment, failures = [], [], []
    pairs = {}
    for row in dataset['rows']:
        for arm, ref in row['documents'].items():
            try:
                if sha(ref['path']) != ref['sha256']:
                    raise ValueError('Historical episode hash changed')
                doc = read(ref['path'])
                request = from_episode_inputs(doc, native)
                source_context = json.loads(doc['task_context']['scene_description'])
                if context_from_request(request) != normalize_context(source_context):
                    raise ValueError('Lost pre-action facts')
                original_objects = source_context['objects']
                values = list(original_objects.values()) if isinstance(original_objects, dict) else original_objects
                if {v['object_id']: v for v in values} != context_from_request(request)['objects']:
                    raise ValueError('Object facts changed')
                for call in doc['turns'][0]['tool_calls']:
                    validate_action(call, request)
                # Gold actions/reasons and evaluation metadata must not affect the request.
                poisoned = copy.deepcopy(doc)
                poisoned['turns'][0]['content'] = 'GOLD_SENTINEL_DO_NOT_FEED'
                poisoned['turns'][0]['tool_calls'] = [{'name': 'GOLD_SENTINEL_DO_NOT_FEED'}]
                poisoned['evaluation_only'] = {'injected': 'GOLD_SENTINEL_DO_NOT_FEED'}
                if from_episode_inputs(poisoned, native) != request:
                    raise ValueError('Gold data leaked into input')
                target = a.runtime / 'training' / arm / row['id'] / 'request.json'
                write(target, request)
                write(target.with_name('TARGET_ONLY.json'), doc['turns'][0])
                pairs.setdefault(row['id'], []).append(digest(request))
                training.append(dict(id=row['id'], arm=arm, source_sha256=ref['sha256'],
                    request=str(target), request_sha256=sha(target), target_sha256=sha(target.with_name('TARGET_ONLY.json')),
                    source_objects_were_list=isinstance(original_objects, list),
                    schema_identity_valid=True, preaction_facts_preserved=True, gold_mutation_input_invariant=True))
            except Exception as exc:
                failures.append(dict(id=row['id'], arm=arm, error=repr(exc)))
    for path in sorted((a.study / 'evaluation').glob('*/recorded/round_*/request.json')):
        try:
            original = read(path)
            request = from_recorded_request(original)
            if context_from_request(request) != normalize_context(context_from_request(original)):
                raise ValueError('Deployment state changed')
            for old, new in zip(original[1]['content'][:5], request[1]['content'][:5]):
                if old != new:
                    raise ValueError('View identity or order changed')
            cell, round_id = path.parts[-4], path.parent.name
            target = a.runtime / 'deployment' / cell / round_id / 'request.json'
            write(target, request)
            deployment.append(dict(cell=cell, round=round_id, source_sha256=sha(path),
                request=str(target), request_sha256=sha(target), state_and_views_preserved=True))
        except Exception as exc:
            failures.append(dict(path=str(path), error=repr(exc)))
    negative_checks = []
    sample = next((x for x in training if x['id'] == 'clothing_build_07'), None)
    if sample:
        request = read(sample['request'])
        context = context_from_request(request)
        oid = next(k for k, v in context['objects'].items() if v['object_type'] == 'furniture')
        cases = [dict(name='get_current_scene_state', arguments={'dummy': 0}),
                 dict(name='add_furniture_to_scene_tool', arguments=dict(asset_id='procedural_reference:bed', x=0, y=0, yaw=0)),
                 dict(name='move_furniture_tool', arguments=dict(object_id=oid, x=0, y=0, rotation=0))]
        for call in cases:
            try:
                validate_action(call, request)
            except (ValueError, ValidationError) as exc:
                negative_checks.append(dict(call=call, rejected=True, error_type=type(exc).__name__))
            else:
                failures.append(dict(error='Known invalid call accepted', call=call))
    pair_equal = len(pairs) == 22 and all(len(x) == 2 and len(set(x)) == 1 for x in pairs.values())
    if len(training) != 44 or len(deployment) != 144 or not pair_equal or len(negative_checks) != 3:
        failures.append(dict(error='Incomplete coverage or unequal arm inputs'))
    result = dict(status='passed' if not failures else 'failed', started_at=started,
        ended_at=datetime.now(timezone.utc).isoformat(), training=training, deployment=deployment,
        counts=dict(training_episodes=len(training), unique_training_actions=len(pairs), recorded_deployment_requests=len(deployment)),
        identical_inputs_between_training_arms=pair_equal, negative_checks=negative_checks, failures=failures,
        code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['edit_request.py', 'check_edit_requests.py']},
        dataset_sha256=sha(a.study/'DATASET.json'), model_calls=0, training_updates=0, native_edits=0,
        limits=['Candidate inputs were not used for inference or training.',
                'Schema/identity checks do not establish task compliance, geometry or professional quality.',
                'Task-specific text restrictions remain verbatim; no automatic semantic enforcement claimed.',
                'Official training encoder integration and token-budget checks are not included. Historical inputs unchanged.'])
    write(a.out, result)
    print(json.dumps(dict(status=result['status'], counts=result['counts'], failures=failures), ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
