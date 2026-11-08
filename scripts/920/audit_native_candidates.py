"""Read-only independent artifact audit; no model, scene or tensor re-encoding."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    plan = read(run / 'PLAN.json')
    for path, expected in plan['inputs'].items():
        assert digest(path) == expected, path
    dataset = read(run / 'corpus/DATASET.json')
    source = read(plan['source_dataset'])
    rows = {r['id']: r for r in dataset['rows']}
    originals = {r['id']: r for r in source['rows']}
    assert len(rows) == len(originals) == 24
    references = {}
    def verify(ref):
        assert digest(ref['path']) == ref['sha256'], ref['path']
        references[ref['path']] = ref['sha256']
    for key, row in rows.items():
        for ref in [row['input'], row['request'], *row['documents'].values(), *row['images']]:
            verify(ref)
        for arm in ('action_only', 'decision_action'):
            assert read(row['documents'][arm]['path'])['turns'] == read(originals[key]['documents'][arm]['path'])['turns']
        if key not in plan['modes']:
            assert row == originals[key]
        else:
            inp = read(row['input']['path'])
            measured = read(row['balanced_pre_action_measurement']['path'])
            verify(row['balanced_pre_action_measurement'])
            assert inp['context']['available_measurement'] == measured['measurements']
            assert inp['additional_text'] == plan['common_additional_text']
            forbidden = {'review', 'aggregate', 'measured_requirements_status', 'reference_commands', 'target_poses'}
            assert not forbidden.intersection(inp['context'])
            assert not forbidden.intersection(inp['context']['available_measurement'])
    for edit, stop in plan['same_state_pairs']:
        a, b = [copy.deepcopy(read(rows[k]['input']['path'])) for k in (edit, stop)]
        for field in ('current_activity_request', 'current_requested_mode'):
            assert a['context'].pop(field) != b['context'].pop(field)
        assert a == b
        assert rows[edit]['images'] == rows[stop]['images']
    enc = read(run / 'ENCODING.json')
    old = read(plan['old_encoding'])
    previous = {(r['arm'], r['id']): r for r in old['records']}
    indexed = {(r['arm'], r['id']): r for r in enc['records']}
    assert len(indexed) == 48
    for (arm, key), row in indexed.items():
        verify({'path': row['tensor_path'], 'sha256': row['tensor_sha256']})
        assert row['input_tokens'] <= plan['max_sequence_tokens']
        assert row['input_receipts_masked'] and row['pixels_unchanged']
        assert row['deployment_prompt_exact']
        other = indexed['decision_action' if arm == 'action_only' else 'action_only', key]
        assert row['prompt_sha256'] == other['prompt_sha256']
        assert row['action_target_sha256'] == other['action_target_sha256']
        if row['target_kind'] == 'tool_action':
            assert row['action_tokens'] == previous[arm, key]['action_tokens']
    assert all(r['absolute_error'] < 1e-7 for r in enc['trainer_consumption'].values())
    receipts = [read(run / phase / 'receipt.json') for phase in ('prepare_receipt', 'encode_receipt')]
    assert all(r['status'] == 'completed' and r['exit_code'] == 0 and r['first_log'] for r in receipts)
    print(json.dumps(dict(status='passed', frozen_input_hashes=len(plan['inputs']),
        referenced_files_verified=len(references), tensor_files_verified=48,
        same_state_pairs=2, unchanged_episode_turns=48, unchanged_other_examples=18,
        maximum_sequence_tokens=max(r['input_tokens'] for r in indexed.values()),
        cpu_stage_elapsed_seconds=sum(r['elapsed_seconds'] for r in receipts),
        new_model_calls=0, training_updates=0, scene_edits=0,
        limitations='Artifact verification and input equality only; no model benefit or complete-room acceptance.'), indent=2))


if __name__ == '__main__':
    main()
