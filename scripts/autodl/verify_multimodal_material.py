"""Independent reload of encoded tensors and frozen original target hashes."""
import hashlib
import json
from pathlib import Path
import sys

from audit_material_routing import token_hash, write
from routed_multimodal import tensor_digest, require_complete_material


def main():
    root = Path('runs/autodl/20261110_multimodal_material')
    manifest = json.loads((root / 'manifest.json').read_text())
    checked_files = 0
    for files, prefix in ((manifest['source_sha256'], Path('.')), (manifest['output_sha256'], root)):
        for path, expected in files.items():
            assert hashlib.sha256((prefix / path).read_bytes()).hexdigest() == expected, path
            checked_files += 1
    result = json.loads((root / 'encoded/encoding.json').read_text())
    components = [json.loads(line) for line in Path('runs/autodl/20261110_material_routing/actual/prospective_components.jsonl').read_text().splitlines()]
    indexed = {(r['unit_id'], r['arm'], r['component']): r for r in components}
    assert len(result['records']) == len(indexed) == 64
    availability = json.loads((root / 'source_image_lookup/image_availability.json').read_text())
    views = {r['unit_id']: r for r in availability['rows']}
    from collections import Counter
    counts = Counter(r['status'] for r in result['records'])
    assert counts == result['statuses']
    import torch
    torch.set_num_threads(1)
    material = {}
    for row in result['records']:
        key = row['unit_id'], row['arm'], row['component']
        source = indexed[key]
        if row['status'] != 'encoded':
            assert row['status'] == 'missing_original_views'
            assert not views[row['unit_id']]['complete_five_views']
            continue
        assert views[row['unit_id']]['complete_five_views']
        for field in ('sequence_file', 'vision_file'):
            assert hashlib.sha256(Path(row[field]).read_bytes()).hexdigest() == row[field + '_sha256']
        batch = {**torch.load(row['vision_file'], weights_only=True), **torch.load(row['sequence_file'], weights_only=True)}
        assert {k: tensor_digest(v) for k, v in batch.items()} == row['tensor_sha256']
        labels = batch['labels']
        assert (labels[:, :row['boundary']] == -100).all()
        assert labels[labels != -100].tolist() == source['target_token_ids']
        assert token_hash(labels[labels != -100].tolist()) == source['target_sha256']
        assert torch.equal(batch['action_mask'] | batch['decision_mask'], labels != -100)
        assert not (batch['action_mask'] & batch['decision_mask']).any()
        assert labels.shape[1] <= 16384
        assert len(batch['image_grid_thw']) == 5
        for view in views[row['unit_id']]['images']:
            assert hashlib.sha256(Path(view['match']['local_path']).read_bytes()).hexdigest() == view['sha256']
        pairkey = row['unit_id'], row['component']
        target = labels[labels != -100].tolist()
        prior = material.setdefault(pairkey, target)
        assert prior == target
    incomplete_rejected = False
    try:
        require_complete_material(result['records'])
    except ValueError:
        incomplete_rejected = True
    assert incomplete_rejected and not result['all_fixed_material_encoded']
    assert result['model_calls'] == result['training_updates'] == 0
    assert not result['cuda_initialized'] and not torch.cuda.is_initialized()
    assert len(result['frozen_decision_framing']) == 8
    assert len(result['archived_output_prefix_audit']) == 35
    write(root / 'verification_receipt.json', {
        'status': 'passed', 'checked_manifest_files': checked_files,
        'independent_component_reloads': counts['encoded'],
        'fixed_components_in_denominator': len(indexed),
        'missing_components_preserved': counts['missing_original_views'],
        'full_corpus_guard_rejected_partial_material': incomplete_rejected,
        'model_calls': 0, 'training_updates': 0, 'cuda_initialized': False,
        'not_a_new_scene_or_training_result': True})
    print((root / 'verification_receipt.json').read_text())


if __name__ == '__main__':
    main()
