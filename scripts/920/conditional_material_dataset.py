"""Equal-material views over immutable official-loader tensors; no training CLI.

Both arms learn D|I once and A once. Only paired A conditions on D. The D
component ends before the tool delimiter, so it does not teach premature EOS.
Source tensors, historical encoders and completed studies remain unchanged.
"""
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset


SEQUENCE_KEYS = {'input_ids', 'attention_mask', 'mm_token_type_ids', 'labels',
                 'action_mask', 'decision_mask'}


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def component_batch(row):
    path = Path(row['source_tensor']['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['source_tensor']['sha256']:
        raise ValueError('Source tensor changed: ' + str(path))
    batch = torch.load(path, map_location='cpu', weights_only=True)
    if not torch.equal(batch['action_mask'] | batch['decision_mask'], batch['labels'] != -100):
        raise ValueError('Source supervision coverage differs')
    if bool((batch['action_mask'] & batch['decision_mask']).any()):
        raise ValueError('Overlapping source objectives')
    if batch['input_ids'].ndim != 2 or batch['input_ids'].shape[0] != 1:
        raise ValueError('Expected one collated source sample')
    if row['component'] == 'decision':
        d = batch['decision_mask'][0].nonzero().flatten()
        a = batch['action_mask'][0].nonzero().flatten()
        if not len(d) or not len(a) or int(d[-1]) + 1 != int(a[0]):
            raise ValueError('Expected a contiguous decision preceding the action')
        if d.tolist() != list(range(int(d[0]), int(a[0]))):
            raise ValueError('Noncontiguous decision targets')
        # Causal prefix is byte-identical to joint D prediction. Remove A and
        # its EOS entirely; preserve the official image tensors/grid unchanged.
        end = int(a[0])
        batch = {k: v[:, :end].clone() if k in SEQUENCE_KEYS else v
                 for k, v in batch.items()}
        batch['action_mask'].zero_()
        batch['labels'][~batch['decision_mask']] = -100
    elif row['component'] == 'action':
        batch['labels'][~batch['action_mask']] = -100
        batch['decision_mask'].zero_()
    else:
        raise ValueError('Unknown material component')
    if not torch.equal(batch['action_mask'] | batch['decision_mask'], batch['labels'] != -100):
        raise ValueError('Transformed objective coverage differs')
    for name in ('input_ids', 'labels', 'pixel_values', 'image_grid_thw'):
        expected = row.get('transformed_sha256', {}).get(name)
        if expected is not None and tensor_digest(batch[name]) != expected:
            raise ValueError('Transformed tensor changed: ' + name)
    return batch


class EqualMaterialExamples(Dataset):
    """Compatible with EditTrainer and its full-pass family/component weighting."""
    def __init__(self, records):
        if not records or len({r['id'] for r in records}) != len(records):
            raise ValueError('Empty or duplicate components')
        if len({r['arm'] for r in records}) != 1:
            raise ValueError('Mixed arms')
        self.records = records
        self.families = sorted({r['family_id'] for r in records})
        self.counts = {f: {c: sum(r[c + '_tokens'] for r in records if r['family_id'] == f)
                          for c in ('action', 'decision')} for f in self.families}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        batch = component_batch(row)
        for c in ('action', 'decision'):
            if int(batch[c + '_mask'].sum()) != row[c + '_tokens']:
                raise ValueError('Supervised component count changed')
            count = self.counts[row['family_id']][c]
            batch[c + '_weight'] = len(self) / (len(self.families) * count) if count else 0.
        return batch
