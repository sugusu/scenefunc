"""Decision-batch dry-run support for the pinned official SFT entry point.

This integration deliberately cannot train audit-only data. The ordinary
upstream training path is unchanged; qualification and authorization for a
future decision-training path have not been implemented here.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from decision_pairing import DecisionUnitCollator, DecisionUnitDataset


def validate_decision_args(args):
    if bool(args.decision_manifest) != bool(args.decision_arm):
        raise ValueError('--decision-manifest and --decision-arm must be provided together')
    if args.decision_manifest and not args.dry_run:
        raise ValueError('Decision manifests are audit-only; --dry-run is required. No training is authorized.')
    if args.decision_audit_report and not args.decision_manifest:
        raise ValueError('--decision-audit-report requires a decision manifest')
    if args.decision_manifest and (args.per_device_batch_size < 1 or args.dataloader_workers != 0):
        raise ValueError('Decision dry-run requires positive batch size and --dataloader-workers 0')


def build_decision_data(args, processor, upstream):
    dataset = DecisionUnitDataset(args.decision_manifest, args.repo_root,
                                  processor, upstream, args.decision_arm, args.max_length)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    return dataset, DecisionUnitCollator(upstream, pad_id, dataset.fixed_length)


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def inspect_decision_batch(args, dataset, collator, processor):
    """Use the actual entry-selected Dataset/Collator, without model weights."""
    started = datetime.now(timezone.utc).isoformat()
    loader = DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=False,
                        num_workers=0, collate_fn=collator)
    batch = next(iter(loader))
    count = min(len(dataset), args.per_device_batch_size)
    assert batch['input_ids'].shape[0] == 2 * count
    padding = batch['attention_mask'] == 0
    assert bool((batch['labels'][padding] == -100).all())
    assert not torch.cuda.is_initialized()
    rows = []
    for index, (ids, labels, attention) in enumerate(zip(
            batch['input_ids'], batch['labels'], batch['attention_mask'])):
        target = labels[labels != -100]
        positions = (labels != -100).nonzero().flatten()
        assert len(target) and torch.equal(target, ids[positions])
        d = dataset.decisions[index // 2]
        reason_doc = json.loads((dataset.root / d['reason']['path']).read_text())
        reason = reason_doc['turns'][0]['content']
        text = processor.tokenizer.decode(ids[attention.bool()], skip_special_tokens=False)
        rows.append({'decision_id': d['id'], 'component': 'reason' if index % 2 == 0 else 'action',
                     'input_tokens': int(attention.sum()), 'supervised_tokens': len(target),
                     'causal_shifted_targets': int((labels[1:] != -100).sum()),
                     'target_ids': target.tolist(), 'target_sha256': _tensor_hash(target),
                     'labels_sha256': _tensor_hash(labels),
                     'reason_present_in_input': reason in text,
                     'reason_present_in_targets': reason in processor.tokenizer.decode(target, skip_special_tokens=False),
                     'evaluation_only_sentinel_present': 'EVAL_ONLY_SENTINEL' in text})
    report = {'started_at': started, 'ended_at': datetime.now(timezone.utc).isoformat(),
              'status': 'passed_official_entry_decision_dry_run', 'arm': args.decision_arm,
              'manifest_sha256': hashlib.sha256(Path(args.decision_manifest).read_bytes()).hexdigest(),
              'dataset_decisions': len(dataset), 'first_batch_decisions': count,
              'first_batch_examples': 2 * count, 'rows': rows,
              'batch_shapes': {k: list(v.shape) for k, v in batch.items()},
              'batch_tensor_sha256': {k: _tensor_hash(v) for k, v in batch.items()},
              'total_supervised_tokens': sum(r['supervised_tokens'] for r in rows),
              'image_instances': len(batch['image_grid_thw']), 'cuda_initialized': False,
              'model_weights_loaded': False, 'model_forward_calls': 0, 'optimizer_updates': 0,
              'trainer_instantiated': False,
              'scope': 'Actual patched official CLI, dataset selection and DataLoader/collator only. No Trainer, model loss, optimization, FLOP parity or professional data validation claimed.'}
    if args.decision_audit_report:
        target = Path(args.decision_audit_report)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report
