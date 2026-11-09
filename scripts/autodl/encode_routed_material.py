"""Run actual five-view CPU encoding and archive target/mask receipts.

Optional tensor files go to a separate cache. No optimizer or model is loaded.
All original 64 components remain in the result, including blocked ones.
"""
import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

from audit_material_routing import PIN, digest, token_hash, write
from routed_multimodal import (resolve_messages, encode_component, tensor_digest,
                               require_complete_material, frame_completed_decision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--processor', type=Path, required=True)
    ap.add_argument('--scene-observation', type=Path, required=True)
    ap.add_argument('--availability', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tensor-cache', type=Path, required=True)
    args = ap.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    args.tensor_cache.mkdir(parents=True, exist_ok=False)
    base = args.repo / 'runs/autodl/20261110_material_routing/actual'
    components = [json.loads(line) for line in (base / 'prospective_components.jsonl').read_text().splitlines()]
    available = json.loads(args.availability.read_text())
    views = {r['unit_id']: r for r in available['rows']}
    old = json.loads((args.repo / 'runs/autodl/20261110_objective_source_review/material_manifest.json').read_text())
    old = {(r['unit_id'], r['component']): r for r in old['records'] if r['arm'] == 'separate'}
    source_hash = digest(args.scene_observation.read_bytes())
    if source_hash != '1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3':
        raise ValueError('Official image budget source differs from the frozen training plan')
    spec = importlib.util.spec_from_file_location('image_contract', args.scene_observation)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    import torch
    import transformers
    from transformers import AutoProcessor
    from PIL import Image
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True, trust_remote_code=False)
    official.configure_image_budget(processor, 65536)
    local_sources = {str(p): digest(p.read_bytes()) for p in args.processor.iterdir() if p.suffix == '.json'}
    records = []
    image_receipts = {}
    # The original D material includes a separator newline after </think>.
    # A host stopping at that closing tag must add the same nonsemantic framing.
    framing = []
    for c in components:
        if c['component'] == 'decision' and c['arm'] == 'paired_tagged':
            framed = frame_completed_decision(c['target_text'].rstrip())
            if framed != c['target_text']:
                raise ValueError('Host delimiter framing changes a frozen D target')
            framing.append({'unit_id': c['unit_id'], 'closing_tag_then_host_newline_exact': True})
    # Archived full-output parsing is an interface audit, not a new rollout.
    rollout_root = 'runs/920/P920-MATCHEDROLL-v1_20261108T062940Z'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PIN, rollout_root], cwd=args.repo, text=True).splitlines()
    old_output_checks = []
    for path in names:
        if '/recorded/round_' not in path or not path.endswith('/model/raw_output.txt'):
            continue
        raw = subprocess.check_output(['git', 'show', f'{PIN}:{path}'], cwd=args.repo)
        text = raw.decode()
        end = text.find('</think>')
        check = {'path': path, 'sha256': digest(raw)}
        try:
            if end < 0:
                raise ValueError('No complete decision prefix in recorded output')
            # Same boundary that a future stop-sequence host would observe.
            observed_prefix = text[:end + len('</think>')]
            framed = frame_completed_decision(observed_prefix)
            check.update(status='completed_prefix_framed', host_separator='one_newline',
                         framed_token_count=len(processor.tokenizer.encode(framed, add_special_tokens=False)),
                         suffix_not_replayed=True)
        except ValueError as exc:
            check.update(status='unusable_decision_prefix', reason=str(exc))
        old_output_checks.append(check)
    for c in components:
        row = {'unit_id': c['unit_id'], 'arm': c['arm'], 'component': c['component'],
               'target_tokens': len(c['target_token_ids']), 'target_sha256': c['target_sha256'],
               'per_token_objective_coefficient': c['target_component_weight']}
        if not views[c['unit_id']]['complete_five_views']:
            row.update(status='missing_original_views', missing_views=[i['path'] for i in views[c['unit_id']]['images'] if not i['match']])
            records.append(row)
            continue
        tick = time.perf_counter()
        try:
            messages, paths = resolve_messages(c['messages'], views[c['unit_id']]['images'])
            images = []
            for p in paths:
                with Image.open(p) as im:
                    images.append(im.convert('RGB'))
            batch, boundary = encode_component(processor, c, messages, images)
            hashes = {k: tensor_digest(v) for k, v in batch.items()}
            archive = old[c['unit_id'], c['component']]['transformed_sha256']
            visual_exact = all(hashes[k] == archive[k] for k in ('pixel_values', 'image_grid_thw'))
            # Keep differences as evidence, rather than weakening the hash check.
            if not visual_exact:
                row.update(status='image_tensor_mismatch', actual_hashes=hashes,
                           expected_image_hashes={k: archive[k] for k in ('pixel_values', 'image_grid_thw')})
                records.append(row)
                continue
            if token_hash(batch['labels'][batch['labels'] != -100].tolist()) != c['target_sha256']:
                raise ValueError('Label token hash changed')
            unit_cache = args.tensor_cache / c['unit_id']
            unit_cache.mkdir(exist_ok=True)
            visual_file = unit_cache / 'vision.pt'
            if not visual_file.exists():
                torch.save({k: batch[k] for k in ('pixel_values', 'image_grid_thw')}, visual_file)
            seq_file = unit_cache / (c['arm'] + '.' + c['component'] + '.pt')
            torch.save({k: v for k, v in batch.items() if k not in ('pixel_values', 'image_grid_thw')}, seq_file)
            # A cold tensor reload must reproduce every field, not merely shapes.
            restored = {**torch.load(visual_file, weights_only=True), **torch.load(seq_file, weights_only=True)}
            if {k: tensor_digest(v) for k, v in restored.items()} != hashes:
                raise ValueError('Serialized component reload differs')
            prefix_len = c['assistant_prefix_tokens']
            if prefix_len and bool((batch['labels'][:, boundary - prefix_len:boundary] != -100).any()):
                raise ValueError('Paired A incorrectly supervises its decision prefix')
            row.update(status='encoded', boundary=boundary, sequence_tokens=int(batch['input_ids'].shape[1]),
                       shapes={k: list(v.shape) for k, v in batch.items()}, tensor_sha256=hashes,
                       original_image_tensor_hashes_exact=True, all_prefix_labels_masked=True,
                       decision_prefix_tokens=prefix_len, decision_prefix_supervised=False,
                       training_reference_prefix_used=(prefix_len > 0), target_shift_min_index=boundary - 1,
                       sequence_file=str(seq_file.resolve()), sequence_file_sha256=digest(seq_file.read_bytes()),
                       vision_file=str(visual_file.resolve()), vision_file_sha256=digest(visual_file.read_bytes()),
                       wall_seconds=time.perf_counter() - tick)
            image_receipts[c['unit_id']] = {k: hashes[k] for k in ('pixel_values', 'image_grid_thw')}
            print(c['unit_id'], c['arm'], c['component'], 'encoded', row['sequence_tokens'], flush=True)
        except Exception as exc:
            row.update(status='encoding_error', error=type(exc).__name__ + ': ' + str(exc), wall_seconds=time.perf_counter() - tick)
        records.append(row)
    try:
        require_complete_material(records)
        complete = True
    except ValueError as exc:
        complete = False
        block = str(exc)
    result = {'source_components': len(components), 'encoded_components': sum(r['status'] == 'encoded' for r in records),
              'source_units': len(views), 'encoded_units': len(image_receipts),
              'statuses': {s: sum(r['status'] == s for r in records) for s in sorted({r['status'] for r in records})},
              'records': records, 'all_fixed_material_encoded': complete,
              'frozen_decision_framing': framing, 'archived_output_prefix_audit': old_output_checks,
              'archived_output_audit_scope': 'Boundary handling only. No new model response, rollout, or decision accuracy is inferred. Downstream archived responses cannot be spliced after a changed protocol.',
              'training_ready': False, 'readiness_note': 'Encoding is necessary, not sufficient. No new trainer/model run or deployment inference was performed.',
              'full_corpus_guard': 'passed' if complete else block,
              'max_pixels': 65536, 'max_sequence_tokens': 16384,
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'transformers', 'tokenizers', 'Pillow')},
              'processor_class': type(processor).__name__, 'image_processor_class': type(processor.image_processor).__name__,
              'processor_configuration': processor.image_processor.to_dict(),
              'local_processor_hashes': local_sources, 'scene_observation_sha256': source_hash,
              'availability_sha256': digest(args.availability.read_bytes()),
              'components_sha256': digest((base / 'prospective_components.jsonl').read_bytes()),
              'wall_seconds': time.perf_counter() - start, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'model_calls': 0, 'training_updates': 0, 'cuda_initialized': torch.cuda.is_initialized()}
    write(args.output / 'encoding.json', result)
    print(json.dumps({k: result[k] for k in ('encoded_components','source_components','encoded_units','statuses','wall_seconds','peak_rss_kib','cuda_initialized')}))


if __name__ == '__main__':
    main()
