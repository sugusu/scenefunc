"""Encode the real refreshed native host requests, without language weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

from phase_deployment import deployment_input
from routed_multimodal import tensor_digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processor', type=Path, required=True)
    ap.add_argument('--observation-source', type=Path, required=True)
    ap.add_argument('--host-run', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(str(args.processor), local_files_only=True)
    source_hash = hashlib.sha256(args.observation_source.read_bytes()).hexdigest()
    assert source_hash == '1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3'
    spec = importlib.util.spec_from_file_location('native_observation', args.observation_source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.configure_image_budget(processor)
    rows = []
    records = [(run, index, record) for run in args.host_run
               for index, record in enumerate(json.loads((run/'result.json').read_text())['rounds'])]
    for run, index, record in records:
        path = run/f'turn_{index:03d}/request.json'
        messages = json.loads(path.read_text())
        paths = [p['image'] for m in messages if isinstance(m.get('content'), list)
                 for p in m['content'] if p.get('type') == 'image']
        assert len(paths) == len(set(paths)) == 5
        images = [Image.open(p).convert('RGB') for p in paths]
        # Scripted D is used ONLY to validate encoder placement; no reason
        # quality, target probability, or free generation is estimated.
        d = record['policy']['generated_decision_for_audit_only']
        vision = None
        for arm, stage in [('paired_phase_v3', 'decision'), ('paired_phase_v3', 'action'),
                           ('separate_phase_v3', 'action')]:
            batch = deployment_input(processor, messages, images, arm=arm, stage=stage,
                       generated_decision=d if arm=='paired_phase_v3' and stage=='action' else None,
                       max_length=16384)
            signature = {k: tensor_digest(v) for k,v in batch.items()
                         if k in ('pixel_values', 'image_grid_thw')}
            assert signature
            if vision is None:vision=signature
            else:assert signature==vision
            assert 'labels' not in batch
            rows.append({'host_run': str(run), 'turn': index, 'arm': arm, 'stage': stage,
                         'input_tokens': int(batch['input_ids'].shape[1]),
                         'input_ids_sha256': tensor_digest(batch['input_ids']),
                         'vision': signature, 'request_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'state_sha256': record['before_sha256']})
    output = {'actual_cpu_encodings': len(rows), 'rows': rows,
              'wall_seconds': time.monotonic()-start,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'real_model_calls': 0, 'training_updates': 0, 'cuda_initialized': torch.cuda.is_initialized(),
              'training_request_alignment': 'not established; this is the new native host format',
              'image_configuration_source_sha256': source_hash,
              'scope': 'Encoding of actual current-view requests; fixture D only, no model-quality result'}
    (args.out/'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
