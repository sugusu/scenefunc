"""Encode archived common-input candidates on CPU without loading model weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise ValueError('Output exists')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    torch.set_num_threads(2)
    plan = read(args.study/'evaluation/Milan_frozen_base_knowledge_feedback_17/PLAN.json')
    if sha(plan['encoder']) != plan['encoder_sha256']: raise ValueError('Encoder changed')
    sys.path.insert(0, str(Path(plan['encoder']).parent))
    from local_vlm import LocalQwenEngine
    budget = plan['budget']
    engine = LocalQwenEngine(plan['model_directory'], args.out.parent/'unused_budget_model_logs',
        max_pixels=budget['max_pixels'], max_input_tokens=budget['max_input_tokens'], max_new_tokens=budget['max_new_tokens'])
    manifest = read(args.requests)
    if manifest['status'] != 'passed': raise ValueError('Input migration has not passed')
    records, failures = [], []
    print(datetime.now(timezone.utc).isoformat(), 'begin144deploymentCPUencodings', flush=True)
    for row in manifest['deployment']:
        if sha(row['request']) != row['request_sha256']: raise ValueError('Request changed')
        try:
            tensors, receipt = engine.encode(read(row['request']))
            if len(tensors['image_grid_thw']) != 5: raise ValueError('Expected five images')
            records.append(dict(cell=row['cell'], round=row['round'], request_sha256=row['request_sha256'], encoding=receipt))
        except ValueError as exc:
            failures.append(dict(cell=row['cell'], round=row['round'], error=str(exc), request_sha256=row['request_sha256']))
        if (len(records)+len(failures)) % 24 == 0:
            print('encoded', len(records), 'rejected', len(failures), flush=True)
    assert len(records)+len(failures) == 144 and not torch.cuda.is_initialized()
    result = dict(status='passed' if not failures else 'budget_or_encoding_failures', records=records, failures=failures,
                  completed=144, planned=144, budgets=budget, recorded_at=datetime.now(timezone.utc).isoformat(),
                  manifest_sha256=sha(args.requests), script_sha256=sha(__file__), model_calls=0, training_updates=0,
                  model_loaded=False, cuda_initialized=False,
                  limit='Historical observed requests only; this does not bound all future generated histories.')
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], passed=len(records), failed=len(failures))), flush=True)
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
