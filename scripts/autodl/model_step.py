"""One local encode or inference subprocess. No API, CPU weights, or retry."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import resource
import socket
import sys
import time
from cpu_continuation import dump, read, sha, resource_snapshot


def native_parser(w):
    source = w / 'native_execution_inventory_20261104/source/Designer/agent_rollout.py'
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('_TOOL_CALL_RE', '_THINK_RE') for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == 'parse_assistant_text':
            nodes.append(node)
    scope = {'re': re, 'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope['parse_assistant_text']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--phase', choices=['encode', 'generate'], required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    w = args.work_base.resolve()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    network = []
    def reject(*a, **kw):
        network.append('attempt')
        raise RuntimeError('Local run forbids network connections')
    socket.socket.connect = reject
    socket.create_connection = reject
    start = time.monotonic()
    receipt = {'phase': args.phase, 'request_sha256': sha(args.request), 'model_calls': 0, 'paid_api_calls': 0, 'training': False,
               'model_revision': '63e235a0c0d6e5bc2f8f3485214df4457fd2e672', 'status': 'started'}
    try:
        sys.path.insert(0, str(w / 'paris_local_run_results_20261104'))
        from local_vlm_sdpa import LocalQwenEngine
        engine = LocalQwenEngine(w / 'critic_public_runtime_20261104/checkpoint', out / 'calls',
                                 max_pixels=65536, max_input_tokens=8192, max_new_tokens=1024)
        import torch
        torch.manual_seed(0)
        if args.phase == 'generate':
            snap = resource_snapshot(out)
            if not torch.cuda.is_available():
                raise RuntimeError('No CUDA device; no CPU weight fallback')
            free, total = torch.cuda.mem_get_info()
            receipt['resources'] = dict(snap, gpu_free_bytes=free, gpu_total_bytes=total, gpu_name=torch.cuda.get_device_name())
            if int(snap['memory_limit_bytes']) < 24 * 1024**3 or free < 26 * 1024**3:
                raise RuntimeError('Below conservative 24 GiB host / 26 GiB free VRAM gates; not measured minima')
            result = engine.complete(read(args.request), purpose='autodl_bounded_complete_state_edit')
            (out / 'raw_output.txt').write_text(result['raw'])
            (out / 'text_output.txt').write_text(result['text'])
            _, calls = native_parser(w)(result['raw'])
            dump(out / 'parsed_calls.json', calls)
            receipt.update(model_calls=1, engine_receipt=result['receipt_dir'], parsed_calls=len(calls),
                           upstream_no_tool_termination=not calls,
                           formatting={'think_open': result['raw'].count('<think>'), 'think_close': result['raw'].count('</think>'),
                                       'tool_open': result['raw'].count('<tool_call>'), 'tool_close': result['raw'].count('</tool_call>')})
        else:
            _, encoded = engine.encode(read(args.request))
            assert engine._model is None
            receipt.update(encoded=encoded, model_loaded=False)
        receipt['status'] = 'completed'
    except Exception as e:
        receipt.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       network_connection_attempts=len(network))
        dump(out / 'receipt.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
