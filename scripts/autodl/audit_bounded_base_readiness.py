"""Revalidate one existing base-policy run without loading weights or changing scenes.

This audits real files and previous processor evidence. It is not an inference,
training, generation, or scene-quality test. No network calls are made.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def scalar(path):
    p = Path(path)
    return p.read_text().strip() if p.is_file() else None


def snapshot(root):
    return {
        'memory_limit_bytes': scalar('/sys/fs/cgroup/memory/memory.limit_in_bytes'),
        'memory_usage_bytes': scalar('/sys/fs/cgroup/memory/memory.usage_in_bytes'),
        'cpu_quota_us': scalar('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
        'cpu_period_us': scalar('/sys/fs/cgroup/cpu/cpu.cfs_period_us'),
        'nvidia_device_nodes': [str(p) for p in Path('/dev').glob('nvidia*')],
        'disk_free_bytes': shutil.disk_usage(root).free,
        'reserve_bytes': 5 * 1024 ** 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reuse-weight-receipt', type=Path,
                    help='Reuse completed full hashes only if every locked file and its size/mtime still match')
    args = ap.parse_args()
    start = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    w = args.work_base.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'started', 'model_calls': 0, 'tool_calls': 0,
              'training_updates': 0, 'new_scene_generations': 0,
              'paid_api_calls': 0, 'resources_before': snapshot(out)}
    try:
        cpu = w / 'autodl_continuation_20261110/cpu_replay'
        stage = cpu / 'keep'
        preflight = root / 'runs/autodl/20261110_consequence_review/corrected_preflight'
        source_map = read(cpu / 'source_hashes_before.json')
        source_actual = {p: sha(p) for p in source_map}
        assert source_actual == source_map, 'Original scene/source changed'
        result['original_source_hashes'] = source_actual

        lock_path = root / 'paris/config/weights.lock.json'
        lock = read(lock_path)
        assert lock['revision'] == '63e235a0c0d6e5bc2f8f3485214df4457fd2e672'
        result['weights_lock_sha256'] = sha(lock_path)
        result['model_repository'] = lock['repository']
        result['model_revision'] = lock['revision']
        result['weights'] = []
        reused = None
        if args.reuse_weight_receipt:
            previous = read(args.reuse_weight_receipt)
            assert previous['model_revision'] == lock['revision']
            assert previous['weights_lock_sha256'] == sha(lock_path)
            assert len(previous['weights']) == len(lock['files'])
            reused = {r['path']: r for r in previous['weights']}
            result['reused_full_weight_hashes'] = {
                'receipt': str(args.reuse_weight_receipt.resolve()),
                'receipt_sha256': sha(args.reuse_weight_receipt),
                'scope': 'Prior completed byte hashes; current sizes/mtimes revalidated. Other failed audit checks are not reused.'}
        checkpoint = w / 'critic_public_runtime_20261104/checkpoint'
        for row in lock['files']:
            p = checkpoint / row['path']
            before = p.stat()
            assert before.st_size == row['bytes'], str(p)
            if reused is None:
                actual = sha(p)
            else:
                old = reused[str(p)]
                assert (old['bytes'], old['mtime_ns']) == (before.st_size, before.st_mtime_ns)
                actual = old['sha256']
            after = p.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert actual == row['sha256'], 'Weight/config mismatch: ' + str(p)
            result['weights'].append({'path': str(p), 'bytes': before.st_size,
                                      'sha256': actual, 'mtime_ns': after.st_mtime_ns})
            print(json.dumps({'verified': p.name, 'bytes': before.st_size}), flush=True)

        spec = importlib.util.spec_from_file_location('bounded_readiness_target', root / 'scripts/autodl/bounded_trajectory.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cp = stage / 'tools/checkpoint.json'
        request = module.make_request(w, cp, read(stage / 'audit.json'),
            read(stage / 'planar_measurements.json'), read(stage / 'complete_finite_pool_joint.json'),
            stage / 'views_receipt.json', [])
        frozen_request = preflight / 'turn_00/request.json'
        assert request == read(frozen_request), 'Changed request; previous encoding no longer applies'
        encoding = read(preflight / 'turn_00/model/receipt.json')
        assert encoding['request_sha256'] == sha(frozen_request)
        assert encoding['status'] == 'completed' and encoding['model_loaded'] is False
        for view in encoding['encoded']['images']:
            assert sha(view['origin']) == view['sha256']
        assert len(encoding['encoded']['images']) == 5
        result['processor_evidence'] = {'path': str(preflight / 'turn_00/model/receipt.json'),
            'sha256': sha(preflight / 'turn_00/model/receipt.json'),
            'request_sha256': sha(frozen_request), 'unchanged_request': True,
            'input_tokens': encoding['encoded']['input_tokens'], 'views': 5,
            'new_processor_execution': False, 'scope': 'Prior real encoding, current byte/hash revalidation'}
        original = read(w / 'paris_local_run_results_20261104/B_mesh_feedback_sdpa_kv/tools/checkpoint.json')
        current = read(cp)
        assert current['payload'] == original['payload'], 'Reference and model starts differ'
        payload_hash = hashlib.sha256(json.dumps(current['payload'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        result['initial_state'] = {'checkpoint': str(cp), 'checkpoint_sha256': sha(cp),
            'payload_sha256': payload_hash, 'glb_path': str(cp.parent / 'partial_scene.glb'),
            'glb_sha256': sha(cp.parent / 'partial_scene.glb'), 'ordinary_same_initial_payload': True,
            'identity': 'Pre-existing furnished Paris development state with real selected meshes; not an independent project or new generation'}

        sources = [root / 'scripts/autodl/bounded_trajectory.py', root / 'scripts/autodl/model_step.py',
            root / 'scripts/autodl/cpu_continuation.py', Path(__file__),
            w / 'paris_local_run_results_20261104/local_vlm_sdpa.py',
            w / 'paris_local_run_results_20261104/render_state.py',
            w / 'native_execution_inventory_20261104/source/Designer/agent_rollout.py',
            w / 'joint_use_feedback_20261104/evaluate.py', w / 'joint_use_witness_20261104/measure_joint_use.py']
        result['execution_source_hashes'] = {str(p): sha(p) for p in sources}
        interpreters = [w / 'v100_runtime_20261104/runtime/bin/python',
            Path('/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python'),
            Path('/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python')]
        assert all(p.exists() for p in interpreters)
        result['interpreters'] = [{'path': str(p), 'resolved': str(p.resolve())} for p in interpreters]
        result['budget'] = read(preflight / 'config.json')
        resources = snapshot(out)
        blockers = []
        if not resources['nvidia_device_nodes']:
            blockers.append('No NVIDIA device exposed; no model subprocess started')
        if int(resources['memory_limit_bytes'] or 0) < 24 * 1024 ** 3:
            blockers.append('Below existing conservative 24 GiB host-memory gate (not a measured minimum)')
        # This reserve is a pre-start budget, not a claim about measured output size.
        if resources['disk_free_bytes'] < 5 * 1024 ** 3 + 200 * 1024 ** 2:
            blockers.append('Below 5 GiB reserve plus 200 MiB prospective output margin')
        result['resources_after'] = resources
        result['resource_blockers'] = blockers
        result['gpu_free_memory_verified'] = False
        result['gpu_requirement'] = 'One already-authorized V100 32 GB or compatible GPU, at least 26 GiB free VRAM under existing gate; verify after allocation'
        result['status'] = 'input_and_weights_verified_resources_unavailable' if blockers else 'cpu_ready_gpu_free_memory_still_to_verify'
        assert {p: sha(p) for p in source_map} == source_map
        result['original_sources_unchanged_after'] = True
    except Exception as exc:
        result.update(status='audit_failed_no_inference', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - start,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ended_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({k: result.get(k) for k in ['status', 'error', 'resource_blockers', 'elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
