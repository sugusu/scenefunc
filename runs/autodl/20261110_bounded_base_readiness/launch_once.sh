#!/usr/bin/env bash
# One pre-existing furnished-room diagnostic. Never opens billed resources.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_REPO="$(cd -- "$TASK_DIR/../../.." && pwd)"
TASK_WORK=/root/autodl-tmp/agent_method_preparation_20261027/work
TASK_PY=/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python
TASK_DEPS=/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK_DEPS"
"$TASK_PY" - "$TASK_DIR/audit_v2/result.json" <<'PY'
import hashlib, json, pathlib, shutil, sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert d['status']=='input_and_weights_verified_resources_unavailable'
for r in d['weights']:
    s=pathlib.Path(r['path']).stat()
    assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns']),r['path']
for p,h in {**d['original_source_hashes'],**d['execution_source_hashes']}.items():
    with pathlib.Path(p).open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==h,p
pr=pathlib.Path(d['processor_evidence']['path'])
assert hashlib.sha256(pr.read_bytes()).hexdigest()==d['processor_evidence']['sha256']
images=json.loads(pr.read_text())['encoded']['images']
assert len(images)==5
for v in images:
    assert hashlib.sha256(pathlib.Path(v['origin']).read_bytes()).hexdigest()==v['sha256']
limit=int(pathlib.Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text())
devices=list(pathlib.Path('/dev').glob('nvidia*'))
free=shutil.disk_usage(pathlib.Path(sys.argv[1]).parent).free
if limit<24*1024**3 or not devices or free<5*1024**3+200*1024**2:
    print(json.dumps({'status':'resources_unavailable_no_model_started',
        'memory_limit_bytes':limit,'nvidia_nodes':[str(p) for p in devices],
        'disk_free_bytes':free,'minimum_disk_start_margin_bytes':200*1024**2,
        'model_calls':0,'training_updates':0,'paid_api_calls':0}))
    sys.exit(2)
# Actual CUDA availability/free VRAM is checked inside the bounded subprocess.
PY
exec "$TASK_PY" "$TASK_REPO/scripts/autodl/bounded_trajectory.py" \
  --phase run --work-base "$TASK_WORK" \
  --cpu-replay "$TASK_WORK/autodl_continuation_20261110/cpu_replay" \
  --output "$TASK_DIR/model_run_v1"
