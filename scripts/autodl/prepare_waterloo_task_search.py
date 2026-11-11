"""Rebuild the exact frozen Waterloo development input in a new scratch dir.

Does not fetch packages, models, or assets. Original raw HSSD sofa must already
exist and match the source hash; generated substitutes were recovered from the
pinned Git source and must match too. Old experiment inputs remain read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--native-source', type=Path, required=True)
    p.add_argument('--hssd-root', type=Path, default=Path('/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/official_assets/hssd-models'))
    a = p.parse_args()
    root = a.root.resolve(); scratch = a.scratch.resolve()
    if scratch.exists():
        raise FileExistsError('Use a fresh scratch directory; no implicit rerun')
    m = json.loads((root/'source_manifest.json').read_text())
    for row in m['files']:
        local = root/'source'/row['path'].split('P920-WATERLOO-v1_20261105T074801Z/', 1)[1]
        assert local.stat().st_size == row['bytes'] and sha(local) == row['sha256'], local
    native_helper = root/'source/code/native_project.py'
    assert sha(native_helper) == '1c418410b1d5db8606305f200154418dd59844ce1c22fee3afb903286417ad19'
    plan = json.loads((root/'source/BUILD_PLAN.json').read_text())
    ref = root/'source/INITIAL_REFERENCE.json'
    plan['initial_reference'] = dict(path=str(ref), sha256=sha(ref))
    for aid, asset in plan['assets'].items():
        if aid.startswith('research_'):
            path = root/'source/generated_assets'/('rectangular_wheeled_table.glb' if 'table' in aid else 'stool.glb')
        else:
            path = a.hssd_root/'objects'/aid[0]/(aid+'.glb')
        assert path.is_file() and sha(path) == asset['sha256'], path
        asset['source_path'] = str(path.resolve())
    scratch.mkdir(parents=True)
    local_plan = scratch/'local_plan.json'
    local_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    cmd = [sys.executable, '-B', str(root/'source/code/build_reference_room.py'),
           '--plan', str(local_plan), '--scenesmith', str(a.native_source.resolve()),
           '--out', str(scratch/'native_build')]
    env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    start = time.monotonic()
    with (scratch/'restore.log').open('w') as log:
        completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    receipt = dict(command=cmd, exit_code=completed.returncode, wall_seconds=time.monotonic()-start,
                   input_code_sha256=sha(root/'source/code/build_reference_room.py'),
                   new_scene_generation=False, model_calls=0, training_updates=0)
    (scratch/'restore_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if completed.returncode:
        raise RuntimeError('Restore failed; retain log, no implicit retry')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
