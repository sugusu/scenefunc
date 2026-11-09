"""Finish existing native views in separate processes, without redoing actions.

The two valid images and prepared blend come from the retained failed render.
Only missing views run; camera, samples, geometry and native functions stay fixed.
This is an explicit recovery command, not an automatic quality retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--failed-views', type=Path, required=True)
    ap.add_argument('--renderer-root', type=Path, required=True)
    ap.add_argument('--python', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root, prior = a.out.resolve(), a.failed_views.resolve()
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads((prior/'receipt.json').read_text())
    proof = json.loads((prior/'prepared/receipt.json').read_text())
    blend = prior/'prepared/current_state.blend'
    assert old['status'] == 'failed' and old['render_exit_code'] == -9
    assert sha(blend) == proof['blend_sha256']
    assert sha(old['source_scene']) == old['source_sha256'] == proof['source_sha256']
    cardinal = a.renderer_root.resolve()/'tools/render_cardinal_views.py'
    five = a.renderer_root.resolve()/'tools/render_five_views.py'
    from PIL import Image
    start = time.monotonic()
    record = dict(status='running', source_scene=old['source_scene'], source_sha256=old['source_sha256'],
                  prior_failed_receipt=str(prior/'receipt.json'), blend_sha256=sha(blend),
                  preparation_receipt=str(prior/'prepared/receipt.json'),
                  source_hashes={str(p):sha(p) for p in (cardinal, five)}, calls=[],
                  paid_api_calls=0, model_calls=0, native_actions=0,
                  scope='Only process lifetime changes. Existing images retained; missing views use pinned native functions.')
    env = dict(os.environ, SCENE_RENDER_THREADS='6', SCENE_RENDER_SAMPLES='24',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    try:
        for view in ('east','north','west','south'):
            name = f'current_state.{view}.png'
            source = prior/'prepared'/name
            if source.exists():
                with Image.open(source) as im: im.verify()
                shutil.copyfile(source, root/name)
                record['calls'].append(dict(view=view, kind='retained_valid_image', path=str(source), sha256=sha(source)))
                continue
            command = [a.python, str(cardinal), str(blend), '--out', str(root), '--views', view]
            tick = time.monotonic()
            with (root/f'{view}.log').open('w') as f:
                p = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900)
            record['calls'].append(dict(view=view, command=command, exit_code=p.returncode, seconds=time.monotonic()-tick))
            if p.returncode: raise RuntimeError(f'{view} failed; no automatic retry')
        # Native five-view entry skips the four completed canonical images,
        # then invokes its original topdown path. No source edits or AST cut.
        command = [a.python, str(five), str(blend), '--out', str(root)]
        tick = time.monotonic()
        with (root/'topdown.log').open('w') as f:
            p = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900)
        record['calls'].append(dict(view='topdown', command=command, exit_code=p.returncode, seconds=time.monotonic()-tick))
        if p.returncode: raise RuntimeError('topdown failed; no automatic retry')
        views = []
        for view in ('topdown','south','east','west','north'):
            path = root/f'current_state.{view}.png'
            with Image.open(path) as im:
                size = list(im.size)
                im.verify()
            views.append(dict(path=str(path), sha256=sha(path), size=size))
        assert sha(blend) == record['blend_sha256']
        assert all(sha(p) == h for p,h in record['source_hashes'].items())
        record.update(status='completed', views=views)
    except BaseException as exc:
        record.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-start
        (root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'elapsed_seconds':record['elapsed_seconds']}))


if __name__ == '__main__':main()
