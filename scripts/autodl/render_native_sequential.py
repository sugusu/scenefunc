"""Separate pinned GLB preparation and five-view rendering into two processes.

No camera, scene, sample count or render function is changed. The existing
preparer otherwise retains a loaded Blender while launching another Blender.
Only a checked AST cut after construction of its renderer command is used to
save the prepared scene and exit. The original source remains read-only.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')


def prepare(source, expected, scene, out):
    if sha(source) != expected:
        raise ValueError('Pinned preparation source changed')
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    block = next(n for n in main.body if isinstance(n, ast.Try))
    slots = [i for i, n in enumerate(block.body) if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets)]
    if len(slots) != 1:
        raise ValueError('Source renderer launch site changed')
    cut = slots[0]
    block.body = block.body[:cut+1] + ast.parse(
        "receipt.update(status='prepared_without_render', command=command, "
        "preparation_source_sha256=SOURCE_HASH)\nreturn\n").body
    # Main's actual finally persists the import/vertex checks and blend hash.
    namespace = {'__name__': '__main__', '__file__': str(source), 'SOURCE_HASH': expected}
    sys.argv = [str(source), '--scene', str(scene), '--out', str(out)]
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--source-sha256', required=True)
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--render-python', required=True)
    ap.add_argument('--prepare-worker', action='store_true')
    args = ap.parse_args()
    if args.prepare_worker:
        prepare(args.source.resolve(), args.source_sha256, args.scene.resolve(), args.out.resolve())
        return
    start = time.monotonic()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prepared = root/'prepared'
    env = dict(os.environ, SCENE_RENDER_THREADS='6', SCENE_RENDER_SAMPLES='24',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    prep_command = [args.render_python, str(Path(__file__).resolve()), '--prepare-worker',
                    '--source', str(args.source.resolve()), '--source-sha256', args.source_sha256,
                    '--scene', str(args.scene.resolve()), '--out', str(prepared),
                    '--render-python', args.render_python]
    result = {'status': 'started', 'renderer_changed': False, 'paid_api_calls': 0,
              'model_calls': 0, 'source_scene': str(args.scene.resolve()),
              'source_sha256': sha(args.scene)}
    try:
        with (root/'prepare.log').open('w') as f:
            p = subprocess.run(prep_command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
        result['preparation_exit_code'] = p.returncode
        if p.returncode:
            raise RuntimeError('Preparation failed')
        proof = json.loads((prepared/'receipt.json').read_text())
        assert proof['status'] == 'prepared_without_render'
        assert proof['source_sha256'] == result['source_sha256']
        command = proof['command']
        assert sha(prepared/'current_state.blend') == proof['blend_sha256']
        tick = time.monotonic()
        with (root/'render.log').open('w') as f:
            p = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1500)
        result.update(render_exit_code=p.returncode, render_seconds=time.monotonic()-tick)
        if p.returncode:
            raise RuntimeError('Rendering failed')
        from PIL import Image
        views = []
        for name in ('topdown', 'south', 'east', 'west', 'north'):
            path = prepared/f'current_state.{name}.png'
            with Image.open(path) as image:
                size = list(image.size)
                image.verify()
            views.append({'path': str(path), 'sha256': sha(path), 'size': size})
        assert sha(args.scene) == result['source_sha256']
        result.update(status='completed', views=views,
                      preparation_receipt=str(prepared/'receipt.json'),
                      scope='Same pinned GLB adapter and native renderer, sequential processes')
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        write(root/'receipt.json', result)


if __name__ == '__main__':
    main()
