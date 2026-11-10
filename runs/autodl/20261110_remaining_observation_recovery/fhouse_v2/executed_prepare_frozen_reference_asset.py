"""Run the archived mesh helper in its original trimesh version.

The native resource builder and mesh export historically used different
environments. Keep them in separate processes rather than swapping modules.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'original-plan', 'local-plan', 'out', 'receipt'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--asset', required=True)
    a = ap.parse_args()
    if trimesh.__version__ != '4.7.4':
        raise ValueError('Frozen canonical mesh exporter is trimesh 4.7.4')
    if sha(a.source) != '18ae517c85b1c3dffce41e269c83d9062142f2b56da0b96f005f0273f3278146':
        raise ValueError('Frozen helper source changed')
    old = json.loads(a.original_plan.read_text())['assets'][a.asset]
    local = json.loads(a.local_plan.read_text())['assets'][a.asset]
    assert sha(local['source_path']) == old['sha256'] == local['sha256']
    def load(path, **kwargs):
        scene = trimesh.load(path, **kwargs)
        scene.metadata.update(file_path=old['source_path'], file_name=Path(old['source_path']).name)
        return scene
    fn = [n for n in ast.parse(a.source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_mesh']
    assert len(fn) == 1
    scope = {'np': np, 'Path': Path, 'sha256': sha, 'json': json,
             'trimesh': SimpleNamespace(load=load, exchange=trimesh.exchange)}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(a.source), 'exec'), scope)
    canonical, bounds, gltf, rotation, shift = scope['prepare_mesh'](local, a.out)
    a.receipt.write_text(json.dumps({'canonical': str(canonical), 'bounds': bounds.tolist(),
        'gltf': str(gltf), 'rotation': rotation.tolist(), 'shift': shift.tolist(),
        'trimesh': trimesh.__version__, 'numpy': np.__version__,
        'helper_sha256': sha(a.source), 'raw_asset_sha256': sha(local['source_path']),
        'asset': a.asset, 'source_path_opened': local['source_path'],
        'serialized_provenance_path': old['source_path'],
        'files': {str(p): sha(p) for p in a.out.iterdir() if p.is_file()}}, indent=2)+'\n')


if __name__ == '__main__':
    main()
