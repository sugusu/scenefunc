"""Restore frozen F-House development geometry using its original builder.

This is an input recovery, not generated furniture, a new task, or a method
result. Local paths are rebased and recorded. Original mesh provenance strings
are restored only inside glTF metadata, after raw geometry hashes match. The
entire original native builder runs, including HouseScene import/roundtrip.
All geometry resources must match the archived checkpoint before rendering.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import trimesh


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
RUN = 'runs/920/P920-FCHOICE-v2_20261105T193428Z/'
BUILDER_SHA = '18ae517c85b1c3dffce41e269c83d9062142f2b56da0b96f005f0273f3278146'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--sofa', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); result = {'status': 'started'}; reads = {}
    old_argv = sys.argv[:]
    def get(path):
        raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', PIN+':'+path])
        reads[path] = hashlib.sha256(raw).hexdigest()
        return raw
    try:
        if trimesh.__version__ != '4.5.3':
            raise ValueError('Frozen mesh runtime 4.5.3 required')
        if shutil.disk_usage(a.out).free < 5*1024**3+24*1024**2:
            raise RuntimeError('Need 24 MiB above retained data-disk reserve')
        cache = a.out/'source'; cache.mkdir()
        for name in ('build_reference_room.py', 'native_project.py', 'prepare_table_modes.py'):
            path = ('scripts/920/'+name if name == 'native_project.py' else RUN+'code/'+name)
            (cache/name).write_bytes(get(path))
        if sha(cache/'build_reference_room.py') != BUILDER_SHA:
            raise ValueError('Builder differs from original frozen plan')
        raw_plan = get(RUN+'REFERENCE_BUILD_PLAN.json'); frozen = json.loads(raw_plan)
        write(a.out/'original_plan.json', frozen)
        ref = a.out/'INITIAL_REFERENCE.json'; ref.write_bytes(get(RUN+'INITIAL_REFERENCE.json'))
        assert sha(ref) == frozen['initial_reference']['sha256']
        mapping = {frozen['objects']['sofa']['asset_id']: a.sofa.resolve()}
        generated = a.out/'raw'; generated.mkdir()
        tree = ast.parse((cache/'prepare_table_modes.py').read_text())
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'furniture_mesh']
        assert len(fn) == 1
        ns = {'np': np, 'trimesh': trimesh}
        exec(compile(ast.Module(body=fn, type_ignores=[]), str(cache/'prepare_table_modes.py'), 'exec'), ns)
        # Exact arguments from archived prepare_family_choice.py, not dimensions
        # selected to make a layout fit or new supervision.
        preparation = get(RUN+'code/prepare_family_choice.py')
        calls = [n for n in ast.walk(ast.parse(preparation)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'furniture_mesh']
        args = [tuple(ast.literal_eval(v) for v in call.args[1:]) for call in calls]
        assert args == [(1.6, .85, .74, False), (.44, .44, .45, False)]
        for aid, values in zip(('family_research_table', 'family_research_stool'), args):
            p = generated/(aid+'.glb'); ns['furniture_mesh'](p, *values); mapping[aid] = p.resolve()
        raw_checks = []
        plan = copy.deepcopy(frozen); plan['initial_reference']['path'] = str(ref.resolve())
        for aid, spec in plan['assets'].items():
            p = mapping[aid]
            assert sha(p) == spec['sha256'], aid
            raw_checks.append({'asset_id': aid, 'actual_path': str(p), 'sha256': sha(p),
                               'frozen_path': spec['source_path']})
            spec['source_path'] = str(p)
        local = a.out/'local_plan.json'; write(local, plan)
        sys.path.insert(0, str(cache.resolve()))
        sys.path.insert(0, str(a.source.resolve()))
        loader = importlib.util.spec_from_file_location('frozen_reference_builder', cache/'build_reference_room.py')
        module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
        original = module.prepare_mesh
        # Only serialized loader provenance is restored. Files opened and their
        # hashes are the local ones above. Original function body is unchanged.
        mesh_tree = ast.parse((cache/'build_reference_room.py').read_text())
        definition = [n for n in mesh_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_mesh']
        def prepare(spec, directory):
            original_spec = frozen['assets'][spec['asset_id']]
            def load(path, **kwargs):
                scene = trimesh.load(path, **kwargs)
                scene.metadata.update(file_path=original_spec['source_path'],
                                      file_name=Path(original_spec['source_path']).name)
                return scene
            scope = {'np': np, 'Path': Path, 'sha256': sha, 'json': json,
                     'trimesh': SimpleNamespace(load=load, exchange=trimesh.exchange)}
            exec(compile(ast.Module(body=definition, type_ignores=[]), str(cache/'build_reference_room.py'), 'exec'), scope)
            return scope['prepare_mesh'](spec, directory)
        module.prepare_mesh = prepare
        sys.argv = [str(cache/'build_reference_room.py'), '--plan', str(local.resolve()),
                    '--scenesmith', str(a.source.resolve()), '--out', str((a.out/'native_build').resolve())]
        try:
            module.main()
        finally:
            module.prepare_mesh = original
        bundle = a.out/'native_build/bundle'
        metadata = json.loads((bundle/'manifest.json').read_text()); rid = metadata['selected_room_id']
        expected = json.loads(get(RUN+'perimeter/checkpoint_metadata/manifest.json'))
        resources = {k: v for k, v in expected['files'].items() if k.startswith('room_'+rid+'/')}
        # The catalogue owns two already-declared solid proxies. Rebuild using
        # exactly its archived helper, without assigning new physical meaning.
        contacts = get('scripts/920/inspect_native_mesh_contacts.py')
        fns = [n for n in ast.parse(contacts).body if isinstance(n, ast.FunctionDef) and n.name == 'sdf_visuals']
        import xml.etree.ElementTree as ET
        scope = {'ET': ET, 'np': np, 'trimesh': trimesh, 'euler_matrix': trimesh.transformations.euler_matrix}
        exec(compile(ast.Module(body=fns, type_ignores=[]), 'pinned_sdf_visuals.py', 'exec'), scope)
        for name in ('sideboard', 'tall_storage'):
            p = bundle/('room_'+rid)/'assets'/('family_catalog_'+name)/'proxy.glb'
            p.parent.mkdir(parents=True, exist_ok=True)
            meshes = [m for _, m in scope['sdf_visuals'](bundle/('room_'+rid)/'objects'/name/'object.sdf', bundle, {})]
            trimesh.util.concatenate(meshes).export(p)
        checks = [{'path': p, 'expected_sha256': d['sha256'], 'actual_sha256': sha(bundle/p),
                   'exact': sha(bundle/p) == d['sha256'], 'bytes': (bundle/p).stat().st_size}
                  for p, d in resources.items()]
        write(a.out/'resource_checks.json', checks); write(a.out/'raw_asset_checks.json', raw_checks)
        if not all(r['exact'] for r in checks):
            raise ValueError('Reconstructed geometry differs from frozen checkpoint')
        # New bundle metadata has local path provenance. The frozen original
        # manifest is NOT overwritten or claimed reproduced. Register only
        # the verified added catalogue geometry in this new local manifest.
        for relative in resources:
            p = bundle/relative
            metadata['files'][relative] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        write(bundle/'manifest.json', metadata)
        module.validate_bundle(bundle)
        result.update(status='native_bundle_restored_geometry_byte_exact', raw_assets=len(raw_checks),
                      native_builder_executed=True, native_roundtrip_executed=True,
                      geometry_files=len(checks), exact_geometry_files=sum(r['exact'] for r in checks),
                      bundle=str(bundle.resolve()), original_manifest_reproduced=False,
                      rebase='Only input paths and recorded local provenance changed; original object geometry hashes required.',
                      objects=json.loads((a.out/'native_build/RESULT.json').read_text())['native_objects'])
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    finally:
        sys.argv = old_argv
        result.update(pin=PIN, sources=reads, seconds=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      model_calls=0, training_updates=0, new_scene_generations=0,
                      renderer_run=False, scope='Researcher-authored development reconstruction; not original historical dwelling or a new generation result.')
        write(a.out/'summary.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
