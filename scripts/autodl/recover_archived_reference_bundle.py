"""Execute a frozen reference builder and require archived geometry hashes.

Only input-path rebasing and a version-isolated original mesh helper are used.
No generated labels, model calls, geometry fixes or proposed scene edits.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import trimesh


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['out','source','assets-root','mesh-deps']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--pin', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--plan-name', default='BUILD_PLAN.json')
    ap.add_argument('--file-ledger', default='SERVER_FILES.json')
    ap.add_argument('--ledger-prefix', default='build/bundle/')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); sources = {}; result = {'status':'started'}
    old_argv = sys.argv[:]
    def dump(name, value):
        (a.out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    def get(name):
        path = 'runs/920/'+a.run_id+'/'+name
        b = subprocess.check_output(['git','-c','gc.auto=0','show',a.pin+':'+path])
        sources[path] = hashlib.sha256(b).hexdigest()
        return b
    try:
        if trimesh.__version__ != '4.5.3':
            raise ValueError('Original builder environment requires trimesh 4.5.3')
        if shutil.disk_usage(a.out).free < 5*1024**3+48*1024**2:
            raise RuntimeError('Insufficient headroom above data reserve')
        source = a.out/'source'; source.mkdir()
        for name in ['build_reference_room.py','native_project.py']:
            (source/name).write_bytes(get('code/'+name))
        plan = json.loads(get(a.plan_name)); dump('original_plan.json',plan)
        if sha(source/'build_reference_room.py') != plan['code_sha256']:
            raise ValueError('Builder differs from frozen plan')
        reference = a.out/'INITIAL_REFERENCE.json'; reference.write_bytes(get('INITIAL_REFERENCE.json'))
        if sha(reference) != plan['initial_reference']['sha256']:
            raise ValueError('Reference differs from frozen plan')
        local = copy.deepcopy(plan); local['initial_reference']['path'] = str(reference.resolve())
        raw = []
        for aid, spec in local['assets'].items():
            p = a.assets_root/'objects'/aid[0]/(aid+'.glb')
            if not p.is_file() or sha(p) != spec['sha256']:
                raise ValueError('Missing or changed original asset: '+aid)
            raw.append({'asset_id':aid,'path':str(p.resolve()),'sha256':sha(p)})
            spec['source_path'] = str(p.resolve())
        dump('local_plan.json',local); dump('raw_asset_checks.json',raw)
        sys.path.insert(0,str(source.resolve())); sys.path.insert(0,str(a.source.resolve()))
        spec = importlib.util.spec_from_file_location('frozen_builder',source/'build_reference_room.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        receipts = a.out/'asset_receipts'; receipts.mkdir()
        original = module.prepare_mesh
        def prepare(asset, directory):
            receipt = receipts/(asset['asset_id']+'.json')
            command = [sys.executable,'-B',str(Path(__file__).with_name('prepare_frozen_reference_asset.py').resolve()),
                '--source',str((source/'build_reference_room.py').resolve()),'--original-plan',str((a.out/'original_plan.json').resolve()),
                '--local-plan',str((a.out/'local_plan.json').resolve()),'--asset',asset['asset_id'],
                '--out',str(directory.resolve()),'--receipt',str(receipt.resolve())]
            env = dict(os.environ,PYTHONPATH=str(a.mesh_deps.resolve()),PYTHONDONTWRITEBYTECODE='1')
            subprocess.run(command,env=env,check=True,timeout=120)
            d = json.loads(receipt.read_text())
            return Path(d['canonical']),np.array(d['bounds']),Path(d['gltf']),np.array(d['rotation']),np.array(d['shift'])
        module.prepare_mesh = prepare
        sys.argv = [str((source/'build_reference_room.py').resolve()),'--plan',str((a.out/'local_plan.json').resolve()),
                    '--scenesmith',str(a.source.resolve()),'--out',str((a.out/'native_build').resolve())]
        try:
            module.main()
        finally:
            module.prepare_mesh = original
        bundle = a.out/'native_build/bundle'
        manifest = json.loads((bundle/'manifest.json').read_text())
        rid = manifest['selected_room_id']; ledger = json.loads(get(a.file_ledger))
        dump('original_file_ledger.json',ledger)
        prefix = a.ledger_prefix+'room_'+rid+'/'
        checks = []
        for path, entry in ledger['files'].items():
            if not path.startswith(prefix):continue
            rel = path[len(a.ledger_prefix):]; p = bundle/rel
            actual = sha(p) if p.is_file() else None
            checks.append({'path':rel,'expected_sha256':entry['sha256'],'actual_sha256':actual,'exact':actual==entry['sha256']})
        dump('resource_checks.json',checks)
        if not checks or not all(c['exact'] for c in checks):
            raise ValueError('Original resource hashes not reproduced')
        local_resources = {str(p.relative_to(bundle)) for p in (bundle/('room_'+rid)).rglob('*') if p.is_file()}
        if local_resources != {c['path'] for c in checks}:
            raise ValueError('Geometry ledger does not cover all local room resources')
        module.validate_bundle(bundle)
        result.update(status='original_builder_and_geometry_verified',bundle=str(bundle.resolve()),
            geometry_files=len(checks),original_manifest_reproduced=False,
            objects=json.loads((a.out/'native_build/RESULT.json').read_text())['native_objects'])
    except BaseException as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc)); raise
    finally:
        sys.argv = old_argv
        result.update(source_pin=a.pin,run_id=a.run_id,sources=sources,seconds=time.monotonic()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,training_updates=0,
            new_scene_generations=0,scope='Frozen development input recovery, not a generated room or professional design label')
        dump('summary.json',result)
        print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
