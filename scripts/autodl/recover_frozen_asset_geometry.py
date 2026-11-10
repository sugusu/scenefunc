"""Rebuild canonical asset files with the pinned original pure mesh helper.

Asset identity comes from a frozen plan, not from a new retrieval or prediction.
This never reconstructs poses, replaces imagery, starts native services or
claims that an asset's existence makes a complete input bundle recoverable.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import resource
import platform
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import trimesh


PIN='75fc783f86d4fea78714639e110fbe77476ee106'
ROOT='runs/920/P920-PAPER-v2_20261105T182424Z/'
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--sofa',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--restore-source-metadata',action='store_true')
    ap.add_argument('--expect-trimesh-version')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.expect_trimesh_version and trimesh.__version__!=a.expect_trimesh_version:
        raise RuntimeError('Pinned trimesh version is not active')
    if shutil.disk_usage(a.out).free<5*1024**3+20*1024**2:raise RuntimeError('Need 20 MiB above reserve')
    tick=time.monotonic();reads={}
    def get(p):
        raw=subprocess.check_output(['git','-c','gc.auto=0','show',PIN+':'+p]);reads[p]=hashlib.sha256(raw).hexdigest();return raw
    plan=json.loads(get(ROOT+'REFERENCE_BUILD_PLAN.json'))
    manifest=json.loads(get(ROOT+'input_metadata/manifest.json'))
    pinned=get('scripts/920/build_reference_room.py');assert a.source.read_bytes()==pinned
    assert hashlib.sha256(pinned).hexdigest()==plan['code_sha256']
    chosen=[n for n in ast.parse(pinned).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_mesh']
    assert len(chosen)==1
    ns={'np':np,'trimesh':trimesh,'Path':Path,'sha256':sha256,'json':json}
    exec(compile(ast.Module(body=chosen,type_ignores=[]),'pinned_prepare_mesh.py','exec'),ns)
    rawdir=a.out/'raw';rawdir.mkdir()
    mapping={'4979cfe2e691ac484078b909543db3f21b80462f':a.sofa}
    for asset,name in [('paper_research_table','rectangular_wheeled_table.glb'),('paper_research_stool','stool.glb')]:
        p=rawdir/name;p.write_bytes(get('runs/920/P920-WATERLOO-v1_20261105T074801Z/generated_assets/'+name));mapping[asset]=p
    rows=[]
    for aid,spec in plan['assets'].items():
        original=mapping[aid];assert sha256(original)==spec['sha256']
        rebased=copy.deepcopy(spec);rebased['source_path']=str(original.resolve())
        # Trimesh serializes the loader path into scene.extras. Restore only
        # the two known original provenance strings after validating raw bytes.
        # Actual local I/O stays in the receipt; no remote path is opened.
        restored={}
        if a.restore_source_metadata:
            restored={'file_path':spec['source_path'],'file_name':Path(spec['source_path']).name}
            def load_with_metadata(path,**kwargs):
                scene=trimesh.load(path,**kwargs)
                scene.metadata.update(restored)
                return scene
            ns['trimesh']=SimpleNamespace(load=load_with_metadata,exchange=trimesh.exchange)
        dest=a.out/'canonical'/aid
        canonical,bounds,gltf,rotation,shift=ns['prepare_mesh'](rebased,dest)
        prefix='room_paper_shared_living_reconstruction/assets/'+aid+'/'
        checks=[]
        for relative,expected in manifest['files'].items():
            if not relative.startswith(prefix):continue
            file=dest/relative[len(prefix):]
            checks.append({'relative':relative,'expected_sha256':expected['sha256'],
                'actual_sha256':sha256(file) if file.exists() else None,
                'byte_exact':file.exists() and sha256(file)==expected['sha256']})
        row={'asset_id':aid,'raw_sha256':sha256(original),'source_identity_exact':True,
            'actual_input_path':str(original.resolve()),'restored_source_metadata':restored,
            'original_helper_sha256':hashlib.sha256(pinned).hexdigest(),
            'native_bounds':bounds.tolist(),'shift':shift.tolist(),'checks':checks,
            'complete_manifest_match':bool(checks) and all(x['byte_exact'] for x in checks)}
        rows.append(row);write(a.out/'asset_checks.json',rows)
        print(aid,'exact',sum(x['byte_exact'] for x in checks),'/',len(checks),flush=True)
    write(a.out/'summary.json',{'pin':PIN,'sources':reads,'assets':len(rows),
        'runtime':{'python':platform.python_version(),'trimesh':trimesh.__version__,
            'numpy':np.__version__,'trimesh_import':trimesh.__file__},
        'restored_original_path_metadata':a.restore_source_metadata,
        'exact_canonical_files':sum(c['byte_exact'] for r in rows for c in r['checks'] if c['relative'].endswith('canonical.glb')),
        'all_geometry_and_material_files_exact':all(r['complete_manifest_match'] for r in rows),
        'file_checks':sum(len(r['checks']) for r in rows),
        'bytes_written':sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file()),
        'seconds':time.monotonic()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'renderer_run':False,'model_calls':0,'training_updates':0,'scene_actions':0,
        'view_equivalence_proven':False,'training_ready':False})


if __name__=='__main__':main()
