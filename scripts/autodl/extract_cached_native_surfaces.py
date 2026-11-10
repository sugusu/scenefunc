"""Run pinned native support extraction on one declared canonical cache asset.

The original extractor consumes Y-up GLB, whereas this cache is Z-up. An
explicit inverse rotation supplies that contract; the native preparation path
must recover the original bounds. This is asset preprocessing, not generation,
physical certification, a learned method, or a human support annotation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    start=time.monotonic(); result={'status':'started'}; before={}
    try:
        if shutil.disk_usage(a.out).free < 5*1024**3+4*1024**2:
            raise RuntimeError('Preserve 5 GiB disk reserve plus bounded output allowance')
        manifest=json.loads((a.source/'manifest.json').read_text())
        assert manifest['pin']=='e888297a8b0a2f3acbb8634edf9def59cceab836'
        for row in manifest['files']:
            if row['status']=='downloaded':
                p=a.source/row['path']; assert sha(p)==row['sha256']; before[str(p)]=sha(p)
        cat=json.loads(a.catalogue.read_text())
        matches=[r for r in cat['assets'] if r['asset_id']==a.asset_id]
        if len(matches)!=1: raise ValueError('Need one exact declared asset ID')
        entry=matches[0]; mesh_path=Path(entry['geometry']['path'])
        assert sha(mesh_path)==entry['geometry']['sha256']
        before.update({str(a.catalogue):sha(a.catalogue),str(mesh_path):sha(mesh_path)})
        sys.path.insert(0,str(a.source.resolve()))
        from scenesmith.agent_utils.support_surface_extraction import (
            SupportSurfaceExtractionConfig, extract_support_surfaces_from_mesh,
            _load_and_prepare_mesh)
        cfg=SupportSurfaceExtractionConfig()
        mesh=trimesh.load(mesh_path,force='mesh')
        expected=np.asarray(entry['geometry']['bounds'])
        assert np.max(np.abs(mesh.bounds-expected))<1e-6
        y_to_z=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],dtype=float)
        mesh.apply_transform(np.linalg.inv(y_to_z))
        converted=a.out/'extractor_y_up_input.glb'
        converted.write_bytes(mesh.export(file_type='glb'))
        prepared=_load_and_prepare_mesh(converted)
        error=float(np.max(np.abs(prepared.bounds-expected)))
        if error>1e-6: raise ValueError('Native coordinate roundtrip changed geometry bounds')
        tick=time.monotonic()
        surfaces=extract_support_surfaces_from_mesh(converted,cfg)
        extraction_seconds=time.monotonic()-tick
        rows=[]
        for s in surfaces:
            rows.append({'surface_id':str(s.surface_id),
                'transform':s.transform.GetAsMatrix4().tolist(),
                'bounding_box_min':s.bounding_box_min.tolist(),
                'bounding_box_max':s.bounding_box_max.tolist(),
                'area':float(s.area),'vertices':s.mesh.vertices.tolist(),
                'faces':s.mesh.faces.tolist(),
                'frame':'surface-local vertices; transform maps surface to canonical asset frame'})
        write(a.out/'surfaces.json',{'asset_id':a.asset_id,'surfaces':rows,
            'provenance':'Unmodified native geometric extractor with default parameters; not independently verified semantic/support ground truth.'})
        result.update(status='native_extraction_completed',asset_id=a.asset_id,
            surface_count=len(rows),surface_areas_m2=[r['area'] for r in rows],
            surface_origins_m=[np.array(r['transform'])[:3,3].tolist() for r in rows],
            native_coordinate_roundtrip_max_bounds_error_m=error,
            canonical_bounds_m=expected.tolist(),config=asdict(cfg),
            input_conversion=np.linalg.inv(y_to_z).tolist(),
            converted_input_sha256=sha(converted),extraction_seconds=extraction_seconds,
            native_source_edited=False,thresholds_changed=False)
    except BaseException as e:
        result.update(status='failed',error=type(e).__name__+': '+str(e))
        raise
    finally:
        unchanged=all(sha(p)==v for p,v in before.items())
        write(a.out/'source_hashes.json',before)
        result.update(old_sources_unchanged=unchanged,
            wall_seconds=time.monotonic()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            model_calls=0,training_updates=0,new_scene_generations=0,
            physics_run=False,full_scene_success=None,
            scope='One selected real desk asset; no broad library or full-runtime readiness claim')
        write(a.out/'summary.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','catalogue','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--asset-id',required=True)
    main(p.parse_args())
