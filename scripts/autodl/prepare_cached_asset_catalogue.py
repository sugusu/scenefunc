"""Inspect a retained native mesh cache without importing model/SDK services.

Exports a pose-free catalogue with original asset descriptions and geometry
provenance. Does not invent physics, support surfaces, or semantic validation.
Calls the original RoomScene directive method on the actual cached templates
to distinguish GLB availability from native Drake export coverage.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from collections import Counter

import numpy as np
import trimesh


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    doc=json.loads(a.checkpoint.read_text());state=doc['payload']
    canonical=json.dumps(state,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest()!=doc['payload_sha256']:
        raise ValueError('Old checkpoint content changed')
    templates=state['asset_templates'];records={r['object_id']:r for r in state['asset_records']}
    if len(records)!=len(templates) or {r['object_id'] for r in templates}!=set(records):
        raise ValueError('Template/source record mapping differs')
    before={str(a.checkpoint):sha(a.checkpoint)}
    original_script=a.base/'native_room_bridge_20261104/native_kernels.py'
    native=load(original_script,'cached_asset_native_kernels')
    G=native.G
    room=G['RoomScene'].__new__(G['RoomScene'])
    room.objects={G['UniqueID'](r['object_id']):G['SceneObject'].from_dict(r) for r in templates}
    # Room excluded here to isolate actual object export behavior. This is NOT
    # an empty-room generator, a parser run, or a renderer test.
    directive=room.to_drake_directive(exclude_room_geometry=True)
    (a.out/'actual_cached_object_directive.dmd.yaml').write_text(directive+'\n')
    exported=directive.count('- add_model:')
    catalogue=[];audit=[]
    for t in templates:
        r=records[t['object_id']];geometry=Path(t['geometry_path']);source=Path(r['source_path'])
        frame=r['coordinate_frame'];frame_path=Path(frame['source_path'])
        expected=state['geometry_hashes'].get(str(geometry))
        if expected is None or sha(geometry)!=expected:raise ValueError('Cache mesh hash mismatch')
        if sha(source)!=r['source_sha256']:raise ValueError('Original HSSD mesh hash mismatch')
        if sha(frame_path)!=frame['source_sha256']:raise ValueError('Orientation metadata hash mismatch')
        for p in (geometry,source,frame_path):before[str(p)]=sha(p)
        mesh=trimesh.load(geometry,force='mesh')
        bounds=np.array([t['bbox_min'],t['bbox_max']],dtype=float)
        error=float(np.max(np.abs(mesh.bounds-bounds)))
        if error>1e-6:raise ValueError('Cached mesh differs from recorded asset bounds')
        aid='hssd:'+r['asset_id_source']
        entry={'asset_id':aid,'description':r['actual_name'],'categories':r['categories'],
            'geometry':{'path':str(geometry),'sha256':expected,'bounds':bounds.tolist(),
                        'triangles':len(mesh.faces)},
            'original_asset':{'path':str(source),'sha256':r['source_sha256']},
            'coordinate_frame_source':frame,
            'native_sdf':t.get('sdf_path'),'support_surfaces':t.get('support_surfaces',[]),
            'semantic_front_verified':t['metadata'].get('semantic_front_verified',False),
            'functional_quality_verified':t['metadata'].get('functional_quality_verified',False),
            'scope':'Existing asset geometry/metadata only. No historical room pose, requested dimensions, chosen action, or evaluation label.'}
        catalogue.append(entry)
        audit.append({'asset_id':aid,'old_template_id':t['object_id'],
            'mesh_bounds_max_error_m':error,'native_sdf_present':bool(t.get('sdf_path')),
            'native_sdf_exists':bool(t.get('sdf_path')) and Path(t['sdf_path']).is_file(),
            'support_surface_count':len(t.get('support_surfaces',[])),
            'semantic_front_verified':entry['semantic_front_verified']})
        del mesh
    if len({r['asset_id'] for r in catalogue})!=len(catalogue):
        raise ValueError('Duplicate source asset identities; deduplicate explicitly before publishing')
    # Keep source labels separate from any previous model-assigned instance name.
    dump(a.out/'catalogue.json',{'format':'geometry_cache_catalogue_v1','assets':catalogue,
        'selection':'All templates in the declared old checkpoint; this is a previously selected subset, not a complete HSSD catalogue.',
        'new_annotations':False,'whole_scene_ready':False})
    dump(a.out/'asset_audit.json',audit)
    for row in native.READ:
        p=Path(row['path']);before[str(p)]=sha(p)
    before[str(original_script)]=sha(original_script)
    after={p:sha(p) for p in before}
    if before!=after:raise ValueError('Read-only source changed')
    dump(a.out/'source_hashes.json',before)
    result={'status':'actual_cache_inspected','asset_count':len(catalogue),
        'native_object_records':len(room.objects),'native_drake_add_model_count':exported,
        'missing_sdf_count':sum(not r['native_sdf_present'] for r in audit),
        'nonempty_support_surface_assets':sum(r['support_surface_count']>0 for r in audit),
        'semantic_front_verified_count':sum(r['semantic_front_verified'] for r in audit),
        'categories':dict(Counter(c for r in catalogue for c in r['categories'])),
        'all_actual_mesh_bounds_match':True,'old_sources_unchanged':True,
        'native_directive_method_executed':True,'drake_parser_executed':False,
        'new_native_placements':0,'model_calls':0,'training_updates':0,'new_scene_generations':0,
        'whole_native_runtime_ready':False,'wall_seconds':time.monotonic()-start,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'conclusion':'A valid local mesh cache does not supply the missing native SDF/support state. Do not claim complete native replay or remove such objects from the scoring denominator.'}
    dump(a.out/'summary.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    main(ap.parse_args())
