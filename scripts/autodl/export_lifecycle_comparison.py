"""Cold-restore both actual endpoints and export all whole-home visual meshes.

No neural inference/physics/target-plan inputs. Large geometry is built in RAM;
content-identical new outputs can share one local artifact, with both arm-to-
artifact mappings explicit. No old evidence is deduplicated by this script.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import trimesh


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--scratch',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();scratch=a.scratch.resolve()
    out=scratch/'geometry'
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    sys.path.insert(0,str(Path('runs/autodl/20261110_native_state_runtime/source').resolve()))
    sys.path.insert(0,str(Path('runs/autodl/20261110_public_function_search/original_evaluator').resolve()))
    from native_project import import_project,validate_bundle
    from inspect_native_mesh_contacts import sdf_visuals,world_transform
    source=scratch/'initial/bundle';m,h,r,_=validate_bundle(source);rid=m['selected_room_id']
    room=source/('room_'+rid)
    up=np.eye(4);up[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    cat={e['template']['name']:e for e in read(source/'catalog.json')['entries']}
    rows=[]
    states={'initial':h['rooms'][rid]}
    for arm in read(root/'protocol.json')['arms']:
        states[arm]=read(root/arm/'final_state.json')
    for label,s in states.items():
        cold=None
        if label!='initial':
            bundle=scratch/label/'final_bundle';shutil.copytree(source,bundle)
            house=copy.deepcopy(h);house['rooms'][rid]=s
            runtime=copy.deepcopy(r);bindings=runtime['rooms'][rid]['asset_bindings']
            if 'former_child_bed_0' not in s['objects']:
                old=bindings.pop('former_child_bed_0')
                runtime['rooms'][rid]['reserved_deleted_ids']={'former_child_bed_0':old}
            for oid,typ in [('new_work_table_0','dining_table'),('new_work_chair_0','dining_chair_1')]:
                if oid in s['objects']:
                    bindings[oid]={**cat[typ]['binding'],'registry_id':cat[typ]['template']['object_id'],
                                   'identity_provenance':'research_completion_instance'}
            write(bundle/'house_state.json',house);write(bundle/'runtime.json',runtime)
            # Preserve all unique resources, including the removed bed's
            # template, so the permitted reversible edit remains recoverable.
            files={p.relative_to(bundle).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
                   for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='manifest.json'}
            mm=copy.deepcopy(m);mm['files']=files
            write(bundle/'manifest.json',mm);validate_bundle(bundle)
            native=import_project(bundle,scratch/label/'cold_restore')
            actual=native.room.to_state_dict()
            assert set(actual['objects'])==set(s['objects'])
            errors=[float(np.max(np.abs(world_transform(o)-world_transform(actual['objects'][oid])))) for oid,o in s['objects'].items()]
            assert max(errors)<1e-12
            cold=dict(bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),
                      object_ids_exact=True,maximum_world_transform_error=max(errors),reserved_removed_identity_preserved=True)
            write(root/label/'final_runtime.json',runtime)
            write(root/label/'final_house_state.json',house)
            write(root/label/'final_native_manifest.json',mm)
        scene=trimesh.Scene();cache={};visual_ids=[];struct=[]
        for oid,obj in s['objects'].items():
            if not obj['sdf_path']:continue
            visual_ids.append(oid)
            for i,(name,mesh) in enumerate(sdf_visuals(room/obj['sdf_path'],source,cache)):
                mesh.metadata.update(object_id=oid,source_visual=name,up='+Y',units='metres')
                scene.add_geometry(mesh,node_name=oid+'::'+str(i),transform=up@world_transform(obj))
        for i,(name,mesh) in enumerate(sdf_visuals(room/s['room_geometry']['sdf_path'],source,cache)):
            scene.add_geometry(mesh,node_name='structure::'+name+'::'+str(i),transform=up);struct.append(name)
        scene.metadata.update(units='metres',up='+Y',physics_ready=False,
            reconstruction='Researcher-authored lifecycle apartment; not historical surveyed Michelet')
        raw=scene.export(file_type='glb');restored=trimesh.load(io.BytesIO(raw),file_type='glb',force='scene',process=False)
        assert set(restored.graph.nodes_geometry)==set(scene.graph.nodes_geometry)
        errors=[]
        for node in scene.graph.nodes_geometry:
            t,g=scene.graph[node];mesh=scene.geometry[g].copy();mesh.apply_transform(t)
            t2,g2=restored.graph[node];again=restored.geometry[g2].copy();again.apply_transform(t2)
            assert len(mesh.faces)==len(again.faces)
            errors.append(float(np.max(np.abs(mesh.bounds-again.bounds))))
        assert max(errors)<1e-5
        path=out/(label+'.glb');path.write_bytes(raw)
        packed=out/(label+'.glb.gz')
        packed.write_bytes(gzip.compress(raw,mtime=0))
        rows.append(dict(label=label,glb_path=str(path),glb_bytes=len(raw),glb_sha256=sha(path),
            gzip_path=str(packed),gzip_bytes=packed.stat().st_size,gzip_sha256=sha(packed),
            total_objects=len(s['objects']),visual_furniture_objects=len(visual_ids),
            exported_visual_ids=visual_ids,structure_visuals=struct,
            geometry_nodes=len(scene.graph.nodes_geometry),maximum_bound_error_m=max(errors),cold_restore=cold))
        print(json.dumps(dict(label=label,gzip_bytes=packed.stat().st_size,nodes=len(scene.graph.nodes_geometry))),flush=True)
    write(root/'geometry_export_receipt.json',dict(outputs=rows,wall_seconds=time.monotonic()-started,
        material_fidelity_verified=False,geometry_is_openable=True,physics_ready=False,
        persisted=False,large_files_in_RAM=True,model_calls=0,
        identical_arm_glbs=rows[1]['glb_sha256']==rows[2]['glb_sha256']))


if __name__=='__main__':
    main()
