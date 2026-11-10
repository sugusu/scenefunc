"""Cold parse exact archived floor geometry in RAM using Drake, no model/render.

Original default-material images are companion dependencies, not silently
assumed to be embedded. Original JSON absolute strings are not reinterpreted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(stage):
    stage=stage.resolve();out=stage/'packaged_geometry_cold_read.json'
    if out.exists():raise FileExistsError(out)
    archive=stage/'native_floor_geometry.zip'
    receipt=json.loads((stage/'actual_floor_geometry_v1/packaging_receipt.json').read_text())
    assert sha(archive)==receipt['archive_sha256']
    blocked=[]
    def audit(event,args):
        if event in ('socket.connect','subprocess.Popen'):
            blocked.append(event);raise RuntimeError('No services/model/subprocess/network in cold geometry read')
    sys.addaudithook(audit)
    started=time.monotonic();result=dict(model_calls=0,API_calls=0,training_steps=0,new_scene_rendered=False)
    with tempfile.TemporaryDirectory(prefix='native_floor_replay_',dir='/dev/shm') as scratch:
        bundle=Path(scratch)/'bundle';geometry=bundle/'actual_floor_geometry_v1'
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('paths.json'))
            for row in manifest['files']:
                rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
                data=z.read('blobs/'+row['sha256'])
                assert hashlib.sha256(data).hexdigest()==row['sha256']
                dest=geometry/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        materials=[]
        for source in sorted((stage/'runtime/native_source/materials/Wood094_1K-JPG').glob('*.jpg')):
            dest=bundle/'runtime/native_source/materials/Wood094_1K-JPG'/source.name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            materials.append(dict(source=str(source),sha256=sha(source),bytes=source.stat().st_size))
        assert len(materials)==3
        from pydrake.all import DiagramBuilder,AddMultibodyPlantSceneGraph,Parser
        builder=DiagramBuilder();plant,scene_graph=AddMultibodyPlantSceneGraph(builder,time_step=0.)
        sdf=geometry/'scene/room_geometry/room_geometry_bedroom.sdf'
        models=Parser(plant).AddModels(str(sdf));plant.Finalize()
        result.update(status='archive_restored_and_geometry_cold_parsed',archive_sha256=sha(archive),
            exact_restored_files=len(manifest['files']),native_models=len(models),
            drake_num_bodies=plant.num_bodies(),drake_plant_finalized=plant.is_finalized(),
            material_companion_dependencies=materials,archive_is_not_alone_texture_self_contained=True,
            original_absolute_JSON_strings_not_used_as_relocated_scene=True)
    result.update(seconds=time.monotonic()-started,temporary_restore_removed=True,blocked_operations=blocked)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    main(p.parse_args().stage)
