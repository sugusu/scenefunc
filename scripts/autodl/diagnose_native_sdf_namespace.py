"""Actual same-state Drake comparison of SDF namespace spelling, zero models.

This is a compatibility diagnosis, not a new policy, search or scene result.
Source inputs are immutable. Only restored SDF copies have their XML namespace
prefix changed; expanded XML names and all coordinates/meshes stay identical.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths,sha


def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
def expanded_tree(element):
    return (element.tag,sorted(element.attrib.items()),element.text,element.tail,[expanded_tree(c) for c in element])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    stage=a.stage.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    overlay=stage/'runtime';native=Path(json.loads((stage/'runtime_overlay_manifest.json').read_text())['native_root'])
    for p in (overlay,overlay/'src'):sys.path.insert(0,str(p))
    os.environ.update(SCENESMITH_ROOT=str(native),PYTHONDONTWRITEBYTECODE='1',OPENAI_AGENTS_DISABLE_TRACING='1')
    blocked=[]
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'):
            blocked.append(event);raise RuntimeError('No models, services or subprocesses in SDF diagnostic')
    sys.addaudithook(audit)
    started=time.monotonic();result=dict(status='started',model_forwards=0,paid_API_calls=0,training_steps=0,
                                       scene_coordinates_modified=False,whole_scene_success=None)
    try:
        importlib.import_module('src.runtime')
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.drake_utils import create_drake_plant_and_scene_graph_from_scene
        from scenesmith.agent_utils.physics_validation import compute_scene_collisions
        from pydrake.all import DiagramBuilder
        with tempfile.TemporaryDirectory(prefix='native_sdf_namespace_',dir='/dev/shm') as scratch:
            root=Path(scratch)/'restored'
            proof=restore_solid_evidence(stage/'evidence_solid_v2/receipt.json',root)
            relocation=relocate_restored_paths(root,proof['original_root'])
            dump(out/'restore_proof.json',dict(**proof,path_relocations=relocation))
            state=json.loads((root/'last_actual_room_state.json').read_text())
            scene=RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom');scene.restore_from_state_dict(state)
            initial_objects=scene.to_state_dict()['objects']
            def measure(label):
                builder=DiagramBuilder()
                plant,graph=create_drake_plant_and_scene_graph_from_scene(scene,builder=builder,weld_furniture=False,
                                                                         free_mounted_objects_for_collision=True)
                diagram=builder.Build();context=diagram.CreateDefaultContext()
                query=graph.get_query_output_port().Eval(graph.GetMyContextFromRoot(context));inspector=query.inspector()
                types={};geometries=[]
                from pydrake.geometry import Role
                for gid in inspector.GetAllGeometryIds(Role.kProximity):
                    shape=inspector.GetShape(gid);kind=type(shape).__name__;types[kind]=types.get(kind,0)+1
                    geometries.append(dict(name=inspector.GetName(gid),shape=kind))
                contacts=[dict(a=inspector.GetName(r.id_A),b=inspector.GetName(r.id_B),signed_distance=r.distance)
                          for r in query.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.)]
                # Invoke the same original reporting function, separately from
                # raw contact enumeration; no threshold is changed.
                reported=[vars(x) for x in compute_scene_collisions(scene)]
                dump(out/(label+'.json'),dict(types=types,geometries=geometries,raw_pairs=contacts,
                                              original_native_reported_pairs=reported,bodies=plant.num_bodies()))
                return dict(shape_types=types,raw_pairs=len(contacts),native_reported_pairs=len(reported),
                            maximum_raw_penetration=max([-x['signed_distance'] for x in contacts],default=0.))
            before=measure('original_prefix')
            changed=[]
            for path in root.rglob('*.sdf'):
                data=path.read_bytes()
                if b'xmlns:ns0="drake.mit.edu"' not in data:continue
                rewritten=data.replace(b'xmlns:ns0="drake.mit.edu"',b'xmlns:drake="drake.mit.edu"').replace(b'ns0:',b'drake:')
                assert expanded_tree(ET.fromstring(data))==expanded_tree(ET.fromstring(rewritten))
                path.write_bytes(rewritten)
                changed.append(dict(path=str(path.relative_to(root)),before_sha256=hashlib.sha256(data).hexdigest(),
                                    after_sha256=sha(path),expanded_XML_identical=True))
            after=measure('drake_prefix')
            dump(out/'SDF_prefix_changes.json',changed)
            assert scene.to_state_dict()['objects']==initial_objects
            result.update(status='same_scene_namespace_comparison_executed',before=before,after=after,
                          changed_SDF_files=len(changed),expanded_XML_preserved=True,source_trial_reclassified_as_success=False,
                          not_a_method_quality_comparison=True)
    except BaseException as exc:
        import traceback
        result.update(status='diagnostic_failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        result.update(seconds=time.monotonic()-started,blocked_operations=blocked)
        dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
