"""Use the original native collision reporter on one unchanged archived room."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths
from native_penetration_query_adapter import native_penetration_query
from conservative_contact_query import native_conservative_contact_query


def dump(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')


def main(repo,stage,backend):
    out=stage/('same_room_control_v1' if backend=='point' else 'same_room_veto_v1')
    out.mkdir(parents=True,exist_ok=False)
    prior=repo/'runs/autodl/20261110_complete_native_runtime'
    overlay=prior/'runtime';native=Path(json.loads((prior/'runtime_overlay_manifest.json').read_text())['native_root'])
    for p in (overlay,overlay/'src'):sys.path.insert(0,str(p))
    os.environ.update(SCENESMITH_ROOT=str(native),PYTHONDONTWRITEBYTECODE='1',OPENAI_AGENTS_DISABLE_TRACING='1')
    blocked=[]
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'):
            blocked.append(event);raise RuntimeError('No models/network/subprocess in same-state control')
    sys.addaudithook(audit)
    def deadline(sig,frame):raise TimeoutError('Fixed180s same-room contact control')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
    tick=time.monotonic();result=dict(status='started',model_forwards=0,paid_API_calls=0,
        training_updates=0,scene_coordinates_modified=False,original_scores_modified=False,
        whole_scene_success=None,not_a_learned_method=True)
    try:
        importlib.import_module('src.runtime')
        from scenesmith.agent_utils.room import RoomScene
        from scenesmith.agent_utils.drake_utils import create_drake_plant_and_scene_graph_from_scene
        from pydrake.all import DiagramBuilder
        physics=importlib.import_module('scenesmith.agent_utils.physics_validation')
        original_source=Path(physics.__file__);before_hash=hashlib.sha256(original_source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='native_contact_feedback_',dir='/dev/shm') as scratch:
            root=Path(scratch)/'restored'
            proof=restore_solid_evidence(prior/'evidence_solid_v2/receipt.json',root)
            relocate_restored_paths(root,proof['original_root'])
            scene=RoomScene(room_geometry=None,scene_dir=root/'scene/room_bedroom')
            scene.restore_from_state_dict(json.loads((root/'last_actual_room_state.json').read_text()))
            initial=json.dumps(scene.to_state_dict()['objects'],sort_keys=True)
            original=[vars(x) for x in physics.compute_scene_collisions(scene)]
            contacts=[]
            hook=native_penetration_query(physics,contacts) if backend=='point' else native_conservative_contact_query(physics,contacts,threshold=.001)
            with hook:
                revised=[vars(x) for x in physics.compute_scene_collisions(scene)]
            # Third query is independent of the reporter/proxy. It checks exact
            # same-state native geometry against the newly used engine operation.
            direct=None
            if backend=='point':
                builder=DiagramBuilder();plant,graph=create_drake_plant_and_scene_graph_from_scene(
                    scene,builder=builder,weld_furniture=False,free_mounted_objects_for_collision=True)
                diagram=builder.Build();context=diagram.CreateDefaultContext()
                query=graph.get_query_output_port().Eval(graph.GetMyContextFromRoot(context))
                inspector=query.inspector();raw=query.ComputePointPairPenetration()
                direct=[dict(a=inspector.GetName(x.id_A),b=inspector.GetName(x.id_B),depth_m=float(x.depth),
                             p_WCa=x.p_WCa.tolist(),p_WCb=x.p_WCb.tolist()) for x in raw]
                key=lambda x:(x['a'],x['b'],x['depth_m'])
                assert sorted(contacts,key=key)==sorted(direct,key=key)
            assert json.dumps(scene.to_state_dict()['objects'],sort_keys=True)==initial
            assert hashlib.sha256(original_source.read_bytes()).hexdigest()==before_hash
            dump(out/'original_native_report.json',original)
            dump(out/'alternative_native_report.json',revised)
            dump(out/'backend_receipts.json',contacts)
            if direct is not None:dump(out/'actual_point_contacts.json',direct)
            result.update(status='original_native_reporter_same_state_control_completed',
                backend=backend,
                original_reported_pairs=len(original),alternative_reported_pairs=len(revised),
                original_report=original,alternative_report=revised,
                raw_point_contacts=len(direct) if direct is not None else None,
                new_engine_query_calls=3 if backend=='point' else 2,
                original_defaults=dict(penetration_threshold_m=.001,
                    floor_penetration_tolerance_m=.05,manipuland_furniture_tolerance_m=.02),
                native_source_sha256=before_hash,original_source_unchanged=True,
                same_original_object_IDs_poses_scales_boxes=True,
                direct_query_matches_reporter_backend=True if backend=='point' else None,
                source_archive_SHA256=proof['archive_sha256'],
                qualification='One original native partial room and the fixed2body diagnostics only; not a general physics verifier or revised scene-success label.')
    except BaseException as exc:
        import traceback
        result.update(status='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0);result.update(seconds=time.monotonic()-tick,blocked_operations=blocked)
        dump(out/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('traceback','original_report','alternative_report')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--backend',choices=['point','conservative'],default='point')
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve(),a.backend)
