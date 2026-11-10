"""Two-body CPU query diagnosis with one fixed actual collision part.

Controlled vertical offsets are diagnostic states, never proposed room edits or
new generation scores. Original source/geometry/scene evidence stays immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import time

from native_trial_evidence import restore_solid_evidence,relocate_restored_paths


def dump(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')


def main(repo,stage):
    out=stage/'two_body_v1';out.mkdir(parents=True,exist_ok=False)
    prior=repo/'runs/autodl/20261110_complete_native_runtime'
    contacts_path=prior/'partial_replay_v2/actual_drake_contacts.json'
    contacts=json.loads(contacts_path.read_text())
    certificate=json.loads((prior/'floor_halfspace_certificate_v1/result.json').read_text())
    certified={(x['object_id'],x['collision_name']) for x in certificate['rows']
               if x['disjoint_interiors_by_Z_plane'] and x['actual_query_distance']<-.001}
    chosen=None
    for pair in contacts:
        if pair['a']['name']=='room_geometry::floor_collision':floor,other=pair['a'],pair['b']
        elif pair['b']['name']=='room_geometry::floor_collision':floor,other=pair['b'],pair['a']
        else:continue
        identity=tuple(other['name'].split('::',1))
        if identity in certified:
            chosen=(pair,floor,other,identity);break
    if chosen is None:raise ValueError('No fixed archived contradiction; no substitute')
    pair,floor,other,(oid,part_name)=chosen
    frames=json.loads((prior/'asset_collision_frame_audit/geometry_bounds.json').read_text())
    obj=next(x for x in frames['objects'] if x['object_id']==oid)
    part=next(x for x in obj['parts'] if x['name']==part_name)
    protocol=json.loads((stage/'PROTOCOL.json').read_text())
    dump(out/'SELECTED_PAIR.json',dict(original_contact=pair,original_mesh=part,
         original_contact_file_sha256=hashlib.sha256(contacts_path.read_bytes()).hexdigest()))
    blocked=[]
    def guard(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'):
            blocked.append(event);raise RuntimeError('No models/network/subprocess in contact diagnosis')
    sys.addaudithook(guard)
    def deadline(sig,frame):raise TimeoutError('Fixed180s contact query diagnosis')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
    started=time.monotonic();records=[]
    result=dict(status='started',model_forwards=0,training_updates=0,paid_API_calls=0,
                scene_coordinates_modified=False,original_scores_modified=False,
                whole_scene_success=None,not_a_learned_method=True)
    try:
        import numpy as np
        from pydrake.all import (DiagramBuilder,AddMultibodyPlantSceneGraph,
            RigidTransform,Box,Mesh,Convex,CoulombFriction,SpatialInertia,UnitInertia)
        with tempfile.TemporaryDirectory(prefix='actual_contact_query_',dir='/dev/shm') as scratch:
            root=Path(scratch)/'restored'
            proof=restore_solid_evidence(prior/'evidence_solid_v2/receipt.json',root)
            mesh=root/part['mesh_path']
            assert hashlib.sha256(mesh.read_bytes()).hexdigest()==part['sha256']
            mesh_original=mesh.read_bytes()
            floor_shape=floor['shape'];assert floor_shape['type']=='Box'
            for representation,shape_class in [('Mesh',Mesh),('Convex',Convex)]:
                for delta in protocol['two_body_offsets_m']:
                    shape=shape_class(str(mesh),float(other['shape']['scale']))
                    builder=DiagramBuilder();plant,graph=AddMultibodyPlantSceneGraph(builder,time_step=0.)
                    friction=CoulombFriction(.5,.5)
                    gf=plant.RegisterCollisionGeometry(plant.world_body(),RigidTransform(np.asarray(floor['X_WG'])),
                        Box(floor_shape['width'],floor_shape['depth'],floor_shape['height']),'actual_floor',friction)
                    body=plant.AddRigidBody('actual_collision_part',SpatialInertia(1.,np.zeros(3),UnitInertia.SolidSphere(1.)))
                    gp=plant.RegisterCollisionGeometry(body,RigidTransform(),shape,'actual_part',friction)
                    plant.Finalize();diagram=builder.Build();context=diagram.CreateDefaultContext()
                    pose=np.asarray(other['X_WG'],dtype=float);pose[2,3]+=delta
                    plant.SetFreeBodyPose(plant.GetMyMutableContextFromRoot(context),body,RigidTransform(pose))
                    query=graph.get_query_output_port().Eval(graph.GetMyContextFromRoot(context))
                    tick=time.monotonic()
                    signed=query.ComputeSignedDistancePairClosestPoints(gf,gp)
                    penetrations=query.ComputePointPairPenetration()
                    hull=shape.GetConvexHull()
                    vertices=np.asarray([hull.vertex(i) for i in range(hull.num_vertices())])
                    # Drake's hull is returned with the shape's actual scale.
                    world=vertices @ pose[:3,:3].T + pose[:3,3]
                    floor_top=floor['X_WG'][2][3]+floor_shape['height']/2
                    records.append(dict(representation=representation,vertical_offset_m=delta,
                        signed_distance_m=float(signed.distance),
                        signed_p_ACa=signed.p_ACa.tolist(),signed_p_BCb=signed.p_BCb.tolist(),
                        signed_nhat_BA_W=signed.nhat_BA_W.tolist(),
                        penetration_contacts=[dict(depth_m=float(x.depth),p_WCa=x.p_WCa.tolist(),p_WCb=x.p_WCb.tolist()) for x in penetrations],
                        actual_Drake_convex_hull_world_min=world.min(axis=0).tolist(),
                        actual_Drake_convex_hull_world_max=world.max(axis=0).tolist(),floor_top=floor_top,
                        exact_Z_separation=float(world[:,2].min())>=floor_top,
                        seconds=time.monotonic()-tick))
                    dump(out/'query_records.json',records)
            assert mesh.read_bytes()==mesh_original
            result.update(status='fixed_actual_two_body_queries_completed',
                 original_mesh_sha256=part['sha256'],selected_object=oid,selected_part=part_name,
                 engine_query_calls=2*len(records),source_archive_SHA256=proof['archive_sha256'],
                 original_archived_signed_distance=pair['signed_distance'],records=records)
    except BaseException as exc:
        import traceback
        result.update(status='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),records=records)
    finally:
        signal.alarm(0);result.update(seconds=time.monotonic()-started,blocked_operations=blocked)
        dump(out/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','traceback')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
