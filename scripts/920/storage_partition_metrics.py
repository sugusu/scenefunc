"""Task-specific proxies for a declared storage-partition transfer.

Uses actual transformed visual triangles for finite sight segments. The reused
workspace surface/seat routine measures two patches; they are explicitly dining
proxies here, never a claim of privacy, real storage capacity or human comfort.
"""
import copy
import itertools
import numpy as np
import trimesh
from edit_workspace_metrics import evaluate_workspace
from inspect_native_mesh_contacts import world_transform


def segment_hits(mesh, origin, target):
    delta=target-origin;length=float(np.linalg.norm(delta))
    if length<=1e-9:raise ValueError('Degenerate sight segment')
    locations,_,faces=mesh.ray.intersects_location([origin],[delta/length],multiple_hits=True)
    hits=[]
    for point,face in zip(locations,faces):
        distance=float((point-origin)@(delta/length))
        if 1e-5<distance<length-1e-5:
            hits.append(dict(distance_m=distance,point_m=point.tolist(),face_index=int(face)))
    return sorted(hits,key=lambda h:h['distance_m'])


def evaluate_storage_partition(plan,root,state,initial,meshes,inventory,settings):
    spec=plan['partition'];local=copy.deepcopy(plan)
    # Two disjoint .40 x .30m top-surface patches and two independently usable
    # seats, exactly the existing general static two-place search.
    local['activity']['task']='Pine'
    result=evaluate_workspace(local,root,state,initial,meshes,inventory,settings)
    for s in result['scenarios']:
        if s['witness']:
            s['witness']['two_dining_places']=s['witness'].pop('two_simultaneous_work_places')
        s['two_dining_place_proxy_pass']=s.pop('activity_proxy_pass')
    missing=[oid for oid in spec['required_ids'] if oid not in state['objects']]
    rays=[]
    if spec['observer_id'] not in missing and spec['target_id'] not in missing:
        observer_transform=world_transform(state['objects'][spec['observer_id']])
        target_transform=world_transform(state['objects'][spec['target_id']])
        for eye,target in itertools.product(spec['observer_local_points_m'],spec['target_local_points_m']):
            a=trimesh.transform_points([eye],observer_transform)[0]
            b=trimesh.transform_points([target],target_transform)[0]
            blockers=[]
            for oid,mesh in meshes.items():
                if oid in (spec['observer_id'],spec['target_id']):continue
                hits=segment_hits(mesh,a,b)
                if hits:blockers.append(dict(object_id=oid,first_hit=hits[0]))
            rays.append(dict(origin_m=a.tolist(),target_m=b.tolist(),clear=not blockers,blockers=blockers))
    visible=sum(r['clear'] for r in rays)
    total=len(spec['observer_local_points_m'])*len(spec['target_local_points_m'])
    metric=dict(sampled_segments=total,evaluated_segments=len(rays),clear_segments=visible,
                blocked_segments=len(rays)-visible,rays=rays,
                screen_condition_met=len(rays)==total and visible==0,
                open_condition_met=len(rays)==total and visible==total,
                real_privacy='unknown',two_sided_storage='not_implemented',occluder_model='actual visual mesh triangles, all treated opaque')
    for scenario in result['scenarios']:
        scenario.update(all_required_ids_preserved=not missing and all(result['identity_asset_dimensions_preserved'].values()),
            all_required_approaches_reachable=all(scenario['role_access'].values()) and not scenario['missing_roles'],
            visible_segments=visible,total_segments=total,
            visual_requirement=spec['condition'],visual_proxy_pass=metric[spec['condition']+'_condition_met'])
        scenario['combined_declared_activity_proxy_pass']=all(scenario[k] for k in (
            'all_required_ids_preserved','all_required_approaches_reachable','two_dining_place_proxy_pass','visual_proxy_pass'))
    result.update(visibility=metric,scope='entire represented sleeping/living/dining room, not full source dwelling',
        limits=['All dimensions, demands and sampling positions are research assumptions.',
                'Finite mesh visibility does not establish human privacy; mesh opacity assumed.',
                'Single-sided ordinary shelves do not reproduce double-access custom joinery.',
                'Static meal patches, square seats and conservative routes do not establish ergonomics.',
                'Shelf capacity, structural stability, doors, continuous moving paths and actual use unknown.'])
    return result
