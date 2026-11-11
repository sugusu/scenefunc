"""Ordinary resource/mode compiler on real complete-scene tabletop geometry.

Material reservations can persist across modes while a person's seat can be
reused in nonoverlapping modes. This is ordinary scheduling and packing, not a
new learned method. Witnesses certify only the declared planar proxies.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import trimesh
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def disjoint_indices(bounds, other):
    return np.flatnonzero((bounds[:,2]<=other[0])|(bounds[:,0]>=other[2])|
                          (bounds[:,3]<=other[1])|(bounds[:,1]>=other[3]))


def three_rectangles(work,north,south):
    """Complete finite-pool disjointness using ordinary endpoint extrema.

    Filter the two meal pools against work, then use four directional extrema
    of south to decide if any north/south pair is disjoint. Every returned
    witness gets actual polygon intersection validation. No continuous UNSAT
    conclusion follows from finite candidate exhaustion.
    """
    if not work or not north or not south:
        return dict(status='not_found',finite_pool_complete=True,reason='empty candidate pool',
                    work_candidates_considered=0,pair_extreme_tests=0)
    nb=np.array([x['polygon'].bounds for x in north]);sb=np.array([x['polygon'].bounds for x in south])
    tests=0
    for wi,w in enumerate(work):
        ni=disjoint_indices(nb,w['polygon'].bounds);si=disjoint_indices(sb,w['polygon'].bounds)
        if not len(ni) or not len(si):continue
        s=sb[si];ext=[int(si[np.argmin(s[:,2])]),int(si[np.argmax(s[:,0])]),
                    int(si[np.argmin(s[:,3])]),int(si[np.argmax(s[:,1])])]
        for nidx in ni:
            a,b,c,d=nb[nidx]
            for direction,sidx in enumerate(ext):
                tests+=1;e,f,g,h=sb[sidx]
                if [g<=a,e>=c,h<=b,f>=d][direction]:
                    selected=[w,north[int(nidx)],south[sidx]]
                    assert all(x['polygon'].intersection(y['polygon']).area==0.
                               for x,y in itertools.combinations(selected,2))
                    return dict(status='witness',finite_pool_complete=True,
                        work_candidates_considered=wi+1,pair_extreme_tests=tests,
                        witnesses={name:{k:(mapping(v) if k=='polygon' else v) for k,v in obj.items()}
                                   for name,obj in zip(['persistent_work','meal_north','meal_south'],selected)})
    return dict(status='not_found',finite_pool_complete=True,
                reason='finite0/90 grid exhausted; arbitrary continuous orientation unknown',
                work_candidates_considered=len(work),pair_extreme_tests=tests)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();bundle=a.bundle.resolve()
    if (root/'compiled_programs.json').exists():raise FileExistsError('No implicit rerun')
    start=time.monotonic();p=read(root/'protocol.json')
    assert all(sha(Path(k))==h for k,h in p['source_inputs'].items())
    sys.path.insert(0,str(Path('runs/autodl/20261110_public_function_search/original_evaluator').resolve()))
    sys.path.insert(0,'/root/autodl-tmp/agent_method_preparation_20261027/work/joint_use_witness_20261104')
    from inspect_native_mesh_contacts import sdf_visuals,world_transform
    from native_project import validate_bundle
    from measure_joint_use import candidates
    meta,house,_,_=validate_bundle(bundle);rid=meta['selected_room_id'];state=house['rooms'][rid]
    room=bundle/('room_'+rid);table=state['objects'][p['common']['work_table']]
    mesh=trimesh.util.concatenate([m for _,m in sdf_visuals(room/table['sdf_path'],bundle,{})])
    mesh.apply_transform(world_transform(table));z=float(mesh.bounds[1,2]);tri=mesh.triangles
    sel=(mesh.face_normals[:,2]>.99)&(tri[:,:,2].min(1)>=z-.03)
    top=unary_union([Polygon(t[:,:2]) for t in tri[sel] if Polygon(t[:,:2]).area>1e-10])
    assert top.is_valid and not top.is_empty
    dump(root/'measured_table_surface.json',dict(surface_id=table['object_id'],z=z,
        projected_top_area_m2=float(top.area),polygon=mapping(top),bounds=mesh.bounds.tolist(),
        method='Exact projected top triangles in the same frozen3cm top band; no AABB-as-surface shortcut',
        real_human_use_validated=False))
    inverse=np.linalg.inv(world_transform(table));rows=[]
    for case in p['cases']:
        t=time.monotonic();area=float(np.prod(case['work_m'])+2*np.prod(case['meal_m']))
        # Area is a necessary condition irrespective of translations/angles.
        # If it fails, there is a sound impossibility certificate for the
        # declared disjoint planar reservations, not all real human uses.
        if area>top.area:
            shared=dict(status='infeasible_under_disjoint_planar_contract',
                certificate=dict(kind='necessary_area_bound',required_m2=area,available_m2=float(top.area),
                                 excess_m2=area-float(top.area)),candidate_packing_run=False)
            pool_counts=None
        else:
            w=candidates(top,tuple(case['work_m']),table['object_id'],z)
            meal=candidates(top,tuple(case['meal_m']),table['object_id'],z)
            north=[x for x in meal if (inverse@np.array([*x['center_xy'],z,1]))[1]>=0]
            south=[x for x in meal if (inverse@np.array([*x['center_xy'],z,1]))[1]<=0]
            shared=three_rectangles(w,north,south);shared['candidate_packing_run']=True
            pool_counts=dict(work=len(w),meal_north=len(north),meal_south=len(south))
        # One extra copy of this same tabletop can support work independently;
        # compute its real surface capacity, do not infer from total area.
        independent_work=candidates(top,tuple(case['work_m']),table['object_id'],z)
        meals=candidates(top,tuple(case['meal_m']),table['object_id'],z)
        n=[x for x in meals if (inverse@np.array([*x['center_xy'],z,1]))[1]>=0]
        s=[x for x in meals if (inverse@np.array([*x['center_xy'],z,1]))[1]<=0]
        # Using a zero-area distant reservation reuses the same finite-pool
        # exact N/S test; it does not consume any real table region.
        anchor={'surface_id':'separate_table','polygon':Polygon([(100,100),(101,100),(101,101),(100,101)])}
        mealpair=three_rectangles([anchor],n,s)
        independent_possible=bool(independent_work and mealpair['status']=='witness')
        program=dict(modes=p['common']['mode_people'],
            material_reservation_active_in=p['common']['persistent_materials'],
            surface_occupancy_conflicts=[['persistent_work','meal_north'],['persistent_work','meal_south'],['meal_north','meal_south']],
            seat_use=dict(work={'resident':p['common']['work_seat']},
                          meal={'resident':p['common']['meal_seats'][0],'guest':p['common']['meal_seats'][1]}),
            chair_reuse_across_nonoverlapping_modes=True,
            seat_concurrent_capacity_claim=False,work_people_during_meal=0)
        row=dict(id=case['id'],kind=case['kind'],work_m=case['work_m'],meal_m=case['meal_m'],
            required_area_m2=area,shared=shared,pool_counts=pool_counts,
            independent_top_capacity_verified=independent_possible,
            independent_work_witness=({k:(mapping(v) if k=='polygon' else v) for k,v in independent_work[0].items()} if independent_work else None),
            independent_meal_pair=mealpair,activity_program=program,
            new_furniture_if_shared=0,new_furniture_for_cached_independent_solution=2,
            native_scene_unchanged=True,new_physical_material_objects_created=False,
            cpu_wall_seconds=time.monotonic()-t,full_scene_success=None)
        rows.append(row);print(json.dumps({k:row[k] for k in ['id','shared','pool_counts','independent_top_capacity_verified','cpu_wall_seconds']}),flush=True)
    dump(root/'compiled_programs.json',dict(rows=rows,wall_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,
        training_updates=0,native_geometry_modifications=0,ordinary_method=True,
        evidence_scope='Explicit temporal-resource and planar footprint contract, not original preference or complete ergonomic validity'))
    assert all(sha(Path(k))==h for k,h in p['source_inputs'].items())


if __name__=='__main__':main()
