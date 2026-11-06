"""Declared static activity proxies for arbitrary layouts, including shared tables.

No reference poses or expected furniture names/IDs are used to find new work
places. A failed finite grid search is not a proof that all use is impossible.
"""
import itertools
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from inspect_native_mesh_contacts import world_transform
from measure_room_access import measure
from native_project import validate_bundle


def evaluate_workspace(plan, root, state, initial, meshes, inventory, settings):
    spec=plan['activity'];names={r['object_id']:r['name'] for r in inventory}
    table_ids=[oid for oid,name in names.items() if name in spec['table_names']]
    chair_ids=[oid for oid,name in names.items() if name in spec['chair_names']]
    retained={oid:oid in state['objects'] for oid,obj in initial['objects'].items() if obj['object_type']=='furniture'}
    rid=validate_bundle(root)[0]['selected_room_id']
    _,_,old_runtime,_=validate_bundle(Path(plan['initial_bundle']))
    _,_,new_runtime,_=validate_bundle(root)
    old_bind=old_runtime['rooms'][rid]['asset_bindings'];new_bind=new_runtime['rooms'][rid]['asset_bindings']
    same={oid:retained[oid] and old_bind[oid]==new_bind[oid] and all(
        state['objects'][oid][k]==initial['objects'][oid][k] for k in ('bbox_min','bbox_max','scale_factor')) for oid in retained}
    targets={};missing=[]
    for label,role in spec['roles'].items():
        oid=role['object_id']
        if oid is None:targets[label]=role['initial_xy'];continue
        if oid not in state['objects']:missing.append(label);continue
        point=world_transform(state['objects'][oid])@np.linalg.inv(world_transform(initial['objects'][oid]))@np.array([*role['initial_xy'],0.,1.])
        targets[label]=point[:2].tolist()
    for cid in chair_ids:
        transform=world_transform(state['objects'][cid])
        for side,xy in spec['seat_approach_offsets'].items():
            targets[f'{cid}:{side}']=(transform@np.array([*xy,0.,1.]))[:2].tolist()
    access=measure(state,root/('room_'+rid),root,{**settings,'approach_points_xy_m':targets})
    per_width={s['square_body_width_m']:s for s in access['scenarios']}
    surfaces={};patches={};candidates=[];surface_stats=[]
    for tid in table_ids:
        mesh=meshes[tid];top=float(mesh.bounds[1,2]);tri=mesh.triangles
        select=(mesh.face_normals[:,2]>.99)&(tri[:,:,2].min(axis=1)>=top-.03)
        surface=unary_union([Polygon(t[:,:2]) for t in tri[select] if Polygon(t[:,:2]).area>1e-10]);surfaces[tid]=surface
        surface_stats.append(dict(object_id=tid,height_m=top,area_m2=float(surface.area)))
        if not spec['table_height_range_m'][0]<=top<=spec['table_height_range_m'][1]:continue
        obj=state['objects'][tid];transform=world_transform(obj);lo=np.array(obj['bbox_min']);hi=np.array(obj['bbox_max'])
        w,h=spec['patch_m'];grid=spec['patch_grid_m'];patches[tid]=[]
        xs=np.unique(np.r_[np.arange(lo[0]+w/2,hi[0]-w/2+1e-8,grid),(lo[0]+hi[0])/2])
        ys=np.unique(np.r_[np.arange(lo[1]+h/2,hi[1]-h/2+1e-8,grid),(lo[1]+hi[1])/2])
        for x,y in itertools.product(xs,ys):
            corners=[(transform@np.array([x+dx,y+dy,0.,1.]))[:2] for dx,dy in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]]
            patch=Polygon(corners)
            if surface.buffer(1e-8).covers(patch):patches[tid].append(patch)
        for cid in chair_ids:
            cm=world_transform(state['objects'][cid]);cc=meshes[cid].bounds[:,:2].mean(0)
            choices=[]
            for patch in patches[tid]:
                center=np.array(patch.centroid.coords[0]);delta=center-cc;distance=float(np.linalg.norm(delta))
                if distance<1e-8 or distance>spec['seat_patch_distance_max_m']:continue
                facing=float(cm[:2,1]@delta/distance)
                if facing<spec['seat_toward_patch_cos_min']:continue
                choices.append((distance,patch))
            # A documented finite coverage limit, identical across arms.
            for distance,patch in sorted(choices,key=lambda x:x[0])[:spec['max_patches_per_table_seat']]:
                candidates.append(dict(table_id=tid,chair_id=cid,patch=patch,distance_m=distance))
    def clear_seat(cid,width):
        cc=meshes[cid].bounds[:,:2].mean(0);square=box(*(cc-width/2),*(cc+width/2))
        for oid,m in meshes.items():
            if oid==cid or oid not in names:continue
            if square.intersection(box(*m.bounds[0,:2],*m.bounds[1,:2])).area>1e-8:return False
        s=per_width[width]
        return any(s['targets'][f'{cid}:{side}']['reachable_from_entry'] for side in spec['seat_approach_offsets'])
    def independent(a,b,width):
        if a['chair_id']==b['chair_id'] or a['patch'].intersection(b['patch']).area>1e-8:return False
        c1=meshes[a['chair_id']].bounds[:,:2].mean(0);c2=meshes[b['chair_id']].bounds[:,:2].mean(0)
        return box(*(c1-width/2),*(c1+width/2)).intersection(box(*(c2-width/2),*(c2+width/2))).area<=1e-8
    def serial(c):return dict(table_id=c['table_id'],chair_id=c['chair_id'],patch_xy=list(c['patch'].exterior.coords),seat_patch_distance_m=c['distance_m'])
    scenarios=[]
    all_patches=[(tid,p) for tid,values in patches.items() for p in values]
    for width,scenario in per_width.items():
        valid_chairs={cid for cid in chair_ids if clear_seat(cid,width)}
        usable=[c for c in candidates if c['chair_id'] in valid_chairs]
        witness=None
        for a,b in itertools.combinations(usable,2):
            if not independent(a,b,width):continue
            if spec['task']=='Pine':
                witness=dict(two_simultaneous_work_places=[serial(a),serial(b)]);break
            # In Milan work materials stay while two people dine. The work seat
            # may be reused at mealtime; its material patch must remain disjoint.
            for work in usable:
                if work['patch'].intersection(a['patch']).area<=1e-8 and work['patch'].intersection(b['patch']).area<=1e-8:
                    witness=dict(two_dining_places=[serial(a),serial(b)],persistent_work_materials=serial(work));break
            if witness:break
        scenarios.append(dict(width_m=width,activity_proxy_pass=witness is not None,witness=witness,
            valid_seat_ids=sorted(valid_chairs),role_access={k:scenario['targets'][k]['reachable_from_entry'] if k in scenario['targets'] else False for k in spec['roles']},missing_roles=missing))
    return dict(retained_ids=retained,identity_asset_dimensions_preserved=same,surface_stats=surface_stats,
                candidate_count=len(candidates),scenarios=scenarios,role_and_seat_access=access,
                primary_width_m=.5,whole_room_human_use='unknown',professional_review=False,
                limits=['Declared static patches, square seats and route proxies; not ergonomics or capacity.',
                        'Finite patch grid and per-table-seat cap can miss usable arrangements; no witness is not universal infeasibility.',
                        'Sofa seating, articulated furniture, guest sleep, real comfort and facilities remain unknown.',
                        'Milan permits dining-table sharing with disjoint work materials; Pine permits two places on one table.'])
