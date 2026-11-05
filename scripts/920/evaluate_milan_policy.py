"""Object-relative living/work diagnostics for arbitrary model-produced rooms."""
import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from native_project import validate_bundle, sha256
from inspect_native_mesh_contacts import sdf_visuals, world_transform
from measure_room_access import measure, obstacles
from measure_family_choice import hit_segment


def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();spec=read(a.plan)
    for name,h in spec['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h
    original=Path(spec['initial_bundle']);assert sha256(original/'manifest.json')==spec['initial_manifest_sha256']
    im,ih,ir,_=validate_bundle(original);fm,fh,fr,_=validate_bundle(a.bundle);rid=im['selected_room_id'];assert rid==fm['selected_room_id'];before=ih['rooms'][rid];after=fh['rooms'][rid];room=a.bundle/('room_'+rid)
    print(datetime.now(timezone.utc).isoformat(),'evaluate model room with object-relative requirements',flush=True)
    initial_furniture={k:v for k,v in before['objects'].items() if v['object_type']=='furniture'}
    retained={k:k in after['objects'] for k in initial_furniture}
    same_identity={k:retained[k] and fr['rooms'][rid]['asset_bindings'][k]==ir['rooms'][rid]['asset_bindings'][k] and all(after['objects'][k][f]==v[f] for f in ['bbox_min','bbox_max','scale_factor']) for k,v in initial_furniture.items()}
    immutable={k:after['objects'].get(k)==v for k,v in before['objects'].items() if v['immutable']};immutable['room_geometry']=after['room_geometry']==before['room_geometry']
    targets={};missing_targets=[]
    for label,item in spec['role_approaches'].items():
        oid=item.get('object_id')
        if oid is None:targets[label]=item['initial_xy'];continue
        if oid not in after['objects']:missing_targets.append(label);continue
        old=world_transform(before['objects'][oid]);new=world_transform(after['objects'][oid]);point=new@np.linalg.inv(old)@np.array([*item['initial_xy'],0.,1.]);targets[label]=point[:2].tolist()
    settings={**spec['access'],'approach_points_xy_m':targets};access=measure(after,room,a.bundle,settings)
    role_results=[dict(width_m=s['square_body_width_m'],planned=len(spec['role_approaches']),reachable=sum(x['reachable_from_entry'] for x in s['targets'].values()),failed=[k for k,x in s['targets'].items() if not x['reachable_from_entry']],missing=missing_targets) for s in access['scenarios']]
    cache={};meshes={};surfaces={};asset_names={x['binding']['asset_id']:x['template']['name'] for x in read(a.bundle/'catalog.json')['entries']};kinds={}
    for oid,obj in after['objects'].items():
        if obj['object_type']!='furniture':continue
        m=trimesh.util.concatenate([m for _,m in sdf_visuals(room/obj['sdf_path'],a.bundle,cache)]);m.apply_transform(world_transform(obj));meshes[oid]=m;kinds[oid]=asset_names.get(fr['rooms'][rid]['asset_bindings'][oid]['asset_id'],'unknown')
        if kinds[oid] in spec['table_asset_names']:
            top=float(m.bounds[1,2]);tri=m.triangles;sel=(m.face_normals[:,2]>.99)&(tri[:,:,2].min(axis=1)>=top-.03)
            surfaces[oid]=unary_union([Polygon(t[:,:2]) for t in tri[sel] if Polygon(t[:,:2]).area>1e-10])
    pairs=[];blocked=obstacles(after,room,a.bundle,1.8)
    for tid,surface in surfaces.items():
        desk=meshes[tid];center=desk.bounds[:,:2].mean(axis=0);w,h=spec['material_patch_m'];rot=world_transform(after['objects'][tid])[:2,:2]
        patch=Polygon([center+rot@np.array([x,y]) for x,y in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]])
        for cid in [k for k in kinds if kinds[k] in spec['chair_asset_names']]:
            chair=after['objects'][cid];cm=world_transform(chair);delta=center-cm[:2,3];distance=float(np.linalg.norm(delta))
            if distance<1e-8 or distance>spec['pair_center_distance_max_m']:continue
            facing=float(np.dot(cm[:2,1],delta)/distance)
            proposed={name:(cm@np.array([*xy,0.,1.]))[:2].tolist() for name,xy in spec['chair_approach_offsets_m'].items()}
            pa=measure(after,room,a.bundle,{**spec['access'],'approach_points_xy_m':proposed})
            reachable=[dict(width_m=s['square_body_width_m'],reachable_sides=[k for k,v in s['targets'].items() if v['reachable_from_entry']]) for s in pa['scenarios']]
            rays=[];facing_cosines=[]
            if spec['sofa_id'] in after['objects']:
                sm=world_transform(after['objects'][spec['sofa_id']]);direction=sm[:2,3]-cm[:2,3]
                if np.linalg.norm(direction)>1e-8:facing_cosines.append(float(cm[:2,1]@direction/np.linalg.norm(direction)))
                for z in spec['ray_heights_m']:
                    for offset in spec['sofa_offsets_x_m']:
                        origin=np.array([cm[0,3],cm[1,3],z]);target=(sm@np.array([offset,0,z,1.]))[:3]
                        hits=[b['id'] for b in blocked if hit_segment(origin,target,np.array(b['lo']),np.array(b['hi']))]
                        rays.append(dict(origin=origin.tolist(),target=target.tolist(),clear=not hits,blockers=hits))
            meal=surfaces.get(spec['meal_table_id']);meal_overlap=None if meal is None else float(patch.intersection(meal).area)
            independent=tid!=spec['meal_table_id'] and cid not in spec['meal_seat_ids']
            overlap=np.minimum(desk.bounds[1],meshes[cid].bounds[1])-np.maximum(desk.bounds[0],meshes[cid].bounds[0])
            qualified=bool(independent and surface.buffer(1e-8).covers(patch) and facing>=spec['chair_toward_table_cos_min'] and all(x['reachable_sides'] for x in reachable) and not np.all(overlap>1e-6) and meal_overlap==0.)
            pairs.append(dict(table_id=tid,chair_id=cid,independent_of_existing_meal_places=independent,center_distance_m=distance,table_height_m=float(desk.bounds[1,2]),surface_m2=float(surface.area),patch_covered=bool(surface.buffer(1e-8).covers(patch)),patch_overlap_with_meal_surface_m2=meal_overlap,chair_toward_table_cos=facing,chair_toward_sofa_cos=facing_cosines,approaches=reachable,clear_rays=sum(x['clear'] for x in rays),total_rays=len(rays),rays=rays,conditional_dedicated_work_geometry=qualified,real_ergonomic_use='unknown'))
    result=dict(status='model_living_work_diagnostics_completed',evaluated_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha256(a.plan),final_manifest_sha256=sha256(a.bundle/'manifest.json'),retained_original_ids=retained,identity_assets_dimensions_preserved=same_identity,immutable_unchanged=immutable,role_relative_access=role_results,access=access,furniture_count=len(meshes),workspace_pairs=pairs,dedicated_work_geometry_found=any(x['conditional_dedicated_work_geometry'] for x in pairs),shared_meal_work_arrangement='not_assessed: do not classify all shared-table solutions as impossible',whole_dwelling_success='unknown_not_accepted',professional_label=False,model_calls=0,training_updates=0,limits=['Candidate role pairing and centered material patch are declared proxies, not exhaustive furniture affordance search.', 'Seat side/back points and facing cosine do not validate knees, sitting or comfort.', 'Unblocked rays do not imply normal gaze, perceived continuity or daylight.', 'All original meal instances retained does not establish two usable dining places; real use remains unknown.', 'Mesh/contact audit is separate; initial defects and missing required objects remain visible.'])
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(furniture=len(meshes),retained=sum(retained.values()),planned=len(retained),work_geometry=result['dedicated_work_geometry_found'],pairs=len(pairs),role_access=role_results)),flush=True)


if __name__=='__main__':main()
