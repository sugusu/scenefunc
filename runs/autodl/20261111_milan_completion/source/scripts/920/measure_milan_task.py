"""Measure actual living/work reference without equating proxies to home acceptance."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from native_project import validate_bundle, sha256
from inspect_native_mesh_contacts import sdf_visuals, world_transform
from measure_room_access import obstacles
from measure_family_choice import hit_segment


def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=read(a.plan)
    for name,h in p['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h
    spec=read(Path(p['spec']));assert sha256(Path(p['spec']))==p['spec_sha256']
    print(datetime.now(timezone.utc).isoformat(),'measure actual independent-project task consequences',flush=True)
    initial_bundle=Path(p['initial_bundle']);bundle=Path(p['final_bundle']);im,ih,ir,_=validate_bundle(initial_bundle);fm,fh,fr,_=validate_bundle(bundle);rid=im['selected_room_id'];assert rid==fm['selected_room_id']
    initial=ih['rooms'][rid];final=fh['rooms'][rid];room=bundle/('room_'+rid)
    assert set(final['objects'])-set(initial['objects'])=={spec['desk_id'],spec['chair_id']}
    assert all(final['objects'][k]==v for k,v in initial['objects'].items())
    assert final['room_geometry']==initial['room_geometry']
    assert all(fr['rooms'][rid]['asset_bindings'][k]==v for k,v in ir['rooms'][rid]['asset_bindings'].items())
    cache={}
    def mesh(oid):
        obj=final['objects'][oid];m=trimesh.util.concatenate([m for _,m in sdf_visuals(room/obj['sdf_path'],bundle,cache)]);m.apply_transform(world_transform(obj));return m
    desk=mesh(spec['desk_id']);seat=mesh(spec['chair_id']);meal=mesh('meal_table_0')
    zmax=float(desk.bounds[1,2]);triangles=desk.triangles
    selected=(desk.face_normals[:,2]>.99)&(triangles[:,:,2].min(axis=1)>=zmax-spec['table_top_band_m'])
    surface=unary_union([Polygon(t[:,:2]) for t in triangles[selected] if Polygon(t[:,:2]).area>1e-10])
    center=desk.bounds[:,:2].mean(axis=0);w,h=spec['material_patch_xy_m'];patch=Polygon([(center[0]+x,center[1]+y) for x,y in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]])
    lo,hi=meal.bounds;meal_envelope=Polygon([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])])
    chair=final['objects'][spec['chair_id']];front=world_transform(chair)[:2,:2]@np.array([0.,1.]);delta=center-seat.bounds[:,:2].mean(axis=0)
    work=dict(desk_bounds_m=desk.bounds.tolist(),seat_bounds_m=seat.bounds.tolist(),top_height_m=zmax,top_projected_union_m2=float(surface.area),material_patch_m2=float(patch.area),patch_covered=bool(surface.buffer(1e-8).covers(patch)),patch_overlap_with_meal_table_envelope_m2=float(patch.intersection(meal_envelope).area),chair_front_alignment_cosine=float(np.dot(front,delta)/np.linalg.norm(delta)),chair_front_to_desk_near_edge_gap_m=float(desk.bounds[0,1]-seat.bounds[1,1]),ergonomic_use='unknown')
    blockers=obstacles(final,room,bundle,1.8);rays=[]
    originxy=chair['transform']['translation'][:2];targetxy=final['objects']['living_sofa_0']['transform']['translation'][:2]
    for height in spec['ray_heights_m']:
        for dx in spec['sofa_target_offsets_x_m']:
            origin=np.array([*originxy,height]);target=np.array([targetxy[0]+dx,targetxy[1],height])
            hit=[b['id'] for b in blockers if hit_segment(origin,target,np.array(b['lo']),np.array(b['hi']))]
            rays.append(dict(origin=origin.tolist(),target=target.tolist(),blockers=hit,clear=not hit))
    access=read(Path(p['actual_access']));fixed=p['fixed_approach_points'];checks=[]
    for s in access['scenarios']:
        checks.append(dict(width_m=s['square_body_width_m'],fixed_total=len(fixed),fixed_reachable=sum(s['targets'][k]['reachable_from_entry'] for k in fixed),office_seat_side=s['targets']['office_seat_side']['reachable_from_entry']))
    result=dict(status='actual_work_reference_measured',checked_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha256(a.plan),input_manifest_sha256=sha256(initial_bundle/'manifest.json'),checkpoint_manifest_sha256=sha256(bundle/'manifest.json'),retained_initial_furniture=13,final_furniture=15,all_initial_objects_and_assets_unchanged=True,workspace=work,sight=dict(clear=sum(x['clear'] for x in rays),total=len(rays),rays=rays,interpretation='Geometric AABB segments including actual structural openings; not perceived openness, privacy or daylight.'),conditional_access=checks,whole_dwelling_accepted=False,professional_preference_label=False,model_calls=0,training_updates=0,limitations=['Workspace material patch and preserved meal geometry do not establish real simultaneous dining/work use.', 'One admissible native reference and one preflight-rejected proposal, not two validated designs.', 'Exact geometry, initial furniture and household details are research assumptions, not original architect survey.', 'Task belongs to an independently allocated family but analyst saw source design text; not sealed.', 'Physical comfort, door operation, installation, storage capacity, daylight and professional assessment remain unknown.'])
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],workspace=work,sight_clear=result['sight']['clear'],sight_total=len(rays),conditional_access=checks)),flush=True)


if __name__=='__main__':main()
