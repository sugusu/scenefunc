"""Check the frozen Pine transfer consequences without assigning professional labels."""
import argparse,json,itertools
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import trimesh
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from native_project import validate_bundle,sha256
from measure_room_access import measure
from inspect_native_mesh_contacts import sdf_visuals,world_transform

def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=read(a.plan)
    for n,h in p['code_sha256'].items():assert sha256(Path(__file__).parent/n)==h
    protocol=read(p['protocol']);assert sha256(Path(p['protocol']))==p['protocol_sha256']
    print(datetime.now(timezone.utc).isoformat(),'measure source-conditioned two-workstation transfer',flush=True)
    ib=Path(p['initial_bundle']);fb=Path(p['final_bundle'])
    im,ih,ir,_=validate_bundle(ib);fm,fh,fr,_=validate_bundle(fb);rid=im['selected_room_id'];assert fm['selected_room_id']==rid
    initial=ih['rooms'][rid];final=fh['rooms'][rid]
    removed=set(initial['objects'])-set(final['objects']);added=set(final['objects'])-set(initial['objects'])
    assert not removed and added=={'work_desk_0','dining_chair_1_0'}
    moved={'old_desk_0','old_chair_0'}
    for oid,obj in initial['objects'].items():
        if oid in moved:
            assert {k:v for k,v in obj.items() if k!='transform'}=={k:v for k,v in final['objects'][oid].items() if k!='transform'}
        else:assert obj==final['objects'][oid]
    preserved_bindings={k:fr['rooms'][rid]['asset_bindings'][k]==v for k,v in ir['rooms'][rid]['asset_bindings'].items()}
    assert all(preserved_bindings.values()) and initial['room_geometry']==final['room_geometry']
    analyses={};work={};simultaneous={}
    for label,state,root,pairs,roles in [('before',initial,ib,protocol['work_pairs_before'],protocol['initial_role_points']),('after',final,fb,protocol['work_pairs_after'],protocol['final_role_points'])]:
        settings={**protocol['reference_measurement'],'approach_points_xy_m':{**protocol['fixed_approach_points'],**roles}}
        analyses[label]=measure(state,root/('room_'+rid),root,settings)
        meshes={}
        for oid,obj in state['objects'].items():
            if obj['object_type']!='furniture':continue
            mesh=trimesh.util.concatenate([m for _,m in sdf_visuals(root/('room_'+rid)/obj['sdf_path'],root,{})]);mesh.apply_transform(world_transform(obj));meshes[oid]=mesh
        rows=[];patches=[]
        for deskid,chairid in pairs:
            mesh=meshes[deskid];chair=meshes[chairid];obj=state['objects'][chairid]
            zmax=float(mesh.bounds[1,2]);probe=protocol['surface_probe'];triangles=mesh.triangles
            select=(mesh.face_normals[:,2]>probe['min_up_normal'])&(np.min(triangles[:,:,2],axis=1)>=zmax-probe['near_top_band_m'])
            surface=unary_union([Polygon(v[:,:2]) for v in triangles[select] if Polygon(v[:,:2]).area>1e-10])
            centre=(np.array(state['objects'][deskid]['bbox_min'])+np.array(state['objects'][deskid]['bbox_max']))/2
            w=probe['width_m']/2;h=probe['depth_m']/2
            corners=np.array([[centre[0]+x,centre[1]+y,0.] for x,y in [(-w,-h),(w,-h),(w,h),(-w,h)]])
            patch=Polygon(trimesh.transform_points(corners,world_transform(state['objects'][deskid]))[:,:2]);patches.append(patch)
            front=world_transform(obj)[:2,:2]@np.array([0.,1.]);delta=mesh.bounds[:,:2].mean(0)-chair.bounds[:,:2].mean(0)
            rows.append(dict(desk_id=deskid,chair_id=chairid,asset_id=fr['rooms'][rid]['asset_bindings'].get(deskid,ir['rooms'][rid]['asset_bindings'].get(deskid))['asset_id'],
                desk_bounds_m=mesh.bounds.tolist(),chair_bounds_m=chair.bounds.tolist(),top_height_m=zmax,upward_near_top_projected_union_area_m2=float(surface.area),
                central_40x30cm_patch_covered=bool(surface.buffer(1e-8).covers(patch)),chair_front_alignment_cosine=float(np.dot(front,delta)/np.linalg.norm(delta)),
                desk_chair_xy_envelope_distance_m=float(box(*mesh.bounds[0,:2],*mesh.bounds[1,:2]).distance(box(*chair.bounds[0,:2],*chair.bounds[1,:2])))))
        work[label]=dict(workstation_pairs=rows,separate_desktop_probe_count=len(rows),patch_pair_intersection_areas_m2=[float(x.intersection(y).area) for x,y in itertools.combinations(patches,2)])
        scenarios=[]
        for width in protocol['simultaneous_seat_proxies']['square_widths_m']:
            seats={cid:box(*(meshes[cid].bounds[:,:2].mean(0)-width/2),*(meshes[cid].bounds[:,:2].mean(0)+width/2)) for _,cid in pairs}
            conflict=[dict(chair_id=cid,other_id=oid,area_m2=float(seat.intersection(box(*m.bounds[0,:2],*m.bounds[1,:2])).area)) for cid,seat in seats.items() for oid,m in meshes.items() if oid!=cid and seat.intersection(box(*m.bounds[0,:2],*m.bounds[1,:2])).area>1e-8]
            overlap=[dict(chair_ids=[x,y],area_m2=float(seats[x].intersection(seats[y]).area)) for x,y in itertools.combinations(seats,2) if seats[x].intersection(seats[y]).area>1e-8]
            scenarios.append(dict(square_width_m=width,other_furniture_envelope_conflicts=conflict,seat_square_overlaps=overlap,all_static_proxy_zones_disjoint=not conflict and not overlap))
        simultaneous[label]=scenarios
    comparison=[]
    for before,after in zip(analyses['before']['scenarios'],analyses['after']['scenarios']):
        fixed=protocol['fixed_approach_points']
        comparison.append(dict(width_m=before['square_body_width_m'],fixed_targets=len(fixed),before_reachable=sum(before['targets'][x]['reachable_from_entry'] for x in fixed),after_reachable=sum(after['targets'][x]['reachable_from_entry'] for x in fixed),lost_fixed_targets=[x for x in fixed if before['targets'][x]['reachable_from_entry'] and not after['targets'][x]['reachable_from_entry']],initial_seat_approach=before['targets']['old_seat_side']['reachable_from_entry'],final_seat_approaches={x:after['targets'][x]['reachable_from_entry'] for x in protocol['final_role_points']}))
    result=dict(status='pine_transfer_consequences_measured',plan_sha256=sha256(a.plan),initial_manifest_sha256=sha256(ib/'manifest.json'),final_manifest_sha256=sha256(fb/'manifest.json'),removed_ids=sorted(removed),added_ids=sorted(added),retained_moved_ids=sorted(moved),all_original_bindings_preserved=all(preserved_bindings.values()),other_original_records_unchanged=True,work=work,simultaneous_seat_square_proxies=simultaneous,common_access=comparison,access=analyses,
        tradeoffs=dict(additional_desk_instances=1,additional_chair_instances=1,old_desk_identity_preserved=True,retained_guest_sofa_does_not_verify_sleeping=True,bookshelf_capacity_unknown=True,moving_path_effort_unknown=True,work_equipment_and_knee_space_unknown=True,alternative_shared_desk_layout_not_tested=True),
        original_project_historical_replay=False,whole_room_functional_success='unknown',professional_preference_label=False,training_admitted=False,model_calls=0,training_updates=0,
        interpretation='Static surface, identity and access evidence for a declared transfer of a sourced co-produced arrangement. All coordinates/order/substitute instances are reconstruction. One assumed room and proxy-width sensitivity cannot establish source reproduction, ergonomics, aesthetics or learning.')
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],access=comparison,work=work,seat_proxies=simultaneous)),flush=True)

if __name__=='__main__':main()
