"""Measure retained dwelling roles and the explicit cost of one use transition."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from native_project import validate_bundle,sha256
from measure_room_access import measure
from inspect_native_mesh_contacts import sdf_visuals,world_transform


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=read(a.plan)
    for n,h in p['code_sha256'].items():assert sha256(Path(__file__).parent/n)==h
    protocol=read(Path(p['protocol']));assert sha256(Path(p['protocol']))==p['protocol_sha256']
    initial_bundle=Path(p['initial_bundle']);final_bundle=Path(p['final_bundle']);im,ih,ir,_=validate_bundle(initial_bundle);fm,fh,fr,_=validate_bundle(final_bundle);rid=im['selected_room_id'];assert rid==fm['selected_room_id'];initial=ih['rooms'][rid];final=fh['rooms'][rid]
    print(datetime.now(timezone.utc).isoformat(),'measure declared lifecycle consequences',flush=True)
    removed=set(initial['objects'])-set(final['objects']);added=set(final['objects'])-set(initial['objects']);assert removed=={'former_child_bed_0'} and added=={'dining_table_0','dining_chair_1_0'}
    preserved={k:final['objects'][k]==v for k,v in initial['objects'].items() if k not in removed};assert all(preserved.values());assert initial['room_geometry']==final['room_geometry']
    binding_preserved={k:fr['rooms'][rid]['asset_bindings'][k]==v for k,v in ir['rooms'][rid]['asset_bindings'].items() if k not in removed};assert all(binding_preserved.values())
    analyses={}
    for label,state,root,extra in [('before',initial,initial_bundle,protocol['initial_role_points']),('after',final,final_bundle,protocol['final_role_points'])]:
        settings={**protocol['reference_measurement'],'approach_points_xy_m':{**protocol['fixed_approach_points'],**extra}}
        result=measure(state,root/('room_'+rid),root,settings);analyses[label]=result
    comparison=[]
    for before,after in zip(analyses['before']['scenarios'],analyses['after']['scenarios']):
        fixed=protocol['fixed_approach_points'];lost=[k for k in fixed if before['targets'][k]['reachable_from_entry'] and not after['targets'][k]['reachable_from_entry']]
        comparison.append(dict(body_width_m=before['square_body_width_m'],fixed_points=len(fixed),before_fixed_reachable=sum(before['targets'][k]['reachable_from_entry'] for k in fixed),after_fixed_reachable=sum(after['targets'][k]['reachable_from_entry'] for k in fixed),lost_fixed_points=lost,former_bed_approach_before=before['targets']['former_bed_side']['reachable_from_entry'],office_seat_approach_after=after['targets']['office_seat_side']['reachable_from_entry']))
    objs={}
    for oid in added:
        obj=final['objects'][oid];mesh=trimesh.util.concatenate([m for _,m in sdf_visuals(final_bundle/('room_'+rid)/obj['sdf_path'],final_bundle,{})]);mesh.apply_transform(world_transform(obj));objs[oid]=(obj,mesh)
    desk,mesh=objs['dining_table_0'];chair,cmesh=objs['dining_chair_1_0'];zmax=float(mesh.bounds[1,2]);triangles=mesh.triangles;sel=(mesh.face_normals[:,2]>.99)&(np.min(triangles[:,:,2],axis=1)>=zmax-.03)
    polygons=[Polygon(t[:,:2]) for t in triangles[sel] if Polygon(t[:,:2]).area>1e-10];surface=unary_union(polygons);toparea=float(surface.area)
    center=mesh.bounds[:,:2].mean(0);testpatch=Polygon([(center[0]+x,center[1]+y) for x,y in [(-.2,-.15),(.2,-.15),(.2,.15),(-.2,.15)]])
    front=world_transform(chair)[:2,:2]@np.array([0.,1.]);delta=mesh.bounds[:,:2].mean(0)-cmesh.bounds[:,:2].mean(0);cosine=float(np.dot(front,delta)/np.linalg.norm(delta))
    work=dict(table_asset_id=fr['rooms'][rid]['asset_bindings']['dining_table_0']['asset_id'],table_bounds_m=mesh.bounds.tolist(),top_height_m=zmax,upward_surface_projected_union_area_m2=toparea,central_40x30cm_patch_covered=bool(surface.buffer(1e-8).covers(testpatch)),seat_bounds_m=cmesh.bounds.tolist(),chair_front_alignment_cosine=cosine,chair_front_to_table_near_edge_xy_gap_m=float(mesh.bounds[0,1]-cmesh.bounds[1,1]),interpretation='Actual mesh-surface and static-pose proxies only; not real work equipment, seat comfort, knee space or ergonomic acceptance. 40x30cm patch is a research probe, not a normative standard.')
    result=dict(status='lifecycle_transition_objective_consequences_measured',input_manifest_sha256=sha256(initial_bundle/'manifest.json'),checkpoint_manifest_sha256=sha256(final_bundle/'manifest.json'),removed_ids=sorted(removed),added_ids=sorted(added),preserved_existing_objects=preserved,asset_bindings_preserved=binding_preserved,fixed_access_comparison=comparison,access=analyses,workspace=work,tradeoff=dict(initial_bed_instances=2,final_bed_instances=1,main_bed_identity_retained='main_bed_0' in final['objects'],former_bed_retained_in_scene=False,removed_id_reserved='former_child_bed_0' in read(final_bundle/'session_checkpoint.json')['reserved_object_ids'],disposal_storage_and_transport='unknown',if_former_sleeping_requirement_persisted='This candidate loses that bed and cannot be accepted as satisfying the unchanged prior sleeping requirement; this is an objective hypothetical, not a source historical rejection.'),whole_dwelling_functional_success='unknown',professional_preference_label=False,training_admitted=False,source_support='Role adaptation principle; removal/addition order and all poses are researcher reconstruction, not authored historical operations.',model_calls=0,training_updates=0)
    write(a.out,result);print(json.dumps({k:result[k] for k in ['removed_ids','added_ids','fixed_access_comparison','workspace']}),flush=True)

if __name__=='__main__':main()
