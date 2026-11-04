"""Score all six scene outcomes and independently recheck positive coordinates.

Only saved final meshes/checkpoints are used for geometry; model feedback text
and model self-assessments are never used as scores. No human-use success claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon, shape, box
from shapely.ops import unary_union


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_coordinates(scene_path, result):
    if result['status']!='witness':return {'status':'not_applicable','reason':'No positive witness to verify'}
    scene=trimesh.load(scene_path,force='scene');meshes={}
    for node in scene.graph.nodes_geometry:
        tf,gid=scene.graph[node];mesh=scene.geometry[gid].copy();mesh.apply_transform(tf);meshes[node]=mesh
    rectangles={};checks=[]
    for task,w in result['witnesses'].items():
        polygon=shape(w['polygon']);m=meshes[w['surface_id']];z=float(m.bounds[1,2])
        expected=[.8,.5] if task=='work' else [.6,.4]
        assert np.allclose(sorted(w['size_xy']),sorted(expected),atol=1e-9,rtol=0)
        center=np.asarray(w['center_xy']);size=np.asarray(w['size_xy'])
        reconstructed=box(*(center-size/2),*(center+size/2))
        assert reconstructed.symmetric_difference(polygon).area<=1e-12
        assert abs(w['z']-z)<=1e-8
        mask=(m.face_normals[:,2]>=1-1e-7)&(np.max(np.abs(m.triangles[:,:,2]-z),axis=1)<=1e-5)
        tops=[Polygon(triangle[:,:2]) for triangle in m.triangles[mask]]
        tops=[p for p in tops if p.is_valid and p.area>0]
        top=unary_union(tops)
        assert top.covers(polygon), (task,'outside actual triangle-union surface')
        for oid,other in meshes.items():
            if oid==w['surface_id'] or not box(*other.bounds[:,:2].reshape(-1)).intersects(polygon):continue
            level=z+1e-4
            if not other.bounds[0,2]<=level<=other.bounds[1,2]:continue
            section=other.section(plane_origin=[0,0,level],plane_normal=[0,0,1])
            assert section is not None,(task,oid,'unknown section')
            for ring in section.discrete:
                assert len(ring)>=4 and np.linalg.norm(ring[0]-ring[-1])<=1e-6,(task,oid,'open section')
                obstruction=Polygon(ring[:,:2]);assert obstruction.is_valid
                assert obstruction.intersection(polygon).area<=1e-10,(task,oid,'blocked witness')
        rectangles[task]=polygon;checks.append({'task':task,'surface_id':w['surface_id'],'coordinate_and_mesh_checks':'pass'})
    assert rectangles['work'].intersection(rectangles['dining']).area<=1e-12
    return {'status':'pass','checks':checks,'scope':'Zero-thickness rectangles only; no reachability or ergonomics'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-id',default='P920-DD-v1_20261104T083214Z');a=p.parse_args()
    run=a.root/'runtime'/a.run_id;rows=[]
    for label in ['A_draft','B_draft','A_G','A_U','B_G','B_U']:
        arm=label[0];draft=run/'drafts'/arm
        if label.endswith('draft'):
            final=draft;application=None;model=None
        else:
            application=read(run/'revisions'/label/'application.json')
            final=Path(application['final_directory']);model=read(run/'model_runs'/('formal_'+label)/'result.json')
        state=read(final/'checkpoint.json')['payload'];audit=read(final/'audit.json');measurement=read(final/'use_measurement.json')
        scene=final/'partial_scene.glb';digest=sha(scene)
        assert digest==audit['scene_sha256']==measurement['scene_sha256']
        assert read(final/'views/receipt.json')['source_sha256']==digest
        joint=measurement['results']['scene_section_free']['simultaneous_use']
        separate=measurement['results']['scene_section_free']['separate_use']
        verification=verify_coordinates(scene,joint)
        names=[x['name'] for x in state['placed_instances']]
        original=read(draft/'checkpoint.json')['payload'];before_ids={x['object_id'] for x in original['placed_instances']};after_ids={x['object_id'] for x in state['placed_instances']}
        before_audit=read(draft/'audit.json');before_pairs={tuple(p) for p in before_audit['furniture_surface_intersection_pairs']};after_pairs={tuple(p) for p in audit['furniture_surface_intersection_pairs']};removed=before_ids-after_ids
        lost_pairs=before_pairs-after_pairs
        row={'label':label,'source_arm':arm,'method':'draft' if model is None else label[-1],
             'origin':'verified historical draft reconstructed on 920' if model is None else 'one actual 920 model revision from verified draft',
             'final_directory':str(final),'checkpoint_sha256':sha(final/'checkpoint.json'),'scene_sha256':digest,
             'objects':len(after_ids),'furniture_intersection_pairs':len(after_pairs),'pairs':sorted(after_pairs),
             'removed_objects':sorted(removed),'added_objects':sorted(after_ids-before_ids),
             'disappeared_pairs_involving_removed_objects':sorted(p for p in lost_pairs if any(o in removed for o in p)),
             'outside_floor':audit['out_of_rectangular_floor_objects'],
             'surface_A': 'witness_without_storage_transition' if verification['status']=='pass' else 'unknown_storage_transition' if separate['status']=='witness' else 'not_found',
             'surface_B':joint['status'],'independent_joint_witness_check':verification,
             'finite_search_not_global_infeasibility':True,
             'presence_only':{'bed':any('bed' in n for n in names),'work_table':any(n in ['desk','office_desk'] for n in names),
                              'dining_table':sum(n=='dining_table' for n in names),'dining_chairs':sum(n=='dining_chair' for n in names),
                              'storage_named_objects':sum(n in ['wardrobe','dresser','bookcase','low_bookcase'] for n in names)},
             'unknown':['seated posture','chair pullout','walking reachability','sleep access','door swing','window access/view quality','storage capacity','storage transition','usable movement space'],
             'whole_room_A':'unknown','whole_room_B':'unknown',
             'tools':{k:application[k] for k in ['tool_calls_attempted','tool_calls_successful','tool_calls_failed']} if application else None,
             'model':{k:model[k] for k in ['status','input_tokens','output_tokens','generation_seconds','total_seconds','peak_allocated_bytes','peak_reserved_bytes']} if model else None}
        rows.append(row)
    by_label={row['label']:row for row in rows}
    matrices={method:{source:{'A':by_label[source+'_'+method]['surface_A'],'B':by_label[source+'_'+method]['surface_B']} for source in ['A','B']} for method in ['G','U']}
    tool_events=[(label,event) for label in ['A_G','A_U','B_G','B_U'] for event in read(run/'revisions'/label/'application.json').get('new_tool_events',[])]
    failures=[{'label':label,'call':e['call'],'response':e['response']} for label,e in tool_events if e['response'].get('success') is not True]
    summary={'handoff_id':a.run_id.split('_')[0],'run_id':a.run_id,'status':'analysis_complete','scored_at':datetime.now(timezone.utc).isoformat(),
             'rows':rows,'cross_requirement_surface_matrices':matrices,'tool_failures':failures,
             'formal_attempts':4,'formal_completed':sum(r['model']['status']=='completed' for r in rows if r['model']),
             'formal_output_tokens':sum(r['model']['output_tokens'] for r in rows if r['model']),
             'formal_generation_seconds':sum(r['model']['generation_seconds'] for r in rows if r['model']),
             'formal_peak_allocated_bytes':max(r['model']['peak_allocated_bytes'] for r in rows if r['model']),
             'tool_calls':len(tool_events),'tool_successes':sum(e['response'].get('success') is True for _,e in tool_events),
             'scientific_scope':'One development room, two related briefs, one seed, one revision each; no training, held-out transfer or human preference evidence.'}
    target=run/'RESULTS.json';assert not target.exists();target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':[{k:r[k] for k in ['label','objects','furniture_intersection_pairs','surface_A','surface_B','removed_objects']} for r in rows], 'tool_calls':summary['tool_calls'],'tool_successes':summary['tool_successes']}))


if __name__=='__main__':
    main()
