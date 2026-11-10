"""Analytic same-state half-space certificates, not replacement scene scores.

The floor is an axis-aligned solid box ending at z=0. A collision part whose
whole AABB is at/above this plane cannot have positive-volume overlap with
that floor. This necessary geometric fact does not validate other contacts,
task semantics, stability, or a revised policy. Original scores stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    stage=a.stage.resolve();a.out.mkdir(parents=True,exist_ok=False)
    contacts=json.loads((stage/'partial_replay_v2/actual_drake_contacts.json').read_text())
    frame_path=stage/'asset_collision_frame_audit/geometry_bounds.json'
    frames=json.loads(frame_path.read_text());objects={x['object_id']:x for x in frames['objects']}
    rows=[]
    for pair in contacts:
        if pair['a']['name']=='room_geometry::floor_collision':floor,other=pair['a'],pair['b']
        elif pair['b']['name']=='room_geometry::floor_collision':floor,other=pair['b'],pair['a']
        else:continue
        assert floor['shape']['type']=='Box'
        F=floor['X_WG'];assert [r[:3] for r in F[:3]]==[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        plane=F[2][3]+floor['shape']['height']/2
        oid,part_name=other['name'].split('::',1)
        parts={x['name']:x for x in objects[oid]['parts']};part=parts[part_name]
        # In the actual evidence the row is exactly [0,0,1]: only yaw is free.
        X=other['X_WG'];assert X[2][:3]==[0.,0.,1.]
        assert abs(other['shape']['scale']-part['scale'][2])<1e-12
        lo=part['min'][2]+X[2][3];hi=part['max'][2]+X[2][3]
        rows.append(dict(object_id=oid,collision_name=part_name,mesh_SHA256=part['sha256'],
                         actual_query_distance=pair['signed_distance'],world_Z_min=lo,world_Z_max=hi,
                         floor_top=plane,disjoint_interiors_by_Z_plane=lo>=plane,
                         AABB_degenerate_Z=part['min'][2]==part['max'][2],
                         geometric_certificate='all collision-mesh vertices and their convex hull have z >= floor top'
                            if lo>=plane else 'not certified; no conclusion from this half-space check'))
    assert rows
    result=dict(status='same_state_half_space_evidence_computed',floor_pairs=len(rows),
                certified_disjoint_interiors=sum(r['disjoint_interiors_by_Z_plane'] for r in rows),
                certified_with_native_distance_below_minus_1mm=sum(r['disjoint_interiors_by_Z_plane'] and r['actual_query_distance']<-.001 for r in rows),
                zero_Z_extent_parts=sum(r['AABB_degenerate_Z'] for r in rows),
                input_frame_SHA256=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                original_score_changed=False,new_success_labels_created=False,scene_coordinates_modified=False,
                other_collisions_not_certified=True,whole_scene_success=None,rows=rows)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
