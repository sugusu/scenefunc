"""Finish a stopped fixed-world probe using exact source-triangle certificates.

Retains the failed floating-union attempt. Changes neither task/geometry nor
clearance; checks the same four already-recorded meal candidates and a centered
work rectangle. No search-until-success loop or learned policy is involved.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import trimesh
from shapely.geometry import Polygon,shape,mapping,box
from shapely.ops import unary_union

from exact_planar_support import coverage

R=Path(__file__).resolve().parents[2]
W=R.parent.parent


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def independent_boundary_checks():
    unit=[[0,0],[1,0],[1,1],[0,1]]
    t=[(0,[[0,0],[1,0],[1,1]]),(1,[[0,0],[1,1],[0,1]])]
    cases=[('two_triangles_tile_square',unit,t,'covered'),
        ('single_triangle_does_not_cover_square',unit,t[:1],'uncovered'),
        ('duplicate_triangle_does_not_fill_missing_half',unit,[t[0],t[0]],'uncovered'),
        ('disjoint_triangle',unit,[(2,[[2,2],[3,2],[2,3]])],'uncovered'),
        ('reversed_orientation',unit,[(i,x[::-1]) for i,x in t],'covered'),
        ('tiny_real_gap_must_not_be_filled',unit,[(0,[[0,0],[.5,0],[.5,1]]),
            (1,[[0,0],[.5,1],[0,1]]),(2,[[.500000000001,0],[1,0],[1,1]]),
            (3,[[.500000000001,0],[1,1],[.500000000001,1]])],'uncovered')]
    rows=[]
    for name,p,tri,expected in cases:
        r=coverage(p,tri);assert r['status']==expected
        rows.append(dict(case=name,expected=expected,actual=r))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--scratch',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();scratch=a.scratch.resolve()
    if (root/'final_result.json').exists():raise FileExistsError('No implicit certificate rerun')
    start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Certificate budget180s')))
    signal.alarm(180)
    sys.path.insert(0,str(R/'runs/autodl/20261110_public_function_search/original_evaluator'))
    from native_project import validate_bundle
    from inspect_native_mesh_contacts import sdf_visuals,world_transform
    protocol=read(root/'protocol.json');old=read(root/'world_results.json')
    assert old['status']=='running' and len(old['rows'])==1
    assert old['rows'][0]['world']=='original_table'
    assert protocol['contract']['work_m']==[.4,.4] and protocol['contract']['meal_m']==[.35,.25]
    assert protocol['world_xy_scales']==[1.,.65]
    dump(root/'exact_predicate_boundary_checks.json',independent_boundary_checks())
    candidate_record=read(root/'local_frame_diagnostic.json') if (root/'local_frame_diagnostic.json').exists() else None
    if candidate_record is not None:
        assert len(candidate_record['tests'])==4 and candidate_record['task_unchanged']
    rows=[]
    for wid in ['original_table','smaller_table']:
        b=scratch/wid;m,h,rt,task=validate_bundle(b)
        assert hashlib.sha256(task['design_request'].encode()).hexdigest()==protocol['task_text_sha256']
        rid=m['selected_room_id'];state=h['rooms'][rid];o=state['objects']['meal_table_0'];room=b/('room_'+rid)
        mesh=trimesh.util.concatenate([x for _,x in sdf_visuals(room/o['sdf_path'],b,{})])
        top_z=float(mesh.bounds[1,2]);tri=mesh.triangles
        idx=np.flatnonzero((mesh.face_normals[:,2]>.99)&(tri[:,:,2].min(1)>=top_z-.03))
        surface=unary_union([Polygon(tri[i,:,:2]) for i in idx if Polygon(tri[i,:,:2]).area>1e-10])
        assert surface.geom_type=='Polygon'
        if wid=='smaller_table':
            center=mesh.bounds[:,:2].mean(0);fixed=[]
            for rotation,(w,d) in enumerate([(.35,.25),(.25,.35)]):
                for axis in [0,1]:
                    offset=np.zeros(2);offset[axis]=(w if axis==0 else d)/2
                    ps=[box(*(np.r_[v-np.array([w,d])/2,v+np.array([w,d])/2]))
                        for v in [center-offset,center+offset]]
                    fixed.append(dict(orientation_degrees=rotation*90,split_axis=axis,
                        rectangles_local=[mapping(p) for p in ps]))
            if candidate_record is None:
                candidate_record=dict(geometry_unchanged=True,task_unchanged=True,
                    no_additional_candidate_positions=True,canonical_local_surface=mapping(surface),
                    tests=fixed,origin='Same fixed four centered candidates; no quality-dependent sampling')
                dump(root/'local_frame_diagnostic.json',candidate_record)
            else:
                for old,new in zip(candidate_record['tests'],fixed):
                    assert old['orientation_degrees']==new['orientation_degrees'] and old['split_axis']==new['split_axis']
                    assert all(shape(p).equals_exact(shape(q),0) for p,q in zip(old['rectangles_local'],new['rectangles_local']))
        def certify(poly):
            p=np.array(poly.exterior.coords)[:-1];lo=p.min(0);hi=p.max(0)
            pool=[int(i) for i in idx if np.all(tri[i,:,:2].min(0)<=hi) and np.all(tri[i,:,:2].max(0)>=lo)]
            assert len(pool)<=256
            result=coverage(p,[(i,tri[i,:,:2]) for i in pool])
            result['candidate_triangles']=len(pool)
            return result
        holes=[];remaining=[]
        for ring in surface.interiors:
            hole=Polygon(ring);proof=certify(hole)
            holes.append(dict(coordinates=list(ring.coords),exact_source_union_proof=proof,
                              removed_from_numeric_region=proof['status']=='covered'))
            if proof['status']!='covered':remaining.append(list(ring.coords))
        corrected=Polygon(surface.exterior.coords,remaining)
        assert corrected.is_valid
        numerical_corrected_area=float(corrected.area)
        requested=float(np.prod(protocol['contract']['work_m'])+2*np.prod(protocol['contract']['meal_m']))
        if wid=='original_table':
            oldrow=old['rows'][0];proofs=[]
            for name,w in oldrow['correct_shared_feasibility']['certificate']['witnesses'].items():
                # The original table is axis aligned; exact pose retained.
                world=shape(w['polygon']);points=np.array(world.exterior.coords)[:-1]
                points-=np.array(o['transform']['translation'][:2])
                p=Polygon(points);proof=certify(p);assert proof['status']=='covered'
                proofs.append(dict(region=name,local_polygon=mapping(p),coverage=proof))
            assert all(shape(x['local_polygon']).intersection(shape(y['local_polygon'])).area==0
                       for x,y in itertools.combinations(proofs,2))
            label=True;certificate=dict(kind='simultaneous_polygon_witnesses',regions=proofs)
            independent_modes=True;mode_checks='Follow directly from verified simultaneous witness by dropping reservations.'
        else:
            assert numerical_corrected_area<requested
            c=mesh.bounds[:,:2].mean(0);work=box(c[0]-.2,c[1]-.2,c[0]+.2,c[1]+.2)
            assert corrected.buffer(-1e-6).covers(work)
            work_proof=certify(work);assert work_proof['status']=='covered'
            meal_checks=[]
            for candidate in candidate_record['tests']:
                ps=[shape(p) for p in candidate['rectangles_local']]
                inset=[corrected.buffer(-1e-6).covers(p) for p in ps]
                proof=[certify(p) for p in ps]
                separation=ps[0].intersection(ps[1]).area==0
                meal_checks.append(dict(orientation_degrees=candidate['orientation_degrees'],
                    split_axis=candidate['split_axis'],same_candidate_positions=True,
                    polygons_local=candidate['rectangles_local'],inside_unchanged_1um_inset=inset,
                    exact_coverage=proof,zero_overlap=separation,
                    accepted=all(inset) and separation and all(x['status']=='covered' for x in proof)))
            accepted=[i for i,x in enumerate(meal_checks) if x['accepted']]
            assert accepted,'Retain failed qualification if fixed candidates do not certify; never change dimensions'
            # A conservative upper bound avoids trusting a float-union area for
            # a negative. Every table-top point is inside this bounding rectangle.
            # Here its .65m-scaled bounding rectangle is too loose, so also use
            # the exact original triangulated projected surface area as an upper
            # bound (sum triangle areas counts overlaps rather than removing them).
            from fractions import Fraction as F
            def twice_area(v):
                p=[[F(float(z)) for z in xy] for xy in v]
                return abs((p[1][0]-p[0][0])*(p[2][1]-p[0][1])-(p[1][1]-p[0][1])*(p[2][0]-p[0][0]))
            area_sum=sum((twice_area(tri[i,:,:2])/2 for i in idx),F(0))
            req_exact=F(2,5)*F(2,5)+2*F(7,20)*F(1,4)
            assert area_sum<req_exact,'No sound area certificate if overlapping projected faces make bound loose'
            label=False;certificate=dict(kind='necessary_area_bound_with_exact_triangle_sum_upper_bound',
                required_area_exact=str(req_exact),top_area_upper_bound_exact=str(area_sum),
                required_area_m2=float(req_exact),top_area_upper_bound_m2=float(area_sum),
                strict_positive_area_deficit_m2=float(req_exact-area_sum),
                valid_for_arbitrary_translations_and_in_plane_orientations=True)
            independent_modes=True;mode_checks=dict(work=work_proof,
                work_polygon_local=mapping(work),fixed_meal_candidates=meal_checks,
                first_accepted_candidate_index=accepted[0],candidate_positions_added=0)
        row=dict(world=wid,objects=len(state['objects']),xy_scale=1. if wid=='original_table' else .65,
            task_text_sha256=protocol['task_text_sha256'],
            correct_shared_surface_feasibility=label,
            label_scope='Explicit disjoint planar reservations; not complete human use or whole-scene task success',
            certificate=certificate,independent_modes_feasible=independent_modes,
            independent_modes_evidence=mode_checks,numerical_hole_proofs=holes,
            unchanged_safety_inset_m=1e-6,
            hypothesis_outputs=dict(required_persistence=label,omit_persistence=independent_modes,
                                    force_separate_tables=False))
        rows.append(row)
    gold=[r['correct_shared_surface_feasibility'] for r in rows]
    assert gold==[True,False]
    predictions={name:[r['hypothesis_outputs'][name] for r in rows] for name in rows[0]['hypothesis_outputs']}
    survivors={str(subset):[h for h,p in predictions.items() if all(p[i]==gold[i] for i in subset)]
               for subset in [[],[0],[1],[0,1]]}
    assert survivors['[0, 1]']==['required_persistence']
    dump(root/'labelled_program_disambiguation.json',dict(predictions=predictions,gold=gold,survivors=survivors,
        same_task_in_both_worlds=True,known_hypotheses_authored_by_researcher=True,
        algorithm='Ordinary labelled consistency filtering; same-semantic-task/world testing has prior art',
        empirical_model_failures=0,policy_accuracy_or_training_not_measured=True,
        general_program_identifiability=False))
    result=dict(status='same_task_world_probe_certified_after_explicit_numeric_diagnosis',rows=rows,
        first_float_union_attempt_status='failed_individual_mode_qualification; preserved in world_results.json and original log',
        correction='Remove only numeric holes proved already covered by source triangles using exact rational subtraction; verify actual rectangle coverage exactly',
        geometry_changed_after_initial_attempt=False,task_changed=False,safety_threshold_changed=False,
        new_meal_candidate_positions_after_initial_four_diagnostic=False,
        original_task_proxy_conflict='Old source structured work40x30 proxy remains archival metadata; these labels refer only to the explicit new40x40 probe contract, not the old score or free-language parsing.',
        source_families=1,controlled_worlds=2,authored_hypotheses=3,
        wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_calls=0,training_updates=0,paid_API_calls=0,new_raw_text_scene_generations=0,
        full_scene_functional_success=None,training_dataset_admitted=False,
        limits=['No new five-view observations; no neural sample or training admission.',
               'Counterfactual scaled geometry is research derived, not historical or professionally approved.',
               'One known three-hypothesis class is not general language-program identification.',
               'A labelled consistency baseline is not a new learning principle.'])
    dump(root/'final_result.json',result);signal.alarm(0)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
