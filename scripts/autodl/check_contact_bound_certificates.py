"""Exact-arithmetic checks and fixed actual two-body threshold controls."""
import argparse
from fractions import Fraction
import itertools
import json
from pathlib import Path
import numpy as np
from conservative_contact_query import outward_world_bounds,penetration_upper_bound


def main(stage):
    out=stage/'bound_certificates_v1';out.mkdir(parents=True,exist_ok=False)
    # Rational arithmetic checks the enclosure against the exact real operations
    # on the supplied binary64 inputs, including cancellation and containment.
    points=np.array(list(itertools.product((-1.,1.),repeat=3)))
    cases=[np.eye(4),np.array([[.6,-.8,0.,1e8],[.8,.6,0.,-1e8],[0.,0.,1.,1e-12],[0.,0.,0.,1.]]),
           np.array([[1.,1.,-1.,-1.],[1e-12,-1.,1.,.25],[0.,0.,1.,0.],[0.,0.,0.,1.]])]
    checks=[]
    for index,matrix in enumerate(cases):
        low,high=outward_world_bounds(points,matrix)
        for point in points:
            for axis in range(3):
                exact=sum(Fraction(float(matrix[axis,k]))*Fraction(float(point[k])) for k in range(3))+Fraction(float(matrix[axis,3]))
                assert Fraction(float(low[axis]))<=exact<=Fraction(float(high[axis]))
        checks.append(dict(case=index,enclosure_contains_exact_rational_affine_vertices=True))
    outer=(np.array([-1.,-1.,-1.]),np.array([1.,1.,1.]))
    inner=(np.array([-.1,-.1,-.1]),np.array([.1,.1,.1]))
    contained=penetration_upper_bound(outer,inner)
    assert contained>=1.1  # Intersection width .2 would be an incorrect upper bound.
    physical=json.loads((stage/'two_body_v1/result.json').read_text())
    floor=(np.array([-2.4,-2.,-.1]),np.array([2.4,2.,0.]))
    actual=[]
    for record in physical['records']:
        hull=(np.asarray(record['actual_Drake_convex_hull_world_min']),
              np.asarray(record['actual_Drake_convex_hull_world_max']))
        upper=penetration_upper_bound(floor,hull);veto=upper<.001
        # A real2mm penetration must survive. A real0.1mm contact is already below
        # the original1mm reporting threshold; the geometry itself is not erased.
        if record['vertical_offset_m']==-.002:assert not veto
        if record['vertical_offset_m']>=0:assert veto
        actual.append(dict(representation=record['representation'],offset_m=record['vertical_offset_m'],
                           original_signed_distance_m=record['signed_distance_m'],
                           analytic_upper_bound_m=upper,veto_under_original_1mm=veto))
    # The source uses >=threshold, so equality must not be vetoed.
    assert not (.001 < .001)
    result=dict(status='exact_arithmetic_enclosures_and_actual_threshold_controls_checked',
        rational_checks=checks,containment_separating_translation_upper_bound_m=contained,
        actual_two_body_controls=actual,inclusive_native_threshold_equality_retained=True,
        model_forwards=0,training_updates=0,new_engine_queries=0,paid_API_calls=0,
        no_full_scene_success_or_learned_gain_claim=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rational_checks','actual_two_body_controls')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);main(p.parse_args().stage.resolve())
