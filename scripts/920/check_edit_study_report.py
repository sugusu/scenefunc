"""Regression checks for scientific outcome accounting, using real audits."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from summarize_edit_study import compare_audits


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    p=a.study;read=lambda f:json.loads(f.read_text())
    initial=read(p/'recorded/Pine_initial_audit.json');reference=read(p/'evaluation/Pine/REFERENCE_AUDIT.json');milan=read(p/'recorded/Milan_initial_audit.json')
    roles={k:v['object_id'] for k,v in read(p/'evaluation/Pine/AUDIT_PLAN.json')['activity']['roles'].items()};checks=[]
    def primary(x,width=.5):return next(s for s in x['scenarios'] if s['width_m']==width)
    x=compare_audits(initial,reference,roles);s=primary(x)
    assert s['activity_transition']=='gained' and s['newly_lost_roles']==['old_seat_side']
    assert s['new_declared_proxy_failure'] and not s['new_geometry_or_identity_failure']
    assert s['lost_role_probe_context'][0]['seat_square_and_any_approach_pass']
    assert s['lost_role_probe_context'][0]['other_reachable_seat_probes']==['old_chair_0:right','old_chair_0:back']
    checks.append('Actual Pine gain retains failed side probe and alternative reachable sides; not a fabricated all-probes pass')
    x=compare_audits(milan,milan);assert primary(x)['activity_transition']=='preserved';checks.append('Actual Milan no-edit success is preserved, never gained')
    x=compare_audits(initial,initial,roles);assert primary(x,.6)['remaining_initial_failed_roles']==['bookshelf_front','central_activity'] and not primary(x,.6)['new_declared_proxy_failure'];checks.append('Preexisting route defects are not newly introduced damage')
    mutated=copy.deepcopy(reference)
    for s in mutated['activity']['scenarios']:s['role_access'].pop('sofa_front')
    x=compare_audits(initial,mutated,roles);assert 'sofa_front' in primary(x)['newly_lost_roles'] and primary(x)['original_role_denominator']==4;checks.append('Missing role cannot shrink denominator')
    mutated=copy.deepcopy(reference);mutated['activity']['retained_ids'].pop('old_desk_0');mutated['activity']['identity_asset_dimensions_preserved'].pop('old_desk_0')
    x=compare_audits(initial,mutated,roles);assert x['missing_original_instances']==['old_desk_0'] and primary(x)['new_geometry_or_identity_failure'];checks.append('Identity deletion overrides an otherwise successful workspace proxy')
    before=copy.deepcopy(initial);after=copy.deepcopy(initial);pair=dict(objects=['old_desk_0','old_chair_0'],triangle_surface_intersection_detected=True)
    before['positive_aabb_pairs'].append(pair);after['positive_aabb_pairs'].append({**pair,'objects':list(reversed(pair['objects']))})
    x=compare_audits(before,after,roles);assert not x['new_surface_intersections'] and len(x['remaining_initial_surface_intersections'])==1;checks.append('Collision pair ordering cannot create a new violation')
    after['positive_aabb_pairs'].append(dict(objects=['bookshelf_0','sofa_0'],triangle_surface_intersection_detected=True))
    x=compare_audits(before,after,roles);assert len(x['new_surface_intersections'])==1 and primary(x)['new_geometry_or_identity_failure'];checks.append('New and preexisting contacts remain separate')
    mutated=copy.deepcopy(reference);mutated['activity']['scenarios'][0]['activity_proxy_pass']=None
    try:compare_audits(initial,mutated,roles)
    except ValueError:checks.append('Unknown proxy cannot be coerced into success/failure')
    else:raise AssertionError('Unknown incorrectly accepted')
    result=dict(status='report_accounting_checks_passed',recorded_at=datetime.now(timezone.utc).isoformat(),checks=checks,count=len(checks),
        code_sha256=hashlib.sha256(Path(__file__).with_name('summarize_edit_study.py').read_bytes()).hexdigest(),new_model_calls=0,scene_edits=0,training_updates=0,
        prior_failed_expectation='An earlier ad-hoc assertion assumed Pine reference had no newly failed probe. Actual old_seat_side is false at every width while other seat approaches pass. Corrected reporting distinguishes probe failure and measured alternatives without changing any frozen audit, feedback, training or scene.')
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks))))


if __name__=='__main__':main()
