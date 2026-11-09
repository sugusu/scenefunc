"""A transparent edit/stop control using only public pre-action measurements.

This does not propose coordinates, execute an edit, or certify a whole room.
All 24 fixed source units are retained; unsupported task schemas return unknown.
"""
import argparse
import json
from pathlib import Path

from audit_material_routing import GitInputs, CORPUS, write


def extract_payload(messages):
    found=[]
    for message in messages:
        if not isinstance(message['content'],list):continue
        for item in message['content']:
            if item.get('type')!='text':continue
            for line in item['text'].splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict) and 'available_measurement' in value:found.append(value)
    if len(found)!=1:return None
    return found[0]


def decide(payload):
    if payload is None or 'declared_activity_requirements' not in payload:
        return {'decision':'unknown','reason':'No supported explicit activity/measurement contract.'}
    mode=payload.get('current_requested_mode')
    if mode not in ('separate_work','joined_four_place_hosting'):
        return {'decision':'unknown','reason':'Unsupported activity; no guessed semantics.'}
    req=payload['declared_activity_requirements'];m=payload['available_measurement']
    checks={}
    geom=m.get('geometry',{})
    checks['reported_envelope_geometry']=None if not all(k in geom for k in ('positive_3d_AABB_pairs','outside_declared_wall_inner_faces')) else not geom['positive_3d_AABB_pairs'] and not geom['outside_declared_wall_inner_faces']
    table=m.get('table_geometry')
    if mode=='separate_work':
        checks['active_table_relation']=None if table is None else table['footprint_gap_m']>=req['separate_min_gap_m']
        needed=req['work_chair_ids']
    else:
        checks['active_table_relation']=None if table is None else table['joined']
        overlap=m.get('place_rectangles_overlap')
        checks['place_rectangles_do_not_overlap']=None if overlap is None else not overlap
        needed=[s['chair_id'] for s in req['seat_assignments']]
    for oid in set(req['table_ids'])|set(needed):
        support=m.get('measurement_support',{}).get(oid)
        checks['measurable_'+oid]=True if support=='supported_known_proxy' else False if support=='missing' else None
    for oid in needed:
        checks['alignment_'+oid]=m.get('seat_alignment',{}).get(oid,{}).get('pass_')
    for label,scenarios,targets in [
        ('fixed',m.get('fixed_access',{}).get('scenarios',[]),list(req['access_settings']['approach_points_xy_m'])),
        ('seat',m.get('seat_access',[]),needed)]:
        for width in req['access_settings']['square_body_widths_m']:
            matching=[s for s in scenarios if s.get('square_body_width_m')==width]
            for target in targets:
                checks[f'{label}_{width}_{target}']=matching[0].get('targets',{}).get(target,{}).get('reachable_from_entry') if len(matching)==1 else None
    failures=[k for k,v in checks.items() if v is False]
    unknown=[k for k,v in checks.items() if v is None]
    action='edit_required' if failures else 'unknown' if unknown else 'stop_with_scoped_unknowns'
    return {'decision':action,'failed':failures,'unknown':unknown,'checked_predicates':len(checks),
            'whole_room_success':'unknown','retained_unknowns':payload.get('unknowns',[])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    src=GitInputs(a.repo);corpus=src.json(CORPUS);rows=[]
    for unit in corpus['rows']:
        request,path=src.reference(unit['request'])
        prediction=decide(extract_payload(request))
        # Labels are read only AFTER the policy returns, for an audit comparison.
        expected='stop_with_scoped_unknowns' if unit['target_kind']=='terminal_control' else 'edit_required'
        rows.append({'unit_id':unit['id'],'request_path':path,'request_sha256':unit['request']['sha256'],
                     'gate':prediction,'reference_kind':unit['target_kind'],
                     'edit_stop_label_agreement':prediction['decision']==expected if prediction['decision']!='unknown' else None})
    eligible=[r for r in rows if r['edit_stop_label_agreement'] is not None]
    write(a.output,{'source_units':len(rows),'applicable_units':len(eligible),
                    'label_agreement':sum(r['edit_stop_label_agreement'] for r in eligible),
                    'unknown_units':len(rows)-len(eligible),'rows':rows,
                    'source_sha256':src.manifest,
                    'scope':'Development classification of already supplied proxy evidence. No coordinate search, scene execution, unseen-room generalization or human usability claim.',
                    'new_model_calls':0,'new_geometry_queries':0,'new_scene_edits':0})
    print(json.dumps({'applicable':len(eligible),'agreement':sum(r['edit_stop_label_agreement'] for r in eligible),'unknown':len(rows)-len(eligible)}))


if __name__=='__main__':main()
