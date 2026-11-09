"""Link existing supervision to observed multi-step completion boundaries.

The full fixed corpus stays in the ledger. Only the Waterloo source has the
mode-transition schema used below. Observed suffixes are retrospective witnesses,
not generated plans, independent projects, or positive professional labels.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from audit_material_routing import CORPUS, GitInputs, digest, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();src=GitInputs(args.repo);corpus=src.json(CORPUS)
    ledger=[];waterloo=[]
    for unit in corpus['rows']:
        inp,_=src.reference(unit['input']);ep,_=src.reference(unit['documents']['decision_action'])
        turn=ep['turns'][0];ctx=inp['context'];ev=ep.get('evaluation_only',{})
        has_reason=unit['target_kind']=='tool_action' and bool(turn.get('content'))
        calls=turn.get('tool_calls',[])
        ledger.append(dict(id=unit['id'],family_id=unit['family_id'],target_kind=unit['target_kind'],
            has_sourced_reason=has_reason,calls=calls,
            outcome_fields_present=sorted(k for k in ev if k not in ('curation','sentinel','study_origin')),
            current_mode=ctx.get('current_requested_mode'),
            whole_room_professional_target_proved=False,
            note='Source-backed principle and reconstructed action are distinct; fields present do not prove full-room acceptance.'))
        if unit['id'].startswith('waterloo_edit_'):
            waterloo.append(dict(unit=unit,context=ctx,has_reason=has_reason,calls=calls))
    source='runs/920/P920-WATERLOO-v1_20261105T074801Z/SUMMARY.json'
    summary=src.json(source);states=summary['rows']
    waterloo.sort(key=lambda r:int(r['unit']['id'].rsplit('_',1)[1]))
    assert len(states)==len(waterloo)+1==5
    phases=[]
    for index,r in enumerate(waterloo):
        mode=r['context']['current_requested_mode']
        if not phases or phases[-1]['mode']!=mode:phases.append(dict(mode=mode,indices=[]))
        phases[-1]['indices'].append(index)
    segments=[]
    for phase in phases:
        indices=phase['indices'];start_i,end_i=indices[0],indices[-1]+1
        records=waterloo[start_i:end_i];program=records[0]['context']['declared_activity_requirements']
        if any(r['context']['declared_activity_requirements']!=program for r in records):
            raise ValueError('Task contract changed inside a phase')
        needed=program['work_chair_ids'] if phase['mode']=='separate_work' else [s['chair_id'] for s in program['seat_assignments']]
        snapshots=[]
        for i in range(start_i,end_i+1):
            s=states[i];g=s['actual_geometry']
            align={cid:s['seat_alignment'][cid]['aligned_under_declared_proxy'] for cid in needed}
            access=all(all(row[cid] for cid in needed) for row in s['seat_approaches'])
            hard_geometry=not(g['outside_declared_wall_inner_faces'] or g['positive_3d_AABB_pairs'])
            snapshots.append(dict(state=s['state'],source_state_sha256=s['state_ref']['sha256'],
                current_mode=phase['mode'],requested_mode_available=s['mode_geometry'][phase['mode']],
                required_seat_alignment=align,required_seat_approaches=access,
                fixed_access_counts=s['fixed_access_counts'],endpoint_geometry_clear=hard_geometry,
                scoped_goal_available=bool(s['mode_geometry'][phase['mode']] and all(align.values()) and access and hard_geometry)))
        first=snapshots[0];first_edit=snapshots[1]
        losses=[cid for cid,v in first['required_seat_alignment'].items() if v and not first_edit['required_seat_alignment'][cid]]
        completion=next((i for i,s in enumerate(snapshots[1:],1) if s['scoped_goal_available']),None)
        segments.append(dict(mode=phase['mode'],action_ids=[r['unit']['id'] for r in records],
            reason_supervision_positions=[i+1 for i,r in enumerate(records) if r['has_reason']],
            recorded_calls=[r['calls'] for r in records],states=snapshots,
            first_action_loses_required_seat_alignment=losses,
            observed_prefix_to_complete_goal=completion,
            hypothetical_per_step_no_regression_filter_accepts_first=not losses,
            two_action_recorded_endpoint_meets_scoped_goal=snapshots[-1]['scoped_goal_available'],
            hard_geometry_clear_at_every_recorded_endpoint=all(s['endpoint_geometry_clear'] for s in snapshots),
            fixed_access_counts_preserved=all(s['fixed_access_counts']==first['fixed_access_counts'] for s in snapshots),
            original_instances_preserved_count=summary['objects_preserved'],
            oracle_uses_observed_future=True,training_admission=False,
            limitations=['Does not prove two actions are necessary or shortest over all legal plans.',
                         'Does not license temporary collisions, deletion, or access loss.',
                         'Endpoint geometry does not certify continuous motion or human use.']))
    result=dict(status='completed_archived_supervision_analysis',corpus_size=len(ledger),
        families=len({r['family_id'] for r in ledger}),
        target_kinds=dict(Counter(r['target_kind'] for r in ledger)),
        sourced_reason_states=sum(r['has_sourced_reason'] for r in ledger),
        analyzed_multi_step_segments=len(segments),analyzed_multi_step_families=1,
        scope='All fixed training units listed; multi-step comparison limited to the one existing measured mode-switch family. No new reference decisions or 3D scenes.',
        source_sha256=src.manifest,wall_seconds=time.perf_counter()-start,
        model_calls=0,optimizer_updates=0,native_tool_calls=0)
    write(args.output/'corpus_ledger.json',ledger)
    write(args.output/'observed_segments.json',segments)
    write(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
