"""Ordinary explicit-priority projection on the actual development candidates.

No learned method: on this pool all priority vectors differ, so the tool rule
alone determines the order. This is a required control, not an Agent advantage.
"""
import argparse
from pathlib import Path
from cpu_continuation import read,dump


def priority(row):
    return (row['outside_floor_count'],row['wall_intersection_count'],
            row['new_furniture_intersection_count'],row['planar_B_status']!='witness',
            row['remaining_furniture_intersection_count'])


def project(ranking,facts):
    labels={x['label'] for x in facts}
    if ranking is None or len(ranking)!=len(labels) or set(ranking)!=labels:
        raise ValueError('Invalid native ranking; do not fabricate a model answer')
    positions={lab:i for i,lab in enumerate(ranking)}
    return [x['label'] for x in sorted(facts,key=lambda x:(priority(x),positions[x['label']]))]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    dest=root/'ordinary_projection.json'
    if dest.exists():raise FileExistsError(dest)
    facts=read(root/'development_pool_reference_quarantine.json')['facts']
    pred=read(root/'B_native_development_pool/turn_00/model/parsed_ranking.json')['ranking']
    ordered=project(pred,facts)
    # This only tests explicitly supplied subchecks. Even all of these passing
    # would not verify all of the original room's uses or human preferences.
    passing=[x['label'] for x in facts if x['outside_floor_count']==0 and x['wall_intersection_count']==0
             and x['remaining_furniture_intersection_count']==0 and x['planar_B_status']=='witness']
    result=dict(native_ranking=pred,ordinary_projected_ranking=ordered,
                ordinary_reference=read(root/'development_pool_reference_quarantine.json')['ordinary_rule_reference'],
                source_of_order='Explicit already supplied geometric-priority vectors; Critic only breaks equal-vector ties',
                unique_priority_vectors=len({priority(x) for x in facts}),candidate_count=len(facts),
                critic_necessary_to_determine_this_projected_order=False,
                best_partial_candidate=ordered[0],all_declared_subchecks_passed_candidates=passing,
                complete_scene_success=None,no_new_scene_or_model_call=True,no_training=True,
                role='Ordinary baseline implementation; not research innovation or a new generation result')
    assert ordered==result['ordinary_reference']
    dump(dest,result);print(result)


if __name__=='__main__':main()
