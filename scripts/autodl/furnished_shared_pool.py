"""Post-trajectory common candidate-pool diagnostic, not new method success.

Both prompts see identical actual search states and measured consequences. The
ordinary selector uses the stated lexicographic rule; no expert preference gold.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from cpu_continuation import read, dump, sha
from furnished_pair_pilot import W, R, CODE, CPU, MODEL, model_launch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    for arm in ['B0','B1_repaired','B2']:
        assert read(root/arm/'state.json')['status']!='active'
    stages=[dict(checkpoint=str(CPU/'tools/checkpoint.json'),audit=str(CPU/'audit.json'),joint=str(CPU/'complete_finite_pool_joint.json'),views=str(CPU/'views_receipt.json'))]
    for p in sorted((root/'B2').glob('turn_*/checkpoint.json')):
        turn=p.parent
        if (turn/'views_receipt.json').exists():
            stages.append(dict(checkpoint=str(p),audit=str(turn/'audit.json'),joint=str(turn/'joint.json'),views=str(turn/'views_receipt.json')))
    random.Random(1729).shuffle(stages)
    baseline={tuple(p) for p in read(CPU/'audit.json')['furniture_surface_intersection_pairs']}
    basewalls={tuple(p) for p in read(CPU/'audit.json')['architecture_surface_intersection_pairs']['walls_doors_windows']}
    candidates=[];content=[]
    for i,stage in enumerate(stages):
        audit=read(stage['audit']);joint=read(stage['joint']);cid=f'C{i}'
        actual={tuple(p) for p in audit['furniture_surface_intersection_pairs']}
        row=dict(id=cid,scene_sha256=audit['scene_sha256'],B_planar_status=joint['status'],
                 intersections=sorted(actual),new_intersections=sorted(actual-baseline),
                 new_architecture_intersections=sorted({tuple(p) for p in audit['architecture_surface_intersection_pairs']['walls_doors_windows']}-basewalls),
                 outside_floor=audit['out_of_rectangular_floor_objects'],unknowns=audit['not_checked'])
        candidates.append(row)
        content.append(dict(type='text',text='Candidate '+cid+' (the next five images show this exact exported scene):'))
        views=read(stage['views'])
        for v in views['views']:
            assert sha(v['path'])==v['sha256']
            content.append(dict(type='image',image=v['path']))
    rule=lambda x:(len(x['outside_floor']),len(x['new_architecture_intersections']),len(x['new_intersections']),x['B_planar_status']!='witness',len(x['intersections']))
    best=min(rule(x) for x in candidates);ties=[x['id'] for x in candidates if rule(x)==best]
    protocol=dict(role='Post-trajectory shared-pool diagnostic; separates candidate availability from selection, not a new scene or independent sample',
                  pool_rule='initial checkpoint plus every actually executed B2 checkpoint, all retained, order shuffled seed1729',
                  ordinary_selector='lexicographic minimize outside-floor count, new architecture intersection count, new furniture intersection pair count, absence of B planar witness, residual furniture intersection count; report ties',
                  ordinary_ties=ties,candidates=candidates,sources=stages,model_calls_planned=2,
                  no_actions_executed=True,no_training=True,no_paid_API=True)
    # Freeze before either ranking answer. Labels are only ordinary computed
    # checks, and are not fed back into the earlier policy trajectories.
    dump(root/'shared_pool_protocol.json',protocol)
    instruction=('Choose one candidate from the actual supplied pool for the original B brief. First minimize outside-floor objects, then newly introduced architecture intersections, then newly introduced furniture intersection pairs, then prefer an evidenced B planar witness, then fewer remaining furniture intersection pairs. Report ties if equal. These checks do not establish complete human use or design preference. Return a JSON object with chosen_candidate, ties, evidence, remaining_unknowns. Do not emit a tool call or propose new coordinates.\nOriginal brief:\n'+read(W/'paris_A_inference_20261104/input_B.json')['brief']+'\nActual measured candidate evidence:\n'+json.dumps(candidates,ensure_ascii=False,separators=(',',':'))+'\nShared professional case:\n'+json.dumps(read(root/'common_case.json'),ensure_ascii=False,separators=(',',':')))
    for arm,system in [('B0_pool','Choose among actual scene candidates using the explicit brief, images and measured tool evidence.'),
                       ('B1_pool','Choose among actual scene candidates using the explicit brief, images and measured tool evidence. '+
                        'Infer the supplied case strategy preconditions; compare them with this brief and observations, separating evidence, assumptions and unknowns. Preserve required functions and only accept permitted sacrifices. Do not invent applicability labels.')]:
        turn=root/arm/'turn_00';turn.mkdir(parents=True,exist_ok=False)
        q=[dict(role='system',content=system),dict(role='user',content=content+[dict(type='text',text=instruction)])]
        dump(turn/'request.json',q)
        model_launch(root,turn)
    dump(root/'shared_pool_completed.json',dict(ordinary_ties=ties,actual_rank_outputs=[str(root/a/'turn_00/model/raw_output.txt') for a in ['B0_pool','B1_pool']],
                                              not_a_generation_comparison=True))


if __name__=='__main__':main()
