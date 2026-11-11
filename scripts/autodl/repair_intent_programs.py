"""Ordinary source-rule and labelled minimum-edit controls on saved responses.

The rule is a POST-HOC explanation of this small development fixture. The
minimum-edit teacher reads gold world labels unavailable to the student: it is
offline data construction, never a fair test-time4/4 Agent result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from intent_geometry_diagnostic import score


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def task_text(messages):
    text=messages[-1]['content'][-1]['text']
    return text[len('## Task\n'):].split('\n\n## Memory\n',1)[0]


def bind(active,facts):
    if active==['work','meal']:return facts['work_materials_plus_two_meal_regions_fit']
    if active==['work']:return facts['one_work_region_fits'] and facts['two_meal_regions_fit']
    return None


def source_rule(text):
    keep='资料用餐时仍留在同一工作面';clear='资料在用餐前全部收走'
    if (keep in text)==(clear in text):return None
    if '住户工作时需要' not in text or '住户与一位客人' not in text:return None
    return dict(people_by_phase=dict(work=['resident'],meal=['resident','guest']),
        materials_active_phases=['work','meal'] if keep in text else ['work'],
        evidence_quotes=[keep if keep in text else clear])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();start=time.monotonic()
    if (root/'ordinary_repair_controls.json').exists():raise FileExistsError('No implicit repair rerun')
    result=read(root/'model_result.json');assert result['status']=='completed' and result['program_arm_ran']
    cases=read(root/'cases_and_gold.json');bycase={c['id']:c for c in cases}
    attempts={r['case']:r for r in result['attempts'] if r['arm']=='program'}
    assert len(attempts)==4
    groups={}
    for case in cases:groups.setdefault(case['task_sha256'],[]).append(case)
    assert len(groups)==2 and all(len(v)==2 for v in groups.values())
    groups_out=[];teachers=[];rules=[]
    for task_hash,group in groups.items():
        seeds=[attempts[c['id']]['parsed'][0]['arguments'] for c in group]
        assert seeds[0]['materials_active_phases']==seeds[1]['materials_active_phases']
        assert all(s['people_by_phase']==dict(work=['resident'],meal=['resident','guest']) for s in seeds)
        # Only the diagnosed activation edge is editable. Work presence and
        # person identities are frozen domain assumptions, not inferred here.
        grammar=[['work'],['work','meal']]
        pool=[]
        for active in grammar:
            values=[bind(active,c['common_geometry']['geometry_fact']) for c in group]
            compatible=[v is c['expected_shared_feasible'] for v,c in zip(values,group)]
            distance=len(set(active)^set(seeds[0]['materials_active_phases']))
            pool.append(dict(materials_active_phases=active,world_verdicts=values,
                all_labelled_worlds_match=all(compatible),matches=compatible,activation_edge_edits=distance))
        valid=[p for p in pool if p['all_labelled_worlds_match']]
        assert valid
        best=min(p['activation_edge_edits'] for p in valid)
        selected=[p for p in valid if p['activation_edge_edits']==best]
        assert len(selected)==1
        chosen=selected[0]
        available={tuple(s['materials_active_phases']) for s in seeds}
        pool_survivors=[x for x in available if all(bind(list(x),c['common_geometry']['geometry_fact']) is c['expected_shared_feasible'] for c in group)]
        groups_out.append(dict(task_sha256=task_hash,case_ids=[c['id'] for c in group],
            observed_greedy_program_pool=[list(x) for x in sorted(available)],
            sampled_pool_contains_consistent_program=bool(pool_survivors),
            enumerated_candidates=pool,selected=chosen,
            new_candidate_created=tuple(chosen['materials_active_phases']) not in available,
            teacher_reads_gold_world_labels=True,
            label_authority='Research-authored task interpretation plus prior geometric certificates; not model consensus',
            novelty='Finite minimum-edit enumeration/CEGIS-style supervision, ordinary strong control'))
        for case,seed in zip(group,seeds):
            text=task_text(read(root/f"{case['id']}_program_request.json"))
            assert hashlib.sha256(text.encode()).hexdigest()==task_hash
            target=copy.deepcopy(seed);target['materials_active_phases']=chosen['materials_active_phases']
            call=dict(name='compile_activity_program',arguments=target)
            grade=score([call],'program',case,text);assert grade['correct']
            teachers.append(dict(case=case['id'],family='single_exposed_milan_family',
                source_request=str(root/f"{case['id']}_program_request.json"),
                source_model_receipt=attempts[case['id']]['receipt_dir'],original_model_arguments=seed,
                corrected_target=call,teacher_method='Ordinary labelled minimum-edit activation repair',
                activation_edge_edits=chosen['activation_edge_edits'],
                teacher_only_label_case_ids=[c['id'] for c in group],
                same_information_as_test_policy=False,chain_of_thought=None,
                fabricated_expert_rationale=False,training_used=False))
            rule=source_rule(text);assert rule is not None
            rule_grade=score([dict(name='compile_activity_program',arguments=rule)],'program',case,text)
            assert rule_grade['correct']
            rules.append(dict(case=case['id'],arguments=rule,score=rule_grade,
                post_hoc=True,reads_only_task_and_common_geometry=True,
                independent_generalization_measure=False))
    assert sum(g['new_candidate_created'] for g in groups_out)==1
    dump(root/'ordinary_repair_controls.json',dict(status='completed',groups=groups_out,
        source_rule_results=rules,all_four_posthoc_source_rule_cases_correct=True,
        arbitrary_semantic_repair_or_full_task_correctness_established=False,
        forward_calls=0,training_updates=0,paid_API_calls=0,wall_seconds=time.monotonic()-start,
        critical_scope='Offline oracle-labelled teacher reconstruction is not an inference policy comparison; posthoc rules explain existing cases only.'))
    dump(root/'repaired_program_teacher_records.json',dict(records=teachers,
        unit_count=4,unique_raw_task_texts=2,independent_source_families=1,
        admitted_for_research_training=False,
        reasons=['Custom component tool, not author full scene-generation action schema',
            'No held-out rooms or independently annotated natural briefs',
            'Oracle labels are used only by offline teacher; no policy result claimed',
            'No generated reasoning trace or actual geometry execution attributed to the teacher'],
        proposed_objective='Ordinary CE on corrected programs, or standard MML on all labelled-consistent grammar programs; not a new loss',
        training_updates=0))
    print(json.dumps(dict(status='ordinary_controls_completed',new_consistent_program_groups=1,
        posthoc_rule_correct=4,teacher_records=4,model_calls=0)),flush=True)


if __name__=='__main__':main()
