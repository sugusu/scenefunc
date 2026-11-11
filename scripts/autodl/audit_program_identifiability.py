"""Label-identifiability construction on previously measured complete-room data.

This is a deliberately constructed finite semantic-program test, not observed
model mistakes, new 3D generation, a novel filter, or a replication of NLVR/WTQ.
Actual unmodified author vote functions are AST-extracted and run on the matrix.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from shapely.geometry import shape

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def votes(source):
    names=['get_vote_score','get_weighted_vote_score_soft_type','get_weighted_vote_score']
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    scope={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),scope)
    return {k:scope[k] for k in names}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    if (root/'identifiability_result.json').exists():raise FileExistsError('No implicit rerun')
    start=time.monotonic();base=Path('runs/autodl/20261111_activity_resource_program').resolve();milan=Path('runs/autodl/20261111_milan_completion').resolve()
    dependencies=[base/'compiled_programs.json',base/'execution_result.json',base/'protocol.json',milan/'completed_room_state.json',
        milan/'source/runs/920/P920-MILAN-TASK-v1_20261105T224015Z/input_metadata/house_state.json',root/'source/worldcode.py']
    before={str(p):sha(p) for p in dependencies};dump(root/'inputs_before.json',before)
    rows={x['id']:x for x in read(base/'compiled_programs.json')['rows']};exe=read(base/'execution_result.json')
    actual={x['task']+'_'+x['plan']:x for x in exe['executions'] if x['policy']=='ordinary'}
    # Check real polygon witnesses, not just success strings. Individual work
    # and the two meals can each fit; persistence requires their joint fit.
    top=shape(read(base/'measured_table_surface.json')['polygon'])
    for case in rows.values():
        assert top.buffer(1e-8).covers(shape(case['independent_work_witness']['polygon']))
        ms=case['independent_meal_pair']['witnesses'];north=shape(ms['meal_north']['polygon']);south=shape(ms['meal_south']['polygon'])
        assert top.buffer(1e-8).covers(north) and top.buffer(1e-8).covers(south) and north.intersection(south).area==0
    # Seed is an existing native full-scene completion; the two probes use the
    # existing native initial scene. Probe dimensions differ, so these are
    # parameterized TASK-FAMILY tests, not same-utterance world interventions.
    contexts=[dict(id='completed_compact_seed',case='compact_materials',native='independent'),
              dict(id='initial_compact_probe',case='compact_materials',native='shared'),
              dict(id='initial_expansive_probe',case='expansive_materials',native='shared')]
    state=read(milan/'completed_room_state.json');h=read(dependencies[4]);initial=h['rooms'][next(iter(h['rooms']))]
    states={'independent':state,'shared':initial}
    assert all(state['objects'][k]==v for k,v in initial['objects'].items())
    program_names=['persistent_sequential','omit_material_persistence','force_distinct_surfaces','add_unrequested_worker']
    matrix=[];facts=[]
    for ctx in contexts:
        case=rows[ctx['case']];s=states[ctx['native']]
        tables=[k for k in ['meal_table_0','ordinary_work_table_0'] if k in s['objects']]
        chairs=[k for k in ['meal_chair_n_0','meal_chair_s_0','ordinary_work_chair_0'] if k in s['objects']]
        shared=(case['shared']['status']=='witness')
        if not shared:assert case['shared']['status']=='infeasible_under_disjoint_planar_contract'
        separate=len(tables)>=2 and len(chairs)>=3 and case['independent_top_capacity_verified']
        regular=(len(chairs)>=2 and shared) or separate
        omitted=bool(case['independent_work_witness']) and case['independent_meal_pair']['status']=='witness' and len(chairs)>=2
        extra_worker=separate  # distinct extra worker uses verified independent desk+seat
        col=[bool(regular),bool(omitted),bool(separate),bool(extra_worker)];matrix.append(col)
        facts.append(dict(**ctx,native_objects=len(s['objects']),table_ids=tables,chair_ids=chairs,
            shared_joint_regions_verified=shared,individual_work_and_meal_regions_verified=True,
            independent_binding_and_capacity_verified=bool(separate),source_case_kind=case['kind'],
            original_brief_complete_success=None,physical_material_objects_created=False))
    matrix=np.array(matrix,dtype=int).T
    assert matrix.shape==(4,3) and np.all(matrix[:,0]==1)
    funcs=votes(root/'source/worldcode.py');weight=np.ones((4,1))
    results={
        'author_unweighted_hard':funcs['get_vote_score'](matrix),
        'author_uniform_weighted_hard':funcs['get_weighted_vote_score'](matrix,weight),
        'author_uniform_weighted_soft':funcs['get_weighted_vote_score_soft_type'](matrix,weight)}
    # Explicit paper NLVR threshold, not quality-tuned for the constructed case.
    threshold=.8
    filtered={k:[n for n,s in zip(program_names,v) if s>=threshold] for k,v in results.items()}
    labels=matrix[0].tolist()  # known typed contract; not guessed from vote or geometry alone
    remaining=list(range(4));steps=[]
    for idx in [0,1,2]:
        remaining=[i for i in remaining if matrix[i,idx]==labels[idx]]
        steps.append(dict(context=contexts[idx]['id'],provided_contract_label=labels[idx],remaining=[program_names[i] for i in remaining]))
    minsets=[]
    for k in range(4):
        for subset in itertools.combinations(range(3),k):
            keep=[i for i in range(4) if all(matrix[i,j]==labels[j] for j in subset)]
            if keep==[0]:minsets.append(list(subset))
        if minsets:break
    dump(root/'input_program_matrix.json',dict(programs=program_names,contexts=facts,execution_matrix=matrix.tolist(),
        constructed_hypotheses_not_model_predictions=True,
        known_reference_contract='Materials persist across sequential work/meal modes, same resident, sharing allowed',
        reference_labels=labels,reference_labels_source='Manually declared synthetic numeric task contract; not human preference',
        matched_only_in_limited_construction=True,geometry_not_rerun=True,
        actual_full_scene_generations=0,new_room_count=0,source_independent_room_count=1,
        task_family_not_same_utterance_world_intervention=True))
    dump(root/'identifiability_result.json',dict(author_function_scores=results,author_filter_threshold=threshold,
        retained=filtered,label_anchored_filter=steps,minimum_test_subsets=minsets,
        all_four_fit_seed=True,unlabeled_signature_class_count=len({tuple(x) for x in matrix.tolist()}),
        correct_program_identified_only_with_known_reference_labels=True,
        examples_of_existing_methods_not_new_algorithm=True,model_calls=0,training_updates=0,paid_API_calls=0,
        wall_seconds=time.monotonic()-start))
    with (root/'summary.csv').open('w') as f:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(['program',*[x['id'] for x in contexts],*results])
        for i,n in enumerate(program_names):wr.writerow([n,*matrix[i].tolist(),*[v[i] for v in results.values()]])
    after={str(p):sha(p) for p in dependencies};assert after==before
    dump(root/'inputs_after.json',dict(before=before,after=after,all_equal=True))
    print(json.dumps(dict(matrix=matrix.tolist(),programs=program_names,retained=filtered,
        labelled_final=steps[-1]['remaining'],minimum_subsets=minsets,wall_seconds=time.monotonic()-start)))

if __name__=='__main__':main()
