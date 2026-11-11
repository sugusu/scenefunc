"""Read-only static audit of the pinned public SceneExplorer generation source.

Never imports/runs the author module, starts its process pool, writes author
outputs or changes geometry. A static audit is not a reproduction experiment.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    receipt=json.loads((a.source/'code_receipt.json').read_text())
    byname={x['name']:x for x in receipt['files']}
    parsed={}
    for name in ['main.py','data.py']:
        raw=(a.source/name).read_bytes();ref=byname[name]
        assert hashlib.sha256(raw).hexdigest()==ref['sha256']
        assert hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()==ref['git_blob_sha1']
        parsed[name]=ast.parse(raw.decode(),filename=str(a.source/name))
    tree=parsed['main.py'];text=(a.source/'main.py').read_text()
    definitions=[dict(name=x.name,kind=type(x).__name__,start=x.lineno,end=x.end_lineno)
                 for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))]
    constants={}
    for x in tree.body:
        if isinstance(x,ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0],ast.Name):
            try:constants[x.targets[0].id]=ast.literal_eval(x.value)
            except (ValueError,TypeError):pass
    action=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='action_2')
    index_loop=next(x for x in ast.walk(action) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='i'
                    and any(isinstance(y,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='main_poly' for t in y.targets) for y in x.body))
    assignment=next(x for x in index_loop.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='main_poly' for t in x.targets))
    source_line=text.splitlines()[assignment.lineno-1]
    assert 'mod.state_list[mod.target_state].poly' in source_line
    module_state=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ModuleState')
    ctor=next(x for x in module_state.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    fields=[x.arg for x in ctor.args.args if x.arg!='self']
    data=json.loads((a.source/'public_asset_directory_metadata.json').read_text())
    result=dict(pin=receipt['pin'],source_verified=True,author_code_executed=False,
        complete_source_text_read=['main.py:1-1948','data.py:1-109'],definitions=definitions,
        literal_constants=constants,actual_CLI_default_grid_m=0.5,
        grid_note='Top-level discrete_step=0.25 is overwritten by main(0.5) without CLI step. This is a code default, not verified paper experiment setting.',
        current_generation_contract=dict(input='One initial room and fixed identity set of annotated transformable modules',
            candidate_generator='Combinations of module states, spatial priors and grouped placements',
            retained_candidates_per_state=5,layout_attempt_rounds=5,maximum_function_state_branches=1000,
            verification='2D polygon containment/intersection, transfer paths and time scheduling',
            state_inventory_changes='No new module identity addition/deletion inside search'),
        module_state_constructor_fields=fields,
        implementation_observation=dict(function='action_2',line=assignment.lineno,source=source_line.strip(),
            effect='The size list uses the same existing mod variable in each i iteration, rather than indexing mods[i]. This block alone does not implement the paper-described per-module size ordering.',
            static_only=True,numeric_impact_measured=False,patch_applied=False),
        branch_observation='When every module has one state, search enumerates only the initial branch and skips it; this entry point alone does not generate alternative static placements in that case. get_all_paths can still plan between externally given static endpoints.',
        hierarchy_scope='Paper functionality vector/hierarchy calculations are not reproduced by reading this generation file; other platform code and data remain separate.',
        released_directory_metadata=dict(initial_scene_files=len(data['scenes']),module_directory_entries=len(data['object']),
            full_repository_expert_data_census=False,human_schedules_acquired=0),
        training_examples_admitted=0,model_calls=0,training_updates=0,scene_generation_runs=0)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='pinned_source_statically_audited',definitions=len(definitions),author_code_executed=False)))


if __name__=='__main__':
    main()
