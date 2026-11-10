"""Execute pinned, reviewed upstream data builders; no model, tools or training.

The constructed examples are logical boundary witnesses, not sampled SceneOrchestra
outputs or estimates of that method's task failure rate.
"""
import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys
import types

from trajectory_supervision_contract import (ClauseEvidence, terminal_admission,
                                            certified_preference)


def main(stage, out):
    out.mkdir(parents=True, exist_ok=False)
    source_root=stage/'sceneorchestra_source'
    manifest=json.loads((source_root/'source_manifest.json').read_text())
    for f in manifest['files']:
        p=source_root/'source'/f['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
    package=types.ModuleType('sceneorchestra')
    package.__path__=[str(source_root/'source/src/sceneorchestra')]
    sys.modules['sceneorchestra']=package
    datasets=importlib.import_module('sceneorchestra.datasets')
    records=importlib.import_module('sceneorchestra.records')
    trajectory=importlib.import_module('sceneorchestra.trajectory')
    scoring=importlib.import_module('sceneorchestra.scoring')
    constants=importlib.import_module('sceneorchestra.constants')
    # A high scalar score alone cannot logically certify a separate task predicate.
    # No scene is executed or alleged to have these numbers.
    instruction='Place one bed and one wardrobe in a bedroom.'
    call=trajectory.ToolCall('init_gpt',{'ideas':instruction,'roomtype':'bedroom'})
    partial=records.RolloutRecord(instruction,'synthetic_score_boundary',[
        records.StepRecord(0,call,1.0,{}, {'composition':8.0})],
        metadata={'synthetic':True,'wardrobe_verified':False})
    built=datasets.build_trajectory_sft([partial],seed=0)
    assert len(built)==1
    rendered=trajectory.parse_trajectory(built[0]['response'])
    assert rendered[-1].name=='terminate' and rendered[-1].arguments['status']=='success'
    ids=['bed','wardrobe']
    evidence=[ClauseEvidence('bed','true','synthetic:bed',True),
              ClauseEvidence('wardrobe','unknown','synthetic:wardrobe',False)]
    guarded=terminal_admission(ids,evidence,terminal_reason='task_complete',execution_errors=0)
    assert not guarded['eligible_complete_positive']
    # A larger object count can raise this surrogate without changing the original
    # task clauses. This arithmetic does not establish how GPT or users score it.
    def metrics(n):
        return {'physics':{'object number':n,'object not inside':0,'collision':0},
                'visual':{'realism':8,'functionality':8,'layout':8,'completeness':8}}
    first=scoring.score_metric(metrics(4),1).to_dict()
    second=scoring.score_metric(metrics(14),1).to_dict()
    assert second['composition']>first['composition']
    # Exercise unmodified DPO construction on scalar preference evidence, retaining
    # its objective identity instead of calling it an independently verified outcome.
    comparison={'original_composition':8,'alternative_composition':12,
                'original_call':call.to_dict(),
                'alternative_call':trajectory.ToolCall('init_physcene',call.arguments).to_dict(),
                'prompt':'synthetic same-context comparison','next_number':1}
    dpo=datasets.build_stepwise_dpo([comparison])
    assert len(dpo)==1 and 'init_physcene' in dpo[0]['chosen']
    # Exhaustive tri-state contract check: this has no model parameters or optimizer.
    clause_ids=('instance','relation','geometry')
    checked=[]
    for truths in itertools.product(('true','false','unknown'),repeat=3):
        ev=[ClauseEvidence(k,v,'synthetic:'+k,True) for k,v in zip(clause_ids,truths)]
        got=terminal_admission(clause_ids,ev,terminal_reason='task_complete',execution_errors=0)
        assert got['eligible_complete_positive']==all(v=='true' for v in truths)
        checked.append(dict(values=truths,admitted=got['eligible_complete_positive']))
    all_true=[ClauseEvidence(k,'true','synthetic:'+k,True) for k in clause_ids]
    cutoff=terminal_admission(clause_ids,all_true,terminal_reason='actor_deadline',execution_errors=0)
    assert not cutoff['eligible_complete_positive']
    assert certified_preference(ids,evidence,evidence)=='not_identified'
    artifact=dict(pinned_source=manifest['pin'],synthetic=True,
                  interpretation='Logical target-construction boundary; no inference about real failure frequency.',
                  upstream_trajectory_SFT=built,upstream_stepwise_DPO=dpo,
                  explicit_task_admission=guarded,score_object_count_sensitivity=[first,second],
                  exhaustive_clause_cases=checked,cutoff_with_true_clauses=cutoff,
                  training_updates=0,model_forwards=0,paid_API_calls=0,
                  executed_source_files=[f for f in manifest['files'] if f['path'].endswith(
                    ('datasets.py','records.py','trajectory.py','scoring.py','constants.py','prompts.py'))])
    (out/'result.json').write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps(dict(status='unmodified_pinned_target_builders_executed',
           synthetic_boundary_upstream_STOP_success=True,task_positive_admitted=False,
           original_composition_threshold=constants.PAPER.trajectory_sft_threshold,
           model_forwards=0,training_updates=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.stage.resolve(),a.out.resolve())
