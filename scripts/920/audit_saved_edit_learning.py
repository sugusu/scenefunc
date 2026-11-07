"""Recover pre-update component losses from immutable training receipts; no model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(Path(p).read_text())
def ref(p): return {'path': str(p), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    plan=read(a.training/'TRAIN_PLAN.json');enc=read(a.training/'ENCODING.json');result=read(a.training/'TRAINING_RESULT.json')
    assert ref(a.training/'ENCODING.json')['sha256']==plan['encoding']['sha256']
    assert result['status']=='completed' and result['exit_code']==0
    sources=[ref(a.training/n) for n in ['ENCODING.json','TRAIN_PLAN.json','TRAINING_RESULT.json']]
    # Log/example association depends on the actually frozen sequential sampler.
    for path,digest in plan['code_files'].items():
        assert ref(path)['sha256']==digest,path
    arms={}
    for arm in plan['arm_order']:
        rows=[r for r in enc['records'] if r['arm']==arm]
        path=a.training/'training_progress'/f'{arm}_MICRO_LOSS.json';logs=read(path);sources.append(ref(path))
        assert len(rows)==24 and len(logs)==8*24
        totals={f:{c:sum(r[c+'_tokens'] for r in rows if r['family_id']==f) for c in ['action','decision']} for f in {r['family_id'] for r in rows}}
        steps=[]
        for step in range(8):
            component=defaultdict(float);examples=[]
            for r,v in zip(rows,logs[24*step:24*(step+1)]):
                assert v['optimizer_step_before']==step
                x={'id':r['id'],'family_id':r['family_id'],'has_gold_decision':bool(r['decision_tokens'])}
                for c in ['action','decision']:
                    n=totals[r['family_id']][c];expected=24/(6*n) if n else 0
                    assert v[c]['tokens']==r[c+'_tokens'] and abs(v[c]['weight']-expected)<1e-12
                    assert v[c]['nll_sum']>=0
                    component[c]+=v[c]['nll_sum']*v[c]['weight']/24
                    x[c+'_mean_nll']=v[c]['nll_sum']/v[c]['tokens'] if v[c]['tokens'] else None
                examples.append(x)
            logged=result['arms'][arm]['training']['training_log'][step]['loss']
            assert abs(sum(component.values())-logged)<0.000051
            steps.append({'optimizer_updates_before_forward':step,'weighted_components':dict(component),'examples':examples})
        arms[arm]={'steps':steps,'first_to_last_preupdate_relative_decrease':{c:(steps[0]['weighted_components'][c]-steps[-1]['weighted_components'][c])/steps[0]['weighted_components'][c] if steps[0]['weighted_components'][c] else None for c in ['action','decision']}}
    # Identical no-decision examples at initial weights provide a log-order cross-check.
    left=arms['action_only']['steps'][0]['examples'];right=arms['decision_action']['steps'][0]['examples']
    matched=[x['id'] for x,y in zip(left,right) if not y['has_gold_decision'] and x['action_mean_nll']==y['action_mean_nll']]
    assert len(matched)==16
    output={'status':'verified_saved_preupdate_losses','arms':arms,'initial_common_examples_exact':matched,'sources':sources,
            'limitations':['Last log observes weights after seven updates, not final eight-update adapter.',
            'Joint action loss conditions on teacher-forced gold decisions at eight examples; cross-arm raw losses are not a fair learning comparison.',
            'These training-mode likelihoods do not measure freely generated actions, room utility or generalization.'],
            'new_model_calls':0,'training_updates':0}
    (a.out/'SAVED_TRAINING_AUDIT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({arm: {'first':v['steps'][0]['weighted_components'],'last_preupdate':v['steps'][-1]['weighted_components'],'relative_decrease':v['first_to_last_preupdate_relative_decrease']} for arm,v in arms.items()},indent=2))


if __name__=='__main__':main()
