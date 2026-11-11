"""Ordinary voting/filtering and objective audit on certified fixed-world data.

No model, optimizer step, invented human vote or program preference annotation.
The likelihood examples below are mathematical objective controls only.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

R=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();start=time.monotonic()
    if (root/'ordinary_baselines.json').exists():raise FileExistsError('No implicit baseline rerun')
    result=read(root/'final_result.json');assert len(result['rows'])==2
    ids=list(result['rows'][0]['hypothesis_outputs'])
    mat=np.array([[r['hypothesis_outputs'][h] for r in result['rows']] for h in ids],dtype=int)
    source=R/'runs/autodl/20261111_tool_contract_semantics/source/worldcode.py'
    names=['get_vote_score','get_weighted_vote_score_soft_type','get_weighted_vote_score']
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    weights=np.ones((len(ids),1));scores={
        'author_hard_uniform':scope['get_vote_score'](mat),
        'author_weighted_hard_uniform':scope['get_weighted_vote_score'](mat,weights),
        'author_weighted_soft_uniform':scope['get_weighted_vote_score_soft_type'](mat,weights)}
    threshold=.8
    retained={name:[i for i,s in zip(ids,v) if s>=threshold] for name,v in scores.items()}
    assert retained['author_hard_uniform']==['required_persistence']
    # Literal paper expression vs ordinary reference-model DPO. We do not
    # infer an unpublished implementation, and assign no preference to scenes.
    examples=[];beta=.1
    for theta_odds,ref_odds in [(0.,0.),(2.,2.),(-2.,-2.),(2.,-2.)]:
        zd=beta*(theta_odds-ref_odds);zf=beta*theta_odds
        def loss(z):return max(-z,0)+math.log1p(math.exp(-abs(z)))
        def derivative(z):return -beta/(1+math.exp(z))
        examples.append(dict(synthetic_log_policy_odds=theta_odds,synthetic_log_reference_odds=ref_odds,
            standard_DPO_loss=loss(zd),reference_free_written_expression_loss=loss(zf),
            derivative_wrt_policy_log_odds_standard=derivative(zd),
            derivative_wrt_policy_log_odds_reference_free=derivative(zf)))
    dump(root/'ordinary_baselines.json',dict(programs=ids,known_authored_execution_matrix=mat.tolist(),
        source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            URL='https://github.com/klee972/exec-filter',revision='3d6130a9a42450b6d95455081fe4ecb34b20b342',
            executed_functions=names),
        scores=scores,threshold=threshold,retained=retained,
        labelled_consistency_survivors=read(root/'labelled_program_disambiguation.json')['survivors'],
        interpretation='Original hard voting also selects the correct hypothesis on this3x2construction. No advantage over that ordinary baseline established.',
        caveat='The three hypotheses and uniform weights were authored, not sampled from a trained model; no published benchmark or policy success rate.',
        model_calls=0,training_updates=0,paid_API_calls=0,wall_seconds=time.monotonic()-start))
    dump(root/'objective_reference_control.json',dict(
        source='SpatialGrammar2604.27555v1 displayed equation5; official project currently marks code coming soon',
        paper_expression='-log sigmoid(beta*(log pi_theta(y_w|x)-log pi_theta(y_l|x)))',
        ordinary_DPO='-log sigmoid(beta*((log pi_theta(y_w|x)-log pi_theta(y_l|x))-(log pi_ref(y_w|x)-log pi_ref(y_l|x))))',
        scope='Audit of written objectives, not a claim that unpublished training code implements the displayed reference-free formula.',
        common_reference_omission_changes_loss_except_zero_reference_log_odds=True,
        beta=beta,examples=examples,labels='Synthetic mathematical scalars, no scene preference labels',
        optimizer_steps=0,not_a_new_loss=True))
    print(json.dumps(dict(retained=retained,wall_seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
