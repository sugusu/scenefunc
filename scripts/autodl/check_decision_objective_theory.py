"""Finite probability counterexample, not a neural or scene experiment."""
import argparse
import json
import math
from pathlib import Path


def distribution(p, conditional):
    assert len(p)==len(conditional) and abs(sum(p)-1)<1e-12
    assert all(0<x<1 for x in p+conditional)
    joint=p[0]*conditional[0]
    marginal=sum(x*y for x,y in zip(p,conditional))
    return dict(decision_probabilities=p,valid_action_given_decision=conditional,
                teacher_D_nll=-math.log(p[0]),teacher_A_nll=-math.log(conditional[0]),
                teacher_joint_nll=-math.log(joint),marginal_valid_action_probability=marginal,
                teacher_joint_probability=joint,teacher_path_lower_bound_holds=joint<=marginal)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    before=distribution([.1,.9],[.5,.9]);after=distribution([.2,.8],[.9,.1])
    assert after['teacher_D_nll']<before['teacher_D_nll']
    assert after['teacher_A_nll']<before['teacher_A_nll']
    assert after['teacher_joint_nll']<before['teacher_joint_nll']
    assert after['marginal_valid_action_probability']<before['marginal_valid_action_probability']
    # Both component losses decline, so every strictly positive weighting of
    # those two components declines. That does not constrain the other branch.
    weight_checks=[]
    for dweight,aweight in [(1,1),(.1,1),(1,.1),(3,7)]:
        b=dweight*before['teacher_D_nll']+aweight*before['teacher_A_nll']
        a=dweight*after['teacher_D_nll']+aweight*after['teacher_A_nll']
        assert a<b
        weight_checks.append(dict(D=dweight,A=aweight,before=b,after=a))
    result={'status':'passed','before':before,'after':after,'positive_weight_examples':weight_checks,
            'interpretation':'A better teacher path is a better lower bound but need not improve the full deployed mixture. This logical counterexample does not predict degradation from the actual training data.',
            'scope':'Ordinary probability identities and a constructed two-decision example, not a novel theorem or empirical 3D result.',
            'actual_objective_boundary':'The repository uses token/family/component weighted CE. Do not identify its scalar loss with unweighted sequence joint NLL or a marginal likelihood bound without undoing those weights.',
            'inference_boundary':'Stochastic mixture differs from greedy two-stage decoding. Neither teacher-path loss nor the mixture alone certifies greedy complete-scene success.',
            'model_calls':0,'training_updates':0,'scene_edits':0}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','before_probability':before['marginal_valid_action_probability'],'after_probability':after['marginal_valid_action_probability']}))


if __name__=='__main__':main()
