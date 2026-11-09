"""Finite probability counterexamples for theory scope, not 3D experiments."""
import argparse
from collections import defaultdict
from itertools import product
import json
import math
from pathlib import Path


def entropy(rows, columns):
    marginal = defaultdict(float)
    for row, probability in rows:
        marginal[tuple(row[c] for c in columns)] += probability
    return -sum(p * math.log2(p) for p in marginal.values() if p)


def mi(rows, left, right, given=()):
    return (entropy(rows, left+given) + entropy(rows, right+given)
            - entropy(rows, given) - entropy(rows, left+right+given))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing result')
    collapsed=[({'X':b,'Y':b,'H':0},.5) for b in (0,1)]
    chain=[({'H':0,'C':b,'Y':b},.5) for b in (0,1)]
    teacher=[({'I':i,'Y':i^z,'D_teacher':i^z,'D_self':i^r},.125) for i,z,r in product((0,1),repeat=3)]
    result={
        'noninvertible_encoder':{'I_Y_X':mi(collapsed,('Y',),('X',)), 'I_Y_H':mi(collapsed,('Y',),('H',)),
          'conclusion':'H=f(X) only implies data processing inequality; equality requires a sufficient/invertible encoding.'},
        'general_chain_identity':{'I_Y_H':mi(chain,('Y',),('H',)),
          'I_Y_C_plus_I_Y_H_given_C':mi(chain,('Y',),('C',))+mi(chain,('Y',),('H',),('C',)),
          'omitted_I_Y_C_given_H':mi(chain,('Y',),('C',),('H',)),
          'conclusion':'Use I(Y;H)=I(Y;C)+I(Y;H|C)-I(Y;C|H). The capacity UPPER BOUND survives by dropping the nonnegative last term; an equality need not hold.'},
        'teacher_vs_self':{'I_Y_teacher_given_I':mi(teacher,('Y',),('D_teacher',),('I',)),
          'I_Y_self_given_I':mi(teacher,('Y',),('D_self',),('I',)),
          'conclusion':'A gold-conditioned teacher can provide target information absent from I; a fixed model using only I and independent randomness cannot. This does not rule out computational/optimization benefits.'},
        'scope':'Exact enumerated toy distributions; not a theorem of 3D generation, measured model behavior or evidence about real-user preferences.',
        'model_calls':0,'training_updates':0,'new_scene_executions':0}
    assert result['noninvertible_encoder']['I_Y_X']==1 and result['noninvertible_encoder']['I_Y_H']==0
    assert result['general_chain_identity']['I_Y_C_plus_I_Y_H_given_C']==1
    assert result['general_chain_identity']['omitted_I_Y_C_given_H']==1
    assert result['teacher_vs_self']['I_Y_teacher_given_I']==1
    assert result['teacher_vs_self']['I_Y_self_given_I']==0
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
