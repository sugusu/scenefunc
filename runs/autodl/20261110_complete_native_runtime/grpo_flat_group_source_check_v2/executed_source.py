"""Execute the pinned original GRPO zero-advantage branch on CPU scalars.

No model forward, parameter, optimizer or training. Checks the precise source
boundary: this implementation skips the objective (including KL) on a flat
local AND global advantage, rather than appealing to a generic RL statement.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    source=Path('/root/autodl-tmp/agent_method_preparation_20261027/work/context_credit_source_review_20261103/sources/train_grpo.py')
    data=source.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert blob=='459216738f7f1e4ce6a2219aa6431db6084eb04e'
    helper_root=Path('/root/autodl-tmp/agent_method_preparation_20261027/work/critic_scope_bridge_20261104/sources')
    helper=(helper_root/'src/grpo_objective.py').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(helper)).encode()+b'\0'+helper).hexdigest()=='54565bc136d5f39091aba1e42d9ca4f1870204e1'
    sys.path.insert(0,str(helper_root))
    function=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='grpo_loss_for_group')
    import torch
    visits=[]
    class NonflatRouteReached(Exception):pass
    def trap(*args,**kwargs):
        visits.append('completion_log_probs');raise NonflatRouteReached()
    ns=dict(torch=torch,BranchGroup=object,compute_completion_token_log_probs=trap)
    code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(source),'exec'),ns)
    rows=[]
    for rewards,global_adv in [([0.,0.,0.,0.],0.),([1.,1.,1.,1.],0.),([0.,0.,0.,0.],1.)]:
        before=len(visits)
        try:
            loss=ns['grpo_loss_for_group'](SimpleNamespace(rewards=rewards,global_adv=global_adv),
                 SimpleNamespace(device=torch.device('cpu')),None,None,.2,.05,1e-8)
            assert global_adv==0 and float(loss)==0 and len(visits)==before
            loss.backward()  # A stand-alone zero leaf, never a model parameter.
            rows.append(dict(rewards=rewards,global_adv=global_adv,status='literal_zero_returned_before_KL',
                             value=float(loss),completion_scoring_calls=0,standalone_zero_leaf_backward=True))
        except NonflatRouteReached:
            assert global_adv!=0 and len(visits)==before+1
            rows.append(dict(rewards=rewards,global_adv=global_adv,status='nonflat_route_reached_before_trap',
                             completion_scoring_calls=1,real_model_forward=False))
    result=dict(status='pinned_source_control_flow_executed',source=str(source),
                source_SHA256=hashlib.sha256(data).hexdigest(),git_blob_SHA1=blob,
                pin='a0565e50ed7d32c0e09132b9311c0b9b91165405',rows=rows,
                model_forwards=0,optimizer_updates=0,model_parameters_created=0,
                generic_RL_claim='Other objectives may retain KL/auxiliary gradients; this exact early-return branch does not',
                no_claim_about_other_GRPO_implementations=True)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
