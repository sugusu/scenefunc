"""Finite probability/gradient checks, with no model or optimizer updates.

These calculations qualify the objective, not a 3D method or TRICE replication.
The notebook is parsed only: its backend stubs and training cells never run.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--notebook',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(19)
    dtype=torch.float64
    # Both distributions depend on the SAME parameter vector. Tiny linear
    # probability tables are not language-model weights or trained parameters.
    theta=torch.tensor([.2,-.7,.4,.1,-.3],dtype=dtype,requires_grad=True)
    pd_map=torch.randn(3,5,dtype=dtype)
    pa_map=torch.randn(3,4,5,dtype=dtype)
    def logs(t):
        return (pd_map@t).log_softmax(0), (pa_map@t).log_softmax(1)
    def objective(t,valid):
        ld,la=logs(t)
        return torch.logsumexp((ld[:,None]+la)[:,valid].reshape(-1),0)
    cases=[]
    for valid in ([0],[0,2]):
        ld,la=logs(theta);joint=ld[:,None]+la
        logq=objective(theta,valid)
        truth=torch.autograd.grad(logq,theta,retain_graph=True)[0]
        posterior=(joint[:,valid]-logq).exp().detach()
        complete_score=(posterior*joint[:,valid]).sum()
        identity=torch.autograd.grad(complete_score,theta,retain_graph=True)[0]
        rationale_only=(posterior.sum(1)*ld).sum()
        incomplete=torch.autograd.grad(rationale_only,theta)[0]
        finite=[];eps=1e-6
        for i in range(theta.numel()):
            delta=torch.zeros_like(theta);delta[i]=eps
            finite.append(float((objective(theta.detach()+delta,valid)-objective(theta.detach()-delta,valid))/(2*eps)))
        finite=torch.tensor(finite,dtype=dtype)
        assert torch.allclose(truth,identity,atol=1e-12,rtol=1e-12)
        assert torch.allclose(truth,finite,atol=2e-9,rtol=2e-9)
        assert float((truth-incomplete).abs().max())>.01
        cases.append({'valid_actions':valid,'log_probability':float(logq.detach()),
                      'exact_gradient':truth.tolist(),'posterior_joint_score':identity.tolist(),
                      'finite_difference':finite.tolist(),'rationale_score_only':incomplete.tolist(),
                      'joint_score_error':float((truth-identity).abs().max()),
                      'finite_difference_error':float((truth-finite).abs().max()),
                      'rationale_only_error':float((truth-incomplete).abs().max())})
    nb=json.loads(a.notebook.read_text())
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tree=ast.parse(code)
    backend_stubs=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)
                   and n.name in ('sample','log_prob','grad','init')
                   and any(isinstance(t,ast.Raise) for t in ast.walk(n))]
    resample=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
              and isinstance(n.func,ast.Attribute) and n.func.attr=='resample_systematic']
    assert len(resample)==1
    expr=ast.unparse(resample[0].args[0]);assert expr=='jax.nn.log_softmax(flat_weights)'
    # Same nonnegative weights and signs as the notebook in a four-example
    # finite case; all memories are correct, only first proposal is correct.
    proposal=torch.tensor([1.,0.,0.,0.],dtype=dtype);mask=torch.ones(4,dtype=dtype)
    beta=(proposal.sum()-proposal)/(mask.sum()-1+1e-10)
    weights=torch.cat([mask*(1-beta*proposal),mask*beta*(1-proposal)])
    clipped=weights.clamp_min(1e-10)
    signs=torch.cat([mask,-mask]);scale=clipped.sum()/(mask.sum()+1e-10)
    probabilities_as_written=clipped.softmax(0)
    probabilities_weighted=clipped/clipped.sum()
    target_coefficients=signs*clipped/(mask.sum()+1e-10)
    written_coefficients=probabilities_as_written*signs*scale
    weighted_coefficients=probabilities_weighted*signs*scale
    assert torch.allclose(weighted_coefficients,target_coefficients,atol=1e-12,rtol=1e-12)
    assert float((written_coefficients-target_coefficients).abs().max())>.01
    result={'marginal_gradient_cases':cases,'shared_parameters':True,
            'notebook':{'path':str(a.notebook),'sha256':hashlib.sha256(a.notebook.read_bytes()).hexdigest(),
              'backend_stubs':backend_stubs,'resampler_log_probability_argument':expr,
              'weights':clipped.tolist(),'probabilities_as_written':probabilities_as_written.tolist(),
              'probabilities_proportional_to_weights':probabilities_weighted.tolist(),
              'target_gradient_coefficients':target_coefficients.tolist(),
              'written_expected_coefficients':written_coefficients.tolist(),
              'max_expected_coefficient_error':float((written_coefficients-target_coefficients).abs().max()),
              'scope':'AST plus exact finite expectation under documented resampler marginals. JAX/TFP notebook, model backend, and original experiments not executed.'},
            'model_calls':0,'optimizer_updates':0,'cuda_initialized':torch.cuda.is_initialized(),
            'wall_seconds':time.monotonic()-start,
            'novelty_claim':False,'three_dimensional_benefit_measured':False}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
