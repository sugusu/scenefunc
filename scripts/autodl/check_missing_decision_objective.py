"""Actual-corpus routing plus exact finite estimator comparisons on CPU.

No language-model forward, parameter update, scene edit or new annotations.
Finite tables test estimator semantics; they are not measured scene policies.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import time

import torch

from missing_decision_objective import (exact_marginal_nll, exact_expected_log_nll,
    latro_rloo_surrogate, rws_surrogate, route_partial_material)


def finite_difference(fn, x, epsilon=1e-6):
    values = []
    for i in range(len(x)):
        delta = torch.zeros_like(x); delta[i] = epsilon
        values.append(float((fn(x.detach()+delta)-fn(x.detach()-delta))/(2*epsilon)))
    return torch.tensor(values, dtype=x.dtype)


def enumerate_estimator(theta, logs, k, kind, reference, beta=0.):
    prior, _ = logs(theta)
    # Exhaust all ordered samples, including duplicates, with their real
    # sampling probabilities. No RNG error and no favorable-sample selection.
    total = torch.zeros_like(theta)
    probabilities = prior.detach().exp()
    probability_sum = 0.
    for indices in itertools.product(range(len(prior)), repeat=k):
        mass = float(probabilities[list(indices)].prod())
        lp, la = logs(theta)
        idx = list(indices)
        if kind == 'rws':
            loss, _ = rws_surrogate(lp[idx], la[idx], lp[idx].detach())
        else:
            loss, _ = latro_rloo_surrogate(lp[idx], la[idx], reference[idx], beta=beta)
        total += mass * torch.autograd.grad(loss, theta)[0]
        probability_sum += mass
    assert abs(probability_sum-1) < 1e-10
    return total, len(prior)**k


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--material',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(1)
    rows=[json.loads(x) for x in a.material.read_text().splitlines()]
    routed=route_partial_material(rows)
    # Preserve the counterpart's exact action and reason labels/coefficients.
    other={(c['unit_id'],c['stage']):c for c in rows if c['arm']=='separate_phase_v3'}
    for c in rows:
        if c['arm']!='paired_phase_v3':continue
        s=other[c['unit_id'],c['stage']]
        for key in ['messages','target_token_ids','token_objective_weights','source_token_origins']:
            assert c[key]==s[key],(c['unit_id'],key)
    assert len(routed)==24 and sum(x['reason_tokens']>0 for x in routed)==10
    (a.out/'unit_routes.json').write_text(json.dumps(routed,ensure_ascii=False,indent=2)+'\n')

    dtype=torch.float64
    # Two parameter blocks; the finite policy contains both reason and action
    # parameters. A shared-parameter case below rules out an accidental
    # independent-head-only implementation.
    theta=torch.tensor([math.log(.8),math.log(.15),math.log(.05),
                        math.log(.01/.99),math.log(.2/.8),math.log(.8/.2)],
                        dtype=dtype,requires_grad=True)
    def logs(t):return t[:3].log_softmax(0),torch.nn.functional.logsigmoid(t[3:])
    reference=torch.tensor([.5,.3,.2],dtype=dtype).log()
    lp,la=logs(theta)
    marginal=exact_marginal_nll(lp,la)
    gm=torch.autograd.grad(marginal,theta)[0]
    fd=finite_difference(lambda t:exact_marginal_nll(*logs(t)),theta)
    assert torch.allclose(gm,fd,atol=1e-9,rtol=1e-8)
    rws=[]
    for k in [1,2,4,6]:
        g,n=enumerate_estimator(theta,logs,k,'rws',reference)
        rws.append({'K':k,'enumerated_ordered_samples':n,'expected_gradient':g.tolist(),
                    'max_error_to_marginal':float((g-gm).abs().max())})
    assert max(abs(v) for v in rws[0]['expected_gradient'][:3])<1e-12
    assert float(gm[:3].abs().max())>.5
    latro=[]
    for beta in [0.,.1]:
        lp,la=logs(theta)
        objective=exact_expected_log_nll(lp,la,reference,beta=beta)
        truth=torch.autograd.grad(objective,theta)[0]
        fd2=finite_difference(lambda t:exact_expected_log_nll(*logs(t),reference,beta=beta),theta)
        assert torch.allclose(truth,fd2,atol=2e-9,rtol=1e-8)
        for k in [2,3,4]:
            g,n=enumerate_estimator(theta,logs,k,'latro',reference,beta)
            assert torch.allclose(g,truth,atol=1e-12,rtol=1e-12)
            latro.append({'K':k,'beta':beta,'enumerated_ordered_samples':n,
                          'expected_gradient':g.tolist(),'exact_objective_gradient':truth.tolist(),
                          'gradient_error':float((g-truth).abs().max()),
                          'finite_difference_error':float((truth-fd2).abs().max()),
                          'difference_from_marginal':float((g-gm).abs().max())})
    generator=torch.Generator().manual_seed(27)
    shared=torch.randn(5,generator=generator,dtype=dtype,requires_grad=True)
    rmap=torch.randn(3,5,generator=generator,dtype=dtype)
    amap=torch.randn(3,5,generator=generator,dtype=dtype)
    def shared_logs(t):return (rmap@t).log_softmax(0),torch.nn.functional.logsigmoid(amap@t)
    slp,sla=shared_logs(shared)
    sg=torch.autograd.grad(exact_expected_log_nll(slp,sla,reference,beta=.1),shared)[0]
    se,_=enumerate_estimator(shared,shared_logs,3,'latro',reference,.1)
    assert torch.allclose(se,sg,atol=1e-12,rtol=1e-12)

    # Fixed current-policy prior: Jensen gives E log p(A|D) <= log E p(A|D).
    # Example policies show the ranking can differ, so one must name the target.
    p=torch.tensor([.5,.5],dtype=dtype)
    options={'variable':[.99,.01],'consistent':[.4,.4]}
    objectives={name:{'marginal_probability':float((p*torch.tensor(v,dtype=dtype)).sum()),
                       'expected_log_probability':float((p*torch.tensor(v,dtype=dtype).log()).sum())}
                for name,v in options.items()}
    assert objectives['variable']['marginal_probability']>objectives['consistent']['marginal_probability']
    assert objectives['variable']['expected_log_probability']<objectives['consistent']['expected_log_probability']

    # Exercise every actual action coefficient; the latent block is a weighted
    # sequence likelihood, not a new independently normalized reason dataset.
    total_coefficient=sum(x['action_coefficient'] for x in routed if not x['reason_tokens'])
    lp,la=logs(theta)
    weighted=sum(exact_marginal_nll(lp,la,x['action_coefficient']) for x in routed if not x['reason_tokens'])
    weighted_gradient=torch.autograd.grad(weighted,theta)[0]
    mass_error=float((weighted_gradient-total_coefficient*gm).abs().max())
    assert mass_error<1e-12
    failures=[]
    for name,fn in [
      ('single_draw_RLOO',lambda:latro_rloo_surrogate(lp[:1],la[:1],reference[:1])),
      ('differentiable_proposal',lambda:rws_surrogate(lp,la,lp)),
      ('unnormalized_exact_prior',lambda:exact_marginal_nll(lp-.2,la))]:
        try:fn()
        except ValueError:failures.append(name)
        else:raise AssertionError(name)
    summary={
      'source':str(a.material),'source_sha256':hashlib.sha256(a.material.read_bytes()).hexdigest(),
      'units_retained':len(routed),'observed_reason_units':10,'latent_reason_units':14,
      'invented_reason_targets':0,'action_target_tokens':sum(x['action_tokens'] for x in routed),
      'reason_target_tokens':sum(x['reason_tokens'] for x in routed),
      'original_action_mass':sum(x['original_action_mass'] for x in routed),
      'original_reason_mass':sum(x['original_reason_mass'] for x in routed),
      'unannotated_action_coefficient_sum':total_coefficient,
      'real_corpus_weighted_gradient_error':mass_error,
      'finite_model':{'prior':[.8,.15,.05],'action_probability':[.01,.2,.8],
                      'marginal_nll':float(marginal.detach()),'exact_marginal_gradient':gm.tolist(),
                      'finite_difference_error':float((gm-fd).abs().max())},
      'rws_finite_K_bias':rws,'latro_on_policy_expectation':latro,
      'shared_parameter_gradient_error':float((sg-se).abs().max()),
      'objectives_rank_policies_differently':objectives,
      'invalid_estimators_rejected':failures,
      'actual_language_model_calls':0,'optimizer_updates':0,'new_scenes':0,
      'cuda_initialized':torch.cuda.is_initialized(),
      'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'scope':'Exact numerical controls and routing of actual immutable targets, not LLM training, annotation validation, or scene-quality evidence',
      'training_ready':False,'new_algorithm_claim':False}
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rws_finite_K_bias','latro_on_policy_expectation']},ensure_ascii=False))


if __name__=='__main__':main()
