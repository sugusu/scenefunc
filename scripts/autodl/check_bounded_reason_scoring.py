"""Stopped-distribution enumeration, official helper execution and archive audit.

No language-model forward, optimizer, new model answer or scene edit. Synthetic
logits test numerical identities; archived text is never given invented logits.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import time

import torch

from bounded_reason_scoring import (reason_outcome, sequence_logprob,
    bounded_expected_log_nll, bounded_rloo_surrogate, bounded_marginal_nll,
    bounded_rws_surrogate)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def leaves(prefix=(), cap=3):
    for token in range(4):
        z=prefix+(token,)
        if token in (1,2) or len(z)==cap:yield z
        else:yield from leaves(z,cap)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--processor',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic();torch.set_num_threads(1)
    source=Path('runs/autodl/20261110_missing_decision_objective/source')
    manifest=json.loads((source/'manifest.json').read_text())
    for entry in manifest['files']:assert sha(source/entry['path'])==entry['sha256']
    # Execute only two pinned pure upstream functions. The first-true helper
    # below supplies its documented n-if-absent behavior, not the full trainer.
    def first_true_indices(mask):
        n=mask.shape[-1]
        return torch.where(mask,torch.arange(n),n).amin(-1)
    ns={'torch':torch,'first_true_indices':first_true_indices}
    path=source/'LaTRO/utils/trainer_utils.py'
    chosen=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
            and n.name in {'detect_stop_sequence_starts','truncate_response'}]
    assert len(chosen)==2
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),ns)
    examples=torch.tensor([[0,1,9],[0,0,1],[2,9,9],[0,0,0]])
    cut,ends=ns['truncate_response']([1,2],9,examples)
    assert cut.tolist()==[[0,9,9],[0,0,9],[9,9,9],[0,0,0]]
    upstream={'commit':manifest['commit'],'source_sha256':sha(path),
        'input':examples.tolist(),'truncated':cut.tolist(),'ending_index':ends.tolist(),
        'finding':'The sampled stop token is excluded from the subsequent rationale score.',
        'scope':'Isolated source helper with a local first-true equivalent; no full trainer run.'}
    write(a.out/'upstream_stop_helper.json',upstream)

    zs=list(leaves());assert len(zs)==22
    # 0=ordinary content, 1=close, 2=EOS, 3=malformed content. State-dependent
    # token probabilities, two continuation tokens, natural stops and a cap.
    theta=torch.tensor([.8,-.2,-1.1,-.7, .4,.1,-.8,-.2, .2,.4,-.3,-.6, .7],
                       dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([z[-1]==1 and 3 not in z for z in zs])
    reference=torch.tensor([-len(z)*math.log(4) for z in zs],dtype=theta.dtype)
    def logs(t):
        token=t[:12].reshape(3,4).log_softmax(-1)
        lp=torch.stack([sum(token[i,v] for i,v in enumerate(z)) for z in zs])
        action=torch.stack([torch.nn.functional.logsigmoid(t[-1]+.15*t[0]+.1*len(z)) for z in zs])
        return lp,torch.where(valid,action,torch.zeros_like(action))
    lp,la=logs(theta);assert abs(float(lp.exp().sum())-1)<1e-12
    objective=bounded_expected_log_nll(lp,la,valid,reference,beta=.1,invalid_reward=-4.)
    exact=torch.autograd.grad(objective,theta)[0]
    delta=1e-6;fd=[]
    for i in range(len(theta)):
        e=torch.zeros_like(theta);e[i]=delta
        def loss(t):return bounded_expected_log_nll(*logs(t),valid,reference,beta=.1,invalid_reward=-4.)
        fd.append(float((loss(theta.detach()+e)-loss(theta.detach()-e))/(2*delta)))
    fd=torch.tensor(fd,dtype=theta.dtype)
    expected=torch.zeros_like(theta);probability_sum=0.
    probabilities=lp.detach().exp()
    for pair in itertools.product(range(len(zs)),repeat=2):
        idx=list(pair);mass=float(probabilities[idx].prod());lpa,laa=logs(theta)
        surrogate=bounded_rloo_surrogate(lpa[idx],laa[idx],valid[idx],reference[idx],beta=.1,invalid_reward=-4.)
        expected+=mass*torch.autograd.grad(surrogate,theta)[0];probability_sum+=mass
    assert abs(probability_sum-1)<1e-12
    assert float((expected-exact).abs().max())<2e-12
    assert float((fd-exact).abs().max())<2e-8
    lp,la=logs(theta)
    dropped=-(lp[valid].softmax(0)*la[valid]).sum()
    dropped_grad=torch.autograd.grad(dropped,theta)[0]
    lp,la=logs(theta)
    fixed_no_kl=bounded_expected_log_nll(lp,la,valid,reference,beta=0.,invalid_reward=-4.)
    fixed_grad=torch.autograd.grad(fixed_no_kl,theta)[0]
    token=theta[:12].reshape(3,4).log_softmax(-1)
    omitted=torch.stack([sum((token[i,v] for i,v in enumerate(z[:-1] if z[-1] in (1,2) else z)),theta.sum()*0) for z in zs])
    omitted_mass=float(omitted.detach().exp().sum())
    wrong_expected_score=torch.autograd.grad((probabilities*omitted).sum(),theta)[0]
    lp,la=logs(theta)
    correct_expected_score=torch.autograd.grad((probabilities*lp).sum(),theta)[0]
    assert float(correct_expected_score.abs().max())<1e-12
    assert float(wrong_expected_score.abs().max())>.01 and omitted_mass>1.1
    finite={'stopped_outcomes':len(zs),'ordered_K2_samples':len(zs)**2,
        'valid_probability':float(probabilities[valid].sum()),'full_probability':float(probabilities.sum()),
        'finite_difference_max_error':float((fd-exact).abs().max()),
        'RLOO_expected_gradient_max_error':float((expected-exact).abs().max()),
        'fixed_failure_utility':-4.,'fixed_utility_is_a_surrogate_not_functional_quality':True,
        'drop_failed_samples_gradient_difference':float((dropped_grad-fixed_grad).abs().max()),
        'omit_actual_stop_probability_mass':omitted_mass,
        'correct_expected_score_max_abs':float(correct_expected_score.abs().max()),
        'omitted_stop_expected_score_max_abs':float(wrong_expected_score.abs().max()),
        'tree':[{'tokens':list(z),'valid':bool(valid[i]),'probability':float(probabilities[i])} for i,z in enumerate(zs)]}
    write(a.out/'finite_distribution.json',finite)

    # A finite fixed failure utility can prefer malformed output when a valid
    # long reference action has a smaller sequence log probability.
    h=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    gate=torch.stack([torch.nn.functional.logsigmoid(h),torch.nn.functional.logsigmoid(-h)])
    target=torch.tensor([-20.,0.],dtype=h.dtype);good=torch.tensor([True,False])
    fixed=bounded_expected_log_nll(gate,target,good,gate.detach(),beta=0.,invalid_reward=-4.)
    fixed_g=torch.autograd.grad(fixed,h,retain_graph=True)[0]
    marginal=bounded_marginal_nll(gate,target,good)
    marginal_g=torch.autograd.grad(marginal,h)[0]
    assert float(fixed_g)==4. and abs(float(marginal_g)+.5)<1e-12
    lp,la=logs(theta);truth_marginal=torch.autograd.grad(bounded_marginal_nll(lp,la,valid),theta)[0]
    rws_expected=torch.zeros_like(theta);zero_batch_mass=0.;zero_batches=0
    for pair in itertools.product(range(len(zs)),repeat=2):
        idx=list(pair);mass=float(probabilities[idx].prod());lpa,laa=logs(theta)
        loss,receipt=bounded_rws_surrogate(lpa[idx],laa[idx],valid[idx],lpa[idx].detach())
        assert receipt['samples']==2 and not receipt['resampling_authorized']
        if loss is None:
            zero_batch_mass+=mass;zero_batches+=1
        else:rws_expected+=mass*torch.autograd.grad(loss,theta)[0]
    assert abs(zero_batch_mass-float((1-probabilities[valid].sum())**2))<1e-12
    write(a.out/'objective_selection.json',{
        'fixed_penalty_counterexample':{'valid_action_logprob':-20.,'invalid_utility':-4.,
            'valid_gate_probability':.5,'negative_utility_gradient':float(fixed_g),
            'marginal_NLL_gradient':float(marginal_g),
            'implication':'Gradient descent on fixed-penalty NLL lowers validity; marginal NLL raises it.'},
        'marginal_K2':{'all_invalid_ordered_batches':zero_batches,
            'all_invalid_probability':zero_batch_mass,'zero_updates_used_only_for_this_expectation':True,
            'finite_K_gradient_max_bias':float((rws_expected-truth_marginal).abs().max())},
        'selected_new_training_objective':None,
        'decision':'Keep these as explicit ordinary controls. Do not automatically train the 14 unannotated units with a fixed malformed-output penalty.'})

    # Source default penalty depends on the entire reward batch. Exact example
    # tests that variant's own expected batch utility, not the published model.
    q=torch.tensor(-.4,dtype=torch.float64,requires_grad=True)
    def batch_terms():
        lp=torch.stack([torch.nn.functional.logsigmoid(q),torch.nn.functional.logsigmoid(-q)])
        value=q*0;surrogate_expectation=0.
        for pair in itertools.product(range(2),repeat=2):
            ix=list(pair);mass=lp[ix].sum().exp();raw=torch.tensor([-1.,-3.],dtype=q.dtype)[ix]
            reward=torch.where(torch.tensor(ix)==0,raw,2*raw.mean())
            value-=mass*reward.mean()
            adv=reward-reward.flip(0)
            s=-(adv*lp[ix]).mean()
            surrogate_expectation+=float(mass.detach())*torch.autograd.grad(s,q,retain_graph=True)[0]
        return value,surrogate_expectation
    v,estimated=batch_terms();truth=torch.autograd.grad(v,q)[0]
    assert abs(float(truth-estimated))>.01
    write(a.out/'batch_dependent_penalty.json',{'valid_probability':float(q.sigmoid()),
        'expected_batch_utility_negative_gradient':float(truth),
        'own_score_RLOO_expected_gradient':float(estimated),
        'difference':float(estimated-truth),
        'source_rule':'invalid reward = 2 * mean of the same batch raw rewards',
        'scope':'Two outcomes and K=2, fixed raw scores; not evidence of upstream empirical failure.'})

    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.processor,local_files_only=True,trust_remote_code=False)
    eos=[tok.eos_token_id];closing=tok.convert_tokens_to_ids('</think>')
    streams={
        'closed':tok.encode('<think>inspect current support</think>',add_special_tokens=False),
        'early_eos':eos,
        'length_limit':tok.encode('<think>inspect current support',add_special_tokens=False),
        'malformed_closed':tok.encode('<think><tool_call>{}</tool_call></think>',add_special_tokens=False),
        'role_closed':tok.encode('<think><|im_start|>user</think>',add_special_tokens=False)}
    boundaries=[]
    for name,ids in streams.items():
        r=reason_outcome(tok,ids,eos_ids=eos,token_cap=len(ids));r['fixture']=name
        assert r['action_prefix_eligible']==(name=='closed')
        assert r['bounded_distribution_score_eligible'] and r['tokens_in_cost']==len(ids)
        boundaries.append(r)
    interrupted=reason_outcome(tok,streams['length_limit'],eos_ids=eos,token_cap=100,interrupted=True)
    assert not interrupted['bounded_distribution_score_eligible'];boundaries.append(interrupted)
    write(a.out/'token_boundary_fixtures.json',{'synthetic_text_actual_tokenizer':True,'rows':boundaries})
    # Causal logit indexing uses a small declared vocabulary; no LM was called.
    x=torch.tensor([[.2,.3,-.5],[.8,-.2,.1],[-.4,.2,.6],[.4,.2,-.9]],dtype=torch.float64,requires_grad=True)
    r={'output_ids':[0,1],'emitted_tokens':2,'bounded_distribution_score_eligible':True}
    actual=sequence_logprob(x,[2,2,0,1],prompt_length=2,outcome=r)
    oracle=x[1].log_softmax(0)[0]+x[2].log_softmax(0)[1]
    assert torch.equal(actual,oracle)
    rejected=[]
    for name,call in [
        ('post_stop_tokens',lambda:reason_outcome(tok,streams['closed']+eos,eos_ids=eos,token_cap=100)),
        ('unknown_short_stop',lambda:reason_outcome(tok,streams['length_limit'],eos_ids=eos,token_cap=100)),
        ('fake_EOS_in_scoring',lambda:sequence_logprob(x,[2,2,0,1,2],prompt_length=2,outcome=r)),
        ('external_stop_scoring',lambda:sequence_logprob(x,[2,2,0,1],prompt_length=2,outcome={**r,'bounded_distribution_score_eligible':False}))]:
        try:call()
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)

    archive=Path('runs/autodl/20261110_decision_content/free_output_review/review.json')
    old=json.loads(archive.read_text());archive_rows=[]
    for item in old['records']:
        p=Path(item['local_path']);assert sha(p)==item['sha256'];text=p.read_text()
        closed=text.count('</think>')==1
        prefix=text.split('</think>')[0]+'</think>' if closed else None
        archive_rows.append({'unit_id':item['unit_id'],'arm':item['arm'],'source_path':item['source_path'],
            'sha256':sha(p),'text_has_single_closing_delimiter':closed,
            'prefix_has_embedded_tool':bool(prefix and '<tool_call>' in prefix),
            'probability_score_available':False,
            'boundary_limitation':'Decoded legacy text only; no original emitted IDs, EOS receipt or logits used.'})
    assert len(archive_rows)==12
    write(a.out/'archived_output_audit.json',archive_rows)
    write(a.out/'summary.json',{'previous_goal_turn':'no_progress_for_algorithm: Git status review restated existing evidence',
        'this_goal_turn':'progress: executable stopped-token scoring, failure-preserving loss and new exact counterexamples',
        'source_commit':manifest['commit'],'finite_outcomes':22,'all_K2_ordered_samples':484,
        'archived_outputs_retained':12,'archived_text_without_closing_reason':sum(not r['text_has_single_closing_delimiter'] for r in archive_rows),
        'real_output_probability_scores':0,'causal_score_check':True,'rejected':rejected,
        'fixed_failure_RLOO_max_error':finite['RLOO_expected_gradient_max_error'],
        'no_new_method_claim':True,'real_model_calls':0,'optimizer_updates':0,'new_scenes':0,
        'cuda_initialized':torch.cuda.is_initialized(),'wall_seconds':time.monotonic()-tick,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'runtime_integrated':False,'training_ready':False,'gpu_requested':False})
    print(json.dumps(json.loads((a.out/'summary.json').read_text()),ensure_ascii=False))


if __name__=='__main__':main()
