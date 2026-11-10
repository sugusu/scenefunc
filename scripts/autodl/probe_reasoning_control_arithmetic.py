"""Unmodified author RED arithmetic + independent control boundaries on CPU.

No VLM calls, generated scenes, model weights, optimizer or synthetic task
success. Fixed logits/features are explicitly mathematical examples only.
Original proprietary evaluation source is never copied into this script.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from reasoning_control_objectives import red_log_distribution, frozen_target_cosine_loss


def main(stage, out_name):
    out=stage/out_name;out.mkdir(exist_ok=False)
    manifest=json.loads((stage/'RED_source_manifest.json').read_text())
    source=Path(manifest['local_read_only_source'])
    for row in manifest['files']:
        assert hashlib.sha256((source/row['path']).read_bytes()).hexdigest()==row['sha256']
    sys.path.insert(0,str(source))
    start=time.monotonic()
    result=dict(status='started',VLM_forwards=0,model_weights_loaded=False,
                optimizer_updates=0,scene_edits=0,paid_API_calls=0)
    try:
        p=source/'src/decoding/multipath.py'
        spec=importlib.util.spec_from_file_location('internal_red_arithmetic',p)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        vectors=[([2.,-1.,.1],[-3.,1.,.7]),([.1,.11,-.4],[3.,-2.,-.7])]
        rows=[]
        for index,(visual,reason) in enumerate(vectors):
            v=torch.tensor(visual,dtype=torch.float64);r=torch.tensor(reason,dtype=torch.float64)
            for weight in [0.,.1,1.,3.]:
                # Execute the original function body without AST extraction,
                # wrapping, editing, installation or invoking its model sampler.
                compiled=original._compile_logits(torch.stack((v,r)),
                    torch.tensor([1.,weight],dtype=torch.float64),'poe')[0].log_softmax(-1)
                reference=red_log_distribution(v,r,weight=weight)
                error=float((compiled-reference).abs().max())
                assert error < 2e-14
                shifted=red_log_distribution(v+17.,r-13.,weight=weight)
                assert float((reference-shifted).abs().max()) < 2e-14
                if weight==0:assert torch.allclose(reference,v.log_softmax(-1),atol=0,rtol=0)
                rows.append(dict(vector=index,weight=weight,error=error,
                                 conditional_probabilities=reference.exp().tolist()))
        # Objective closed form, not a claim about real world task reward.
        v=torch.tensor([.4,-.2,.1],dtype=torch.float64)
        r=torch.tensor([-.3,.2,1.],dtype=torch.float64)
        weight=.7;optimal=red_log_distribution(v,r,weight=weight);beta=1/weight
        log_v=v.log_softmax(-1);log_r=r.log_softmax(-1)
        def utility(log_q):
            q=log_q.exp();return (q*log_r).sum()-beta*(q*(log_q-log_v)).sum()
        competitors=[]
        for candidate in [[.2,.3,.5],[.7,.1,.2],[.1,.8,.1]]:
            log_q=torch.tensor(candidate,dtype=torch.float64).log()
            gap=float(utility(optimal)-utility(log_q))
            kl=float((log_q.exp()*(log_q-optimal)).sum())
            assert abs(gap-beta*kl)<1e-13
            competitors.append(dict(gap=gap,beta_times_KL=beta*kl))
        # Actual first-order derivatives for a declared scalar cosine example;
        # no neural backbone or optimizer. Masked row cannot acquire a gradient.
        features=torch.tensor([[1.,1.],[0.,2.],[1.,-1.]],dtype=torch.float64,requires_grad=True)
        target=torch.tensor([[1.,0.],[0.,1.],[0.,1.]],dtype=torch.float64)
        mask=torch.tensor([True,False,True])
        loss=frozen_target_cosine_loss(features,target,included=mask)
        gradient=torch.autograd.grad(loss,features)[0]
        assert torch.equal(gradient[1],torch.zeros_like(gradient[1]))
        assert frozen_target_cosine_loss(features,target,included=torch.zeros_like(mask)) is None
        rejected=[]
        for label,t in [('learnable_teacher',target.clone().requires_grad_()),
                        ('zero_included_target',torch.zeros_like(target))]:
            try:frozen_target_cosine_loss(features,t,included=mask)
            except ValueError:rejected.append(label)
        assert len(rejected)==2
        result.update(status='original_RED_arithmetic_and_independent_objectives_checked',
            original_source_revision=manifest['revision'],original_function='_compile_logits',
            original_body_unmodified=True,probability_cases=rows,KL_identity=competitors,
            independent_LSS_formula=dict(loss=float(loss.detach()),gradient=gradient.tolist(),
                excluded_row_zero_gradient=True,all_masked_abstains=True,rejected=rejected,
                source_implementation_executed=False,text_encoder_or_phase_annotations_acquired=False),
            interpretation='Finite raw-logit weighted sum and weighted log-softmax have identical final softmax up to per-context constants. This helper does not explain or refute the author-reported batched model discrepancy at lambda0.',
            no_scene_or_learned_effect_claim=True)
    except BaseException as exc:
        import traceback
        result.update(status='failed_no_automatic_retry',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    result.update(seconds=time.monotonic()-start,cuda_initialized=torch.cuda.is_initialized())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out-name',default='arithmetic_v1')
    a=p.parse_args();main(a.stage.resolve(),a.out_name)
