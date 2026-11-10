"""Execute pinned Apache-2.0 SDPO function bodies on CPU distributions.

No model, optimization update, rollout or learning-performance experiment.
The exhaustive one-token expectation checks a mathematical estimator, not an
unbiased estimator over archived deterministic scene trajectories.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Optional


def main(source, out):
    import torch
    import torch.nn.functional as F
    tick = time.monotonic()
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    out.mkdir(parents=True, exist_ok=False)
    namespace = dict(torch=torch, F=F, Any=Any, Optional=Optional)
    hashes = []
    for rel, names in [('verl/utils/torch_functional.py', ['masked_sum']),
                       ('verl/trainer/ppo/core_algos.py', ['agg_loss', 'compute_self_distillation_loss'])]:
        path = source / rel
        code = path.read_text()
        nodes = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert [n.name for n in nodes] == names
        for node in nodes:
            hashes.append(dict(path=rel, function=node.name, first_line=node.lineno,
                               last_line=node.end_lineno,
                               function_sha256=hashlib.sha256(ast.get_source_segment(code, node).encode()).hexdigest()))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        if 'masked_sum' in names:
            namespace['verl_F'] = SimpleNamespace(masked_sum=namespace['masked_sum'])
    kernel = namespace['compute_self_distillation_loss']
    rows = []
    p0 = torch.tensor([.6, .3, .1], dtype=torch.float64)
    q0 = torch.tensor([.2, .5, .3], dtype=torch.float64)
    for alpha in [0., .5, 1.]:
        z = p0.log().clone().requires_grad_()
        lp = z.log_softmax(-1)[None, None, :]
        lq = q0.log()[None, None, :]
        cfg = SimpleNamespace(full_logit_distillation=True, distillation_topk=None,
                              alpha=alpha, is_clip=None)
        value, _ = kernel(lp[..., 0], lq[..., 0], torch.ones(1, 1), cfg,
                          student_all_log_probs=lp, teacher_all_log_probs=lq)
        if alpha == 0.:
            reference = (q0 * (q0.log() - p0.log())).sum()
        elif alpha == 1.:
            reference = (p0 * (p0.log() - q0.log())).sum()
        else:
            mix = (p0 + q0) / 2
            reference = .5 * ((p0 * (p0.log()-mix.log())).sum() +
                               (q0 * (q0.log()-mix.log())).sum())
        assert torch.allclose(value, reference, atol=1e-12, rtol=1e-12)
        grad = torch.autograd.grad(value, z)[0]
        if alpha == 1.:
            full_reverse_gradient = grad
        rows.append(dict(path='original_full_vocabulary', alpha=alpha,
                         scalar=float(value.detach()), independent_scalar=float(reference),
                         gradient=grad.tolist()))
    # The same selected token support is used for both distributions, as the
    # original actor explicitly passes student's topk_indices to teacher.
    for tail in [True, False]:
        lp = p0.log()[None, None, :2].clone().requires_grad_()
        lq = q0.log()[None, None, :2]
        cfg = SimpleNamespace(full_logit_distillation=True, distillation_topk=2,
                              distillation_add_tail=tail, alpha=1., is_clip=None)
        value, _ = kernel(lp[..., 0], lq[..., 0], torch.ones(1, 1), cfg,
                          student_topk_log_probs=lp, teacher_topk_log_probs=lq)
        if tail:
            ps, qs = p0, q0
        else:
            ps, qs = p0[:2]/p0[:2].sum(), q0[:2]/q0[:2].sum()
        expected = (ps*(ps.log()-qs.log())).sum()
        assert torch.allclose(value, expected, atol=1e-12, rtol=1e-12)
        rows.append(dict(path='original_topk', add_tail=tail, scalar=float(value.detach()),
                         independent_scalar=float(expected)))
    # At a fixed observed prefix and fixed teacher distribution, exhaustive
    # averaging over p equals the direct reverse-KL gradient.
    gradients = []
    cfg = SimpleNamespace(full_logit_distillation=False, alpha=1., is_clip=None)
    for index in range(3):
        z = p0.log().clone().requires_grad_()
        logp = z.log_softmax(-1)[index].reshape(1, 1)
        value, _ = kernel(logp, q0.log()[index].reshape(1, 1), torch.ones(1, 1), cfg)
        gradients.append(torch.autograd.grad(value, z)[0])
    expectation = sum(p0[i] * g for i, g in enumerate(gradients))
    assert torch.allclose(expectation, full_reverse_gradient, atol=1e-12, rtol=1e-12)
    greedy_gradient = gradients[int(p0.argmax())]
    assert not torch.allclose(greedy_gradient, expectation, atol=1e-8, rtol=1e-8)
    rows.append(dict(path='sampled_kernel_fixed_prefix', exact_expectation=expectation.tolist(),
                     full_reverse_KL_gradient=full_reverse_gradient.tolist(),
                     greedy_only_gradient=greedy_gradient.tolist(),
                     greedy_expectation_difference_L2=float((greedy_gradient-expectation).norm()),
                     note='Constructed three-token distribution; not a model/scene result.'))
    # Masking and teacher stop-gradient are caller responsibilities in the
    # actual actor. Exercise that contract, not a new optimizer.
    z = p0.log().clone().requires_grad_()
    lp = z.log_softmax(-1)[0].reshape(1, 1)
    value, _ = kernel(lp, q0.log()[0].reshape(1, 1), torch.ones(1, 1), cfg,
                      self_distillation_mask=torch.zeros(1))
    grad = torch.autograd.grad(value, z)[0]
    assert value.item() == 0. and torch.equal(grad, torch.zeros_like(grad))
    rows.append(dict(path='no_feedback_mask', loss=0., gradient_norm=float(grad.norm())))
    result = dict(status='original_SDPO_CPU_objective_and_sampling_contract_checked',
                  original_source_revision='7c457fc1b1f636ae794eb0362ba37d4743b06fbc',
                  original_source_functions=hashes, results=rows,
                  scope='Pinned original function bodies only; no verl/Ray/distributed/multimodal trainer integration',
                  known_limits=['Top-k support selected by student and shared with teacher',
                                'Teacher no_grad enforced in original dp_actor; kernel alone does not enforce it',
                                'Per-token fixed-prefix stop-gradient objective, not full sequential policy gradient',
                                'Greedy archive is not IID current-policy sampling; off-policy correction not provided',
                                'Dense loss/gradient alone proves neither teacher correctness nor scene improvement'],
                  model_weights_loaded=False, VLM_forwards=0, optimizer_updates=0, paid_API_calls=0,
                  CUDA_initialized=torch.cuda.is_initialized(), seconds=time.monotonic()-tick)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path); a=p.parse_args(); main(a.source, a.out)
