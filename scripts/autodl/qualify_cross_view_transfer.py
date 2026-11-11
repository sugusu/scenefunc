"""Exact finite-dimensional witnesses for limits of prompt-view transfer claims.

These are authored logistic probability functions, not neural model objects,
actual model logits, generated scenes, training experiments, or new theorems.
No optimizer is constructed. Arithmetic evaluates a proposed small step only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    assert "20261111_verified_repair_supervision" in out.parts or out.is_relative_to(Path("/dev/shm"))
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    import torch
    torch.set_num_threads(1)
    dtype = torch.float64
    theta = torch.zeros(1, dtype=dtype, requires_grad=True)
    log0 = torch.nn.functional.logsigmoid(theta).sum()
    logs = torch.nn.functional.logsigmoid(-theta).sum()
    g0, = torch.autograd.grad(log0, theta)
    gs, = torch.autograd.grad(logs, theta)
    eta = 0.01
    stepped = theta.detach() + eta * gs
    actual = float(torch.nn.functional.logsigmoid(stepped).sum() - log0.detach())
    prediction = eta * float(g0 @ gs)
    assert float(log0.detach() - logs.detach()) == 0
    assert float(g0 @ gs) < 0 and actual < 0
    one = dict(same_action_probability_both_contexts=0.5, log_probability_gap=0.0,
        unhinted_score=g0.tolist(), hinted_score=gs.tolist(), score_dot=float(g0 @ gs),
        step_size=eta, positive_advantage=1.0, exact_unhinted_log_likelihood_change=actual,
        first_order_prediction=prediction, remainder=actual-prediction)

    # All individual turn scores align, but cross-turn terms can reverse the
    # alignment of trajectory sums. Features are 2*g because sigmoid(0)=1/2.
    f0 = torch.tensor([[2., 0.], [-2., 4.]], dtype=dtype)
    fs = torch.tensor([[2., -4.], [-2., 0.]], dtype=dtype)
    theta2 = torch.zeros(2, dtype=dtype, requires_grad=True)
    l0 = torch.nn.functional.logsigmoid(f0 @ theta2)
    ls = torch.nn.functional.logsigmoid(fs @ theta2)
    per0 = torch.stack([torch.autograd.grad(v, theta2, retain_graph=True)[0] for v in l0])
    pers = torch.stack([torch.autograd.grad(v, theta2, retain_graph=True)[0] for v in ls])
    mat = per0 @ pers.T
    G0, Gs = per0.sum(0), pers.sum(0)
    assert torch.all(torch.diag(mat) > 0) and float(G0 @ Gs) < 0
    step2 = theta2.detach() + eta * Gs
    actual2 = float(torch.nn.functional.logsigmoid(f0 @ step2).sum() - l0.detach().sum())
    two = dict(all_action_probabilities=0.5, all_turn_log_probability_gaps=[0.0, 0.0],
        unhinted_turn_scores=per0.tolist(), hinted_turn_scores=pers.tolist(),
        all_cross_turn_dot_products=mat.tolist(), within_turn_dot_products=torch.diag(mat).tolist(),
        total_unhinted_score=G0.tolist(), total_hinted_score=Gs.tolist(),
        whole_trajectory_dot=float(G0 @ Gs),
        exact_unhinted_log_likelihood_change=actual2,
        first_order_prediction=eta*float(G0 @ Gs))

    # An optimizer/preconditioner changes the relevant direction. This is a
    # positive diagonal matrix witness, not an assertion about measured Adam.
    q0 = torch.tensor([1., 1.], dtype=dtype)
    qs = torch.tensor([2., -1.], dtype=dtype)
    pre = torch.diag(torch.tensor([0.1, 1.], dtype=dtype))
    preconditioned = dict(unhinted_score=q0.tolist(), hinted_score=qs.tolist(),
        raw_dot=float(q0 @ qs), positive_diagonal_preconditioner=pre.tolist(),
        direction_dot=float(q0 @ pre @ qs), actual_Adam_or_training_result=False)
    assert preconditioned["raw_dot"] > 0 > preconditioned["direction_dot"]
    r = dict(kind="Authored mathematical witnesses, not 3D or language-model outcomes",
        equal_probability_does_not_imply_aligned_scores=one,
        positive_within_turn_alignment_does_not_imply_trajectory_alignment=two,
        raw_alignment_not_preconditioned_alignment=preconditioned,
        interpretation="PSP's first-order Taylor identity remains valid; empirical alignment is not a universal consequence of small likelihood gaps, and batch/optimizer directions must be included",
        scientific_novelty_claim=False, model_instances=0, optimizer_updates=0,
        pretrained_forwards=0, scene_executions=0, paid_API_calls=0,
        torch_version=torch.__version__, seconds=time.monotonic()-start)
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(r, indent=2)+"\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
