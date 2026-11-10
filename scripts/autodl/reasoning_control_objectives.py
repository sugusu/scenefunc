"""Arithmetic controls from RED and LSS publications; no model/training entry.

RED: Yamaguchi et al., CVPR2026, equations8/12, arXiv:2507.07685v2.
LSS: Li et al., arXiv:2609.04893v1, equations1/2.
These independently written kernels are existing-method controls, not novel
losses. Hidden states and frozen text targets must be produced elsewhere under
an explicit data/encoder contract; no author model or dataset is reproduced.
"""
import math
import torch
import torch.nn.functional as F


def red_log_distribution(image_logits, rationale_logits, *, weight):
    """Normalized product of two supplied next-token distributions.

    This does not perform either model call or establish same-prefix/cache
    correctness. Finite equal-vocabulary score vectors only. It is not a
    geometry-validity filter and can amplify an incorrect rationale.
    """
    if (not math.isfinite(weight) or weight < 0 or image_logits.ndim != 1
            or rationale_logits.shape != image_logits.shape
            or rationale_logits.device != image_logits.device
            or not image_logits.is_floating_point()
            or not rationale_logits.is_floating_point()
            or not torch.isfinite(image_logits).all()
            or not torch.isfinite(rationale_logits).all()):
        raise ValueError('Finite aligned next-token scores and nonnegative weight required')
    dtype = torch.float64 if image_logits.dtype == rationale_logits.dtype == torch.float64 else torch.float32
    visual = image_logits.to(dtype).log_softmax(-1)
    # Avoid gratuitous second-context arithmetic when lambda==0. The caller's
    # actual inference accounting still records any already-computed context.
    if weight == 0:
        return visual
    rationale = rationale_logits.to(dtype).log_softmax(-1)
    return (visual + weight*rationale).log_softmax(-1)


def frozen_target_cosine_loss(projected, targets, *, included):
    """Equation2 arithmetic with explicit eligibility; all-masked abstains.

    `projected` is g_phi(z), not a generated rationale. The caller owns the
    head, causal feature positions, phase matching, source masks and lambda.
    Frozen targets may not receive gradients. Masked/missing phases cannot be
    filled with scoring answers. Token mean is the cited prior's objective;
    it is separate from this repository's immutable action-CE coefficients.
    """
    if (projected.ndim != 2 or targets.shape != projected.shape
            or included.shape != projected.shape[:1] or included.dtype != torch.bool
            or projected.device != targets.device or included.device != projected.device
            or targets.requires_grad or not projected.is_floating_point()
            or not targets.is_floating_point()):
        raise ValueError('Aligned projected features, frozen targets and explicit mask required')
    if not included.any():
        return None
    p = projected[included]; t = targets[included]
    if (not torch.isfinite(p).all() or not torch.isfinite(t).all()
            or (p.norm(dim=-1) == 0).any() or (t.norm(dim=-1) == 0).any()):
        raise ValueError('Included cosine targets must be finite and nonzero')
    return (1-F.cosine_similarity(p,t,dim=-1)).mean()
