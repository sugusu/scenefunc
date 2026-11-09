"""Ordinary latent-variable controls for partially annotated scene actions.

Numerical kernels only: no policy, sampler, optimizer, native tools, or CLI.
RWS is Bornschein & Bengio (2015); the expected-log objective / RLOO
estimator is the core of LaTRO (Chen et al., 2024). Neither is a new 3D
algorithm. Scores are SUMS of log token probabilities, including the declared
terminal token. Token averages and a list of greedy proposals are not valid
substitutes for sequence probabilities and IID samples.
"""
import math

import torch


def _scores(*xs):
    if not xs or any(x.ndim != 1 or not x.numel() or not x.is_floating_point()
                     for x in xs):
        raise ValueError('Expected nonempty floating-point score vectors')
    if any(x.shape != xs[0].shape or x.device != xs[0].device for x in xs):
        raise ValueError('Scores must align on the same sampled reasons')
    if any(not torch.isfinite(x).all() or (x > 1e-7).any() for x in xs):
        raise ValueError('Expected finite log probabilities, not rewards or logits')


def _weight(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Expected positive immutable action coefficient')


def _complete_prior(log_prior):
    # This checks numerical normalization, NOT that a caller enumerated every
    # possible LLM string. Exact enumeration is only practical for finite tests.
    if abs(float(torch.logsumexp(log_prior.detach(), 0))) > 1e-6:
        raise ValueError('Exact finite enumeration requires a normalized prior')


def exact_marginal_nll(log_prior, log_action, coefficient=1.):
    """-c log sum_D p(D|X) p(A*|X,D), for a complete finite support."""
    _scores(log_prior, log_action); _complete_prior(log_prior); _weight(coefficient)
    return -coefficient * torch.logsumexp(log_prior + log_action, 0)


def rws_surrogate(log_prior, log_action, log_proposal, coefficient=1.):
    """Self-normalized importance estimate of the marginal-NLL gradient.

    Caller must supply IID draws from a KNOWN, full-support proposal. Its
    probability is a frozen sampling receipt, not a learnable denominator.
    At finite K this gradient is generally biased. The returned surrogate's
    scalar value is NOT the marginal NLL. Invalid/truncated draws cannot be
    silently discarded or replaced with new draws.
    """
    _scores(log_prior, log_action, log_proposal); _weight(coefficient)
    if log_proposal.requires_grad:
        raise ValueError('Sampling distribution must be a frozen receipt')
    joint = log_prior + log_action
    log_weights = joint.detach() - log_proposal
    posterior = log_weights.softmax(0)
    loss = -coefficient * (posterior * joint).sum()
    diagnostics = {
        'samples': len(joint),
        'normalized_weights': posterior.tolist(),
        'effective_sample_size': float(1 / posterior.square().sum()),
        'log_evidence_estimate': float(torch.logsumexp(log_weights, 0) - math.log(len(joint))),
        'finite_K_unbiased_marginal_gradient': False,
    }
    return loss, diagnostics


def exact_expected_log_nll(log_prior, log_action, log_reference, *, beta=0., coefficient=1.):
    """-c E_p[log p(A*|X,D) - beta log(p(D|X)/p0(D|X))]."""
    _scores(log_prior, log_action, log_reference)
    _complete_prior(log_prior); _complete_prior(log_reference); _weight(coefficient)
    if beta < 0 or not math.isfinite(beta) or log_reference.requires_grad:
        raise ValueError('Invalid beta or nonfrozen reference')
    return -coefficient * (log_prior.exp() *
        (log_action - beta * (log_prior - log_reference))).sum()


def latro_rloo_surrogate(log_prior, log_action, log_reference, *, beta=0., coefficient=1.):
    """Core on-policy LaTRO / RLOO gradient, without clipping or penalties.

    IID reasons MUST be sampled from the same p_theta whose log_prior is
    scored, at the same parameter version. K>=2. The detached leave-one-out
    baseline is independent of its own draw. The action-score derivative is
    retained because the evaluator and reasoner share parameters. A different
    sampling temperature, top-k, stale model, or filtering breaks this claim.
    """
    _scores(log_prior, log_action, log_reference); _weight(coefficient)
    if len(log_prior) < 2:
        raise ValueError('Leave-one-out needs at least two IID samples')
    if beta < 0 or not math.isfinite(beta) or log_reference.requires_grad:
        raise ValueError('Invalid beta or nonfrozen reference')
    reward = (log_action - beta * (log_prior - log_reference)).detach()
    baseline = (reward.sum() - reward) / (len(reward) - 1)
    advantage = reward - baseline
    loss = -coefficient * (advantage * log_prior + log_action).mean()
    return loss, {'samples': len(reward), 'advantages': advantage.tolist(),
                  'mean_sampled_reward': float(reward.mean()),
                  'objective': 'expected_log_action_minus_KL; not log_marginal_action'}


def original_action_coefficient(component):
    """Convert existing constant token coefficients to a sequence coefficient.

    c*sum_t(-log p(a_t)) equals the original token-weighted action loss.
    Nonuniform coefficients require a separately defined energy objective;
    do not silently move token weights inside a probability distribution.
    """
    if component['stage'] != 'action':
        raise ValueError('Expected an action component')
    ws = component['token_objective_weights']
    if len(ws) != len(component['target_token_ids']) or not ws or len(set(ws)) != 1:
        raise ValueError('Action must have one constant coefficient per original token')
    _weight(ws[0])
    return ws[0]


def route_partial_material(components):
    """Retain all paired-arm units; annotate a prospective ordinary-control route.

    No target reasons are created, no data are sampled, and no images replaced.
    This route does not authorize training or mark its material executable.
    """
    selected = [x for x in components if x['arm'] == 'paired_phase_v3']
    index = {}
    for c in selected:
        key = (c['unit_id'], c['stage'])
        if key in index:
            raise ValueError('Duplicate component')
        index[key] = c
    output = []
    for c in selected:
        if c['stage'] != 'action':
            continue
        d = index.get((c['unit_id'], 'decision'))
        if bool(c['assistant_prefix']) != bool(d):
            raise ValueError('Reason prefix and annotation disagree')
        output.append({'unit_id': c['unit_id'], 'family_id': c['family_id'],
            'source_request_sha256': c['source_request_sha256'],
            'action_sha256': c['target_sha256'],
            'action_tokens': len(c['target_token_ids']),
            'action_coefficient': original_action_coefficient(c),
            'original_action_mass': sum(c['token_objective_weights']),
            'reason_target_sha256': d['target_sha256'] if d else None,
            'reason_tokens': len(d['target_token_ids']) if d else 0,
            'original_reason_mass': sum(d['token_objective_weights']) if d else 0.,
            'route': 'observed_reason_supervision' if d else 'latent_reason_action_likelihood',
            'source_images': c.get('images'),
            'training_ready': False})
    return output
