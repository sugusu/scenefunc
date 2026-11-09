"""Failure-preserving probability boundaries for the optional reason control.

No model loader, sampler, training entrypoint or geometry modification. The
existing phase policy remains unchanged. These are ordinary stopped-sequence
likelihood and score-function estimators, not a proposed new 3D method.
"""
import math

import torch

from phase_policy import classify_stage


def reason_outcome(tokenizer, output_ids, *, eos_ids, token_cap, interrupted=False):
    """Describe actual emitted IDs; never add EOS, repair text or resample.

    A receipt with unknown/external termination remains an attempted draw but
    cannot enter the fixed-token-budget IID estimator below. Decoded text alone
    is insufficient evidence of the original token IDs or stopping boundary.
    """
    ids = list(output_ids)
    close = tokenizer.convert_tokens_to_ids('</think>')
    if tokenizer.encode('</think>', add_special_tokens=False) != [close]:
        raise ValueError('This contract requires a single-token closing delimiter')
    if (type(token_cap) is not int or token_cap < 1 or len(ids) > token_cap or
            any(type(t) is not int or t < 0 for t in ids) or
            not eos_ids or any(type(t) is not int or t < 0 for t in eos_ids) or close in eos_ids):
        raise ValueError('Invalid token receipt or stopping configuration')
    stops = set(eos_ids) | {close}
    if any(t in stops for t in ids[:-1]):
        raise ValueError('Receipt contains tokens after the first stopping event')
    if interrupted:
        cause = 'external_interruption'
    elif ids and ids[-1] == close:
        cause = 'closing_delimiter'
    elif ids and ids[-1] in eos_ids:
        cause = 'native_eos'
    elif len(ids) == token_cap:
        cause = 'token_cap'
    else:
        raise ValueError('Short output without a known stop: preserve as external failure')
    parsed = classify_stage(tokenizer, ids, stage='decision', eos_ids=eos_ids)
    eligible = not interrupted
    return {'output_ids': ids, 'emitted_tokens': len(ids), 'token_cap': token_cap,
            'stop_cause': cause, 'parse_kind': parsed['kind'],
            'parse_reason': parsed.get('reason'),
            'action_prefix_eligible': eligible and parsed['kind'] == 'decision',
            'bounded_distribution_score_eligible': eligible,
            'attempts_in_denominator': 1, 'tokens_in_cost': len(ids),
            'semantics_verified': False, 'original_tokens_required': True}


def sequence_logprob(logits, full_ids, *, prompt_length, outcome):
    """Causal score of ALL emitted tokens, including an actual stopping token.

    logits[t] predicts full_ids[t+1]. Input is one unpadded sequence; a length
    cap has no fictitious terminal event. Caller supplies logits under the
    original multimodal prefix. This routine cannot verify that provenance or
    that the sampler was IID/on-policy; a deployment receipt must establish it.
    """
    ids = list(full_ids)
    if (not outcome['bounded_distribution_score_eligible'] or
            type(prompt_length) is not int or prompt_length < 1 or
            ids[prompt_length:] != outcome['output_ids'] or
            len(ids) != prompt_length + outcome['emitted_tokens'] or
            outcome['emitted_tokens'] == 0):
        raise ValueError('Scoring needs an intact bounded sample and exact prefix boundary')
    if logits.ndim != 2 or logits.shape[0] != len(ids):
        raise ValueError('Expected unpadded full-sequence causal logits')
    if any(type(t) is not int or not 0 <= t < logits.shape[-1] for t in ids):
        raise ValueError('Token outside logit vocabulary')
    part = logits[prompt_length-1:len(ids)-1].log_softmax(-1)
    target = torch.tensor(outcome['output_ids'], device=logits.device)
    return part.gather(1, target[:, None]).sum()


def _terms(log_prior, log_action, valid, log_reference, beta, invalid_reward):
    from missing_decision_objective import _scores
    _scores(log_prior, log_action, log_reference)
    if valid.shape != log_prior.shape or valid.dtype != torch.bool or valid.device != log_prior.device:
        raise ValueError('Validity must match all sampled outcomes')
    if (not math.isfinite(invalid_reward) or invalid_reward > 0 or
            not math.isfinite(beta) or beta < 0 or log_reference.requires_grad):
        raise ValueError('Explicit fixed failure utility and frozen reference required')
    if (log_action[~valid] != 0).any():
        raise ValueError('Invalid reasons have no gold-action score; use zero placeholders')
    action_term = torch.where(valid, log_action, torch.zeros_like(log_action))
    utility = torch.where(valid, log_action, torch.full_like(log_action, invalid_reward))
    reward = utility - beta * (log_prior - log_reference)
    return action_term, reward


def bounded_expected_log_nll(log_prior, log_action, valid, log_reference, *, beta, invalid_reward):
    """Exact finite-support control; fixed failure utility is not scene reward.

    If a valid action's sequence log probability is below invalid_reward,
    this objective can FAVOR invalid output. No finite constant prevents that
    for arbitrary action likelihoods. Numerical correctness is not admission
    of this control as a useful training objective.
    """
    from missing_decision_objective import _complete_prior
    _, reward = _terms(log_prior, log_action, valid, log_reference, beta, invalid_reward)
    _complete_prior(log_prior); _complete_prior(log_reference)
    return -(log_prior.exp() * reward).sum()


def bounded_rloo_surrogate(log_prior, log_action, valid, log_reference, *, beta, invalid_reward):
    """IID on-policy K>=2, including invalid draws. No clipping or filtering.

    Returns the expected-utility gradient surrogate, not marginal NLL. The
    baseline is independent of its own draw only for per-outcome utilities.
    A failure cost recomputed from the same minibatch is not such a utility.
    """
    if len(log_prior) < 2:
        raise ValueError('At least two independent draws are required')
    action, reward = _terms(log_prior, log_action, valid, log_reference, beta, invalid_reward)
    reward = reward.detach()
    baseline = (reward.sum() - reward) / (len(reward) - 1)
    return -((reward - baseline) * log_prior + action).mean()


def bounded_marginal_nll(log_prior, log_action, valid):
    """Exact -log P(valid reason AND reference action), for finite support.

    Invalid outcomes retain their sampling mass but have zero success
    likelihood. This is neither conditional-on-valid likelihood nor actual
    scene utility. An impossible reference cannot yield a finite NLL.
    """
    from missing_decision_objective import _complete_prior
    _terms(log_prior, log_action, valid, log_prior.detach(), 0., -1.)
    _complete_prior(log_prior)
    if not valid.any():
        raise ValueError('Zero success mass; no finite marginal NLL')
    return -torch.logsumexp((log_prior+log_action)[valid],0)


def bounded_rws_surrogate(log_prior, log_action, valid, log_proposal):
    """Zero weights for invalid draws, with explicit all-invalid abstention.

    All draws count. A zero-valid sample set has no self-normalized estimator:
    return None, log the failure, and never resample until success. A caller
    assigning zero update to such sets still has a finite-K biased estimator.
    No fallback loss or change to training has been selected here.
    """
    _terms(log_prior, log_action, valid, log_proposal, 0., -1.)
    count=len(log_prior)
    if not valid.any():
        return None, {'status':'zero_valid_sample_mass','samples':count,
            'valid_samples':0,'resampling_authorized':False,'gradient_estimator_available':False}
    joint=(log_prior+log_action)[valid]
    weights=(joint.detach()-log_proposal[valid]).softmax(0)
    return -(weights*joint).sum(), {'status':'estimated','samples':count,
        'valid_samples':int(valid.sum()),'resampling_authorized':False,
        'gradient_estimator_available':True,'finite_K_unbiased':False,
        'effective_sample_size':float(1/weights.square().sum())}
