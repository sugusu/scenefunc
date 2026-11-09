"""Prospective two-stage material contract and ordinary weighted causal CE.

This is a control implementation, not a new learning algorithm. Per-token
weights refer to the ORIGINAL corpus family/component totals. Repartitioning
terminal prose must not silently change those weights.
"""
import copy


PHASE_MODE = {
    'decision': (
        '\n\nCURRENT OUTPUT TASK: DECISION_RECORD. Produce one proposed decision '
        'record inside <think>...</think>. Include the measured scope and remaining '
        'unknowns when proposing to stop. Do not issue a tool call or claim an '
        'unexecuted edit succeeded. The host ends this stage at </think>.'),
    'action': (
        '\n\nCURRENT OUTPUT TASK: SCENE_ACTION. Emit one permitted tool call, '
        'or end this assistant turn without a tool call to propose STOP. '
        'Do not generate another decision record in this stage. The host checks '
        'actual requirements before accepting completion; termination alone is '
        'not scene success. Any prefilled decision record is an unverified '
        'proposal, not an execution receipt. Use the original task, actual '
        'scene and tool results to decide the action.'),
}


def phase_messages(messages, stage):
    if stage not in PHASE_MODE or not messages or messages[0]['role'] != 'system':
        raise ValueError('Unknown phase or missing system message')
    out = copy.deepcopy(messages)
    if not isinstance(out[0]['content'], str):
        raise ValueError('Expected textual system message')
    out[0]['content'] += PHASE_MODE[stage]
    return out


def weighted_causal_nll(logits, labels, token_weights):
    """Sum CE at position t-1 for target t, with ORIGINAL token coefficients.

    Caller accumulates these sums across the fixed corpus before one update.
    There is intentionally no division by batch size, stage length or new
    stage totals. Trainer/accelerator accumulation scaling must be audited
    before integration. This function alone is not a validated training run.
    """
    import torch
    import torch.nn.functional as F
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or token_weights.shape != labels.shape:
        raise ValueError('Expected [batch, sequence, vocabulary] and aligned targets')
    if labels.dtype != torch.long or not token_weights.is_floating_point():
        raise ValueError('Invalid label/weight dtypes')
    if not torch.isfinite(token_weights).all() or (token_weights < 0).any():
        raise ValueError('Invalid token coefficient')
    supervised = labels != -100
    if supervised[:, 0].any() or (token_weights[~supervised] != 0).any():
        raise ValueError('Prompt, padding or unconditioned first token is supervised')
    if (token_weights[supervised] <= 0).any():
        raise ValueError('Target without a positive coefficient')
    targets = labels[:, 1:]
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                         targets.reshape(-1), ignore_index=-100,
                         reduction='none').reshape_as(targets)
    return (ce * token_weights[:, 1:]).sum()


def attach_objective_weights(batch, component):
    import torch
    labels = batch['labels']
    if labels.shape[0] != 1:
        raise ValueError('Attach before collation, one source component at a time')
    observed = labels[labels != -100].tolist()
    if observed != component['target_token_ids']:
        raise ValueError('Encoded target differs from immutable source')
    values = component['token_objective_weights']
    if len(values) != len(observed):
        raise ValueError('Weight/target length mismatch')
    weights = torch.zeros_like(labels, dtype=torch.float64)
    weights[labels != -100] = torch.tensor(values, dtype=torch.float64)
    batch['token_objective_weights'] = weights
    return batch
