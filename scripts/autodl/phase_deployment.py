"""Deployment prefix for the v3 decision/action task contract.

Only actual current observations and an already generated decision are accepted.
This module does not load a policy, choose an action or certify scene completion.
"""
from phase_objective import phase_messages
from routed_multimodal import encode_prefix, frame_completed_decision


def deployment_input(processor, raw_messages, images, *, arm, stage,
                     generated_decision=None, max_length=16384):
    if arm == 'joint_same_material_v1':
        if stage != 'joint' or generated_decision is not None:
            raise ValueError('Joint inference takes raw observations without a decision prefix')
        batch = encode_prefix(processor, raw_messages, images)
        if batch['input_ids'].shape[1] > max_length:
            raise ValueError('Deployment prefix exceeds the frozen context limit')
        return batch
    if arm not in ('separate_phase_v3', 'paired_phase_v3') or stage not in ('decision', 'action'):
        raise ValueError('Unknown phase or arm')
    if stage == 'decision' and generated_decision is not None:
        raise ValueError('A decision phase must not be given a target decision')
    if arm == 'separate_phase_v3' and generated_decision is not None:
        raise ValueError('The separate action must not read a decision prefix')
    prefix = ''
    if stage == 'action' and arm == 'paired_phase_v3':
        if generated_decision is None:
            raise ValueError('Paired deployment requires its actual generated decision')
        prefix = frame_completed_decision(generated_decision)
    batch = encode_prefix(processor, phase_messages(raw_messages, stage), images, prefix)
    if batch['input_ids'].shape[1] > max_length:
        raise ValueError('Deployment prefix exceeds the frozen context limit')
    return batch


def decision_stop_ids(tokenizer):
    """The host must stop D at its closing tag, before any action is sampled.

    Native EOS can also stop generation, but it is not a valid completed D.
    frame_completed_decision will reject an early/malformed completion.
    """
    token_id = tokenizer.convert_tokens_to_ids('</think>')
    if tokenizer.encode('</think>', add_special_tokens=False) != [token_id]:
        raise ValueError('Closing decision delimiter is not one token')
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    if any(not isinstance(value, int) or value < 0 for value in eos_ids):
        raise ValueError('Missing native EOS token')
    return list(dict.fromkeys([token_id] + eos_ids))
