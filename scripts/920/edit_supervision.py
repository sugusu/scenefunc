"""Candidate future SFT adapter: no hidden, unsupervised assistant prefix.

Historical encoders and completed experiment tensors are deliberately unchanged.
This module does not authorize training or fix heterogeneous scene representations.
"""
import json
from pathlib import Path
from decision_pairing import mask_role


def encode_single_episode(episode, repo_root, processor, official, max_length=16384):
    episode=Path(episode)
    doc=json.loads(episode.read_text())
    if len(doc['turns'])!=1 or doc['turns'][0]['role']!='assistant':
        raise ValueError('Expected a single observed assistant action')
    has_decision=bool(doc['turns'][0].get('content','').strip())
    dataset=official.StepwiseSFTDataset(episode.parent,repo_root,processor,
        max_length=max_length,no_thinking=not has_decision)
    if len(dataset)!=1:
        raise ValueError('Expected exactly one official sample')
    example=dataset[0]
    action=mask_role(example,processor.tokenizer,'action')
    action_mask=action['labels']!=-100
    decision_mask=(example['labels']!=-100)&~action_mask
    if not has_decision and bool(decision_mask.any()):
        raise ValueError('Action-only serialization unexpectedly contains a decision target')
    # Every assistant token is a target, including the first freely generated
    # token. Inputs and actual receipts remain masked by the official loader.
    if not bool(((action_mask|decision_mask)==(example['labels']!=-100)).all()):
        raise ValueError('Unsupervised assistant gap')
    return example,action_mask,decision_mask,dataset.samples[0]
