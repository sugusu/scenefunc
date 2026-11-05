"""Audit-only matched decision units around the pinned official SFT loader.

Both arms supervise the same reason once and the same action once. The paired
action sequence contains its teacher-forced reason; the separate one does not.
This is dataset preparation, not permission or an entry point for training.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def read_locked(root, ref):
    path = (Path(root) / ref['path']).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('Decision input hash mismatch: ' + str(path))
    return path, json.loads(data)


def mask_role(example, tokenizer, role):
    """Keep native tokenization; remove only the specified assistant prefix/tail."""
    out = {key: value.clone() for key, value in example.items()}
    positions = (out['labels'] != -100).nonzero().flatten()
    ids = out['labels'][positions].tolist()
    end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    tool_id = tokenizer.convert_tokens_to_ids('<tool_call>')
    if ids.count(end_id) != 1 or ids[-2:] != [end_id, tokenizer.encode('\n', add_special_tokens=False)[0]]:
        raise ValueError('Unexpected assistant ending; pinned tokenizer contract changed')
    if role == 'reason':
        if tool_id in ids:
            raise ValueError('Reason-only example contains a tool call')
        # Do not teach stopping immediately after the explanation.
        out['labels'][positions[ids.index(end_id):]] = -100
    elif role == 'action':
        if tool_id not in ids:
            raise ValueError('Action example has no actual tool-call delimiter')
        out['labels'][positions[:ids.index(tool_id)]] = -100
    else:
        raise ValueError('Unknown supervised role')
    if not bool((out['labels'] != -100).any()):
        raise ValueError('Empty supervised component')
    return out


def validate_triplet(reason, paired, separate):
    """The arm change may affect only assistant reason before the same action."""
    for doc in (reason, paired, separate):
        if doc.get('stage') != 'furniture' or len(doc.get('turns', [])) != 1:
            raise ValueError('Audit expects one observed furniture decision')
        if doc['turns'][0].get('role') != 'assistant':
            raise ValueError('Expected assistant target')
    pt, st, rt = (doc['turns'][0] for doc in (paired, separate, reason))
    if not pt.get('content', '').strip() or pt['content'] != rt.get('content') or st.get('content', '').strip():
        raise ValueError('Reason identity or separation mismatch')
    if not pt.get('tool_calls') or pt['tool_calls'] != st.get('tool_calls') or rt.get('tool_calls'):
        raise ValueError('Action identity or reason-only target mismatch')
    normalized = []
    for doc in (reason, paired, separate):
        d = deepcopy(doc)
        d['turns'][0]['content'] = ''
        d['turns'][0]['tool_calls'] = []
        normalized.append(d)
    if normalized[0] != normalized[1] or normalized[1] != normalized[2]:
        raise ValueError('Task, images, provenance or state differ across components')


class DecisionUnitDataset(Dataset):
    """One dataset item is a two-example decision, shared by both arms."""
    def __init__(self, manifest_path, repo_root, processor, upstream, arm, max_length=8192):
        if arm not in ('paired', 'separate'):
            raise ValueError('Unknown arm')
        self.root, self.processor, self.upstream = Path(repo_root), processor, upstream
        self.arm, self.max_length = arm, max_length
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get('schema') != 'scenefunc-decision-budget-audit-v1' or self.manifest.get('audit_only') is not True:
            raise ValueError('This implementation only admits explicitly audit-only manifests')
        self.decisions = self.manifest['decisions']
        if not self.decisions or len({d['id'] for d in self.decisions}) != len(self.decisions):
            raise ValueError('Empty or duplicate decision IDs')
        self.fixed_length = self.manifest['fixed_length']
        if not 0 < self.fixed_length <= max_length:
            raise ValueError('Invalid common padding length')

    def __len__(self):
        return len(self.decisions)

    def __getitem__(self, index):
        d = self.decisions[index]
        paths, documents = {}, {}
        for name in ('reason', 'paired', 'separate'):
            paths[name], documents[name] = read_locked(self.root, d[name])
        validate_triplet(documents['reason'], documents['paired'], documents['separate'])
        for ref in d['images']:
            read_locked_image = self.root / ref['path']
            if hashlib.sha256(read_locked_image.read_bytes()).hexdigest() != ref['sha256']:
                raise ValueError('Observation changed')
        declared = [str((self.root / ref['path']).resolve()) for ref in d['images']]
        actual = [str((self.root / path).resolve()) for path in documents['paired']['turns'][0]['images']]
        if actual != declared:
            raise ValueError('Observation lock/order mismatch')
        examples = []
        for name, role in [('reason', 'reason'), (self.arm, 'action')]:
            data = self.upstream.StepwiseSFTDataset(paths[name].parent, self.root,
                self.processor, max_length=self.max_length)
            if len(data) != 1:
                raise ValueError('An isolated component directory must have exactly one sample')
            example = mask_role(data[0], self.processor.tokenizer, role)
            if len(example['input_ids']) > self.fixed_length:
                raise ValueError('Example exceeds common padded budget')
            examples.append(example)
        return {'decision_id': d['id'], 'examples': examples}


class DecisionUnitCollator:
    def __init__(self, upstream, pad_token_id, fixed_length):
        self.collator = upstream.StepwiseCollator(pad_token_id)
        self.pad_token_id, self.fixed_length = pad_token_id, fixed_length

    def __call__(self, decisions):
        if not decisions or any(len(d['examples']) != 2 for d in decisions):
            raise ValueError('Every decision must contain two supervised components')
        batch = self.collator([ex for d in decisions for ex in d['examples']])
        length = batch['input_ids'].shape[1]
        if length > self.fixed_length:
            raise ValueError('Batch exceeds common sequence budget')
        for key, fill in [('input_ids', self.pad_token_id), ('attention_mask', 0),
                          ('labels', -100), ('mm_token_type_ids', 0)]:
            if key in batch:
                batch[key] = torch.nn.functional.pad(batch[key], (0, self.fixed_length-length), value=fill)
        return batch
