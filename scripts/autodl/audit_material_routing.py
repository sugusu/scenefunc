"""Audit actual source targets and prepare a task-tagged material control.

Uses only pinned Git documents and the already installed tokenizer. No image
processing, model weights, gradients, API calls or training are performed.
The output is prospective text material, not a ready multimodal training run.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import time


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
CORPUS = 'runs/920/P920-MATCHEDINPUT-v1_20261108T055328Z/corpus/DATASET.json'
PLAN = 'runs/920/P920-MATCHEDINPUT-v1_20261108T055328Z/PLAN.json'
MODE = {
    'decision': ('\n\nCURRENT OUTPUT TASK: DECISION_RECORD. Produce only a <think> decision record '
                 'closed by </think>. Do not issue a tool call. Describe a proposed decision, '
                 'not an executed result. The host ends this stage at the closing tag.'),
    'action': ('\n\nCURRENT OUTPUT TASK: SCENE_ACTION. Issue one permitted tool call, or finish '
               'with the measured scope and remaining unknowns if no edit is needed. Any '
               'prefilled decision record is an unverified proposal, not an execution receipt. '
               'Use the actual scene and tool results to decide the action.')}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def token_hash(ids):
    return digest(struct.pack('<' + 'q' * len(ids), *ids))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class GitInputs:
    def __init__(self, repo):
        self.repo = repo
        self.cache = {}
        self.manifest = {}

    def blob(self, path):
        if path not in self.cache:
            value = subprocess.check_output(['git', 'show', f'{PIN}:{path}'], cwd=self.repo)
            self.cache[path] = value
            self.manifest[path] = digest(value)
        return self.cache[path]

    def json(self, path):
        return json.loads(self.blob(path))

    def reference(self, ref):
        path = ref['path'].split('/scenefunc/', 1)[1]
        data = self.blob(path)
        if digest(data) != ref['sha256']:
            raise ValueError('Published source/reference mismatch: ' + path)
        return json.loads(data), path


def tagged_request(request, component):
    """Same role tag for both experimental arms; no action answer in the tag."""
    out = copy.deepcopy(request)
    if not out or out[0]['role'] != 'system' or not isinstance(out[0]['content'], str):
        raise ValueError('Expected one leading textual system message')
    out[0]['content'] += MODE[component]
    return out


def action_prefix(arm, *, model_decision, evaluation):
    """Prospective inference boundary: paired must use an actual model draft.

    The caller supplies the draft at deployment, never a corpus lookup. This
    helper does not call a model or choose whether to edit. Empty drafts are
    allowed and must be logged; they do not trigger a free retry.
    """
    if arm == 'separate_tagged':
        return ''
    if arm != 'paired_tagged':
        raise ValueError('Unknown arm')
    if evaluation and model_decision is None:
        raise ValueError('Evaluation requires the recorded model draft, not a gold lookup')
    return model_decision or ''


def assistant_message(turn):
    # The furniture branch of the pinned official _build_messages serializer.
    content = turn.get('content', '').strip()
    message = {'role': 'assistant', 'content': f'<think>\n{content}\n</think>' if content else ''}
    if turn.get('tool_calls'):
        message['tool_calls'] = [{'type': 'function', 'function': {
            'name': call['name'], 'arguments': call.get('arguments') or {}}} for call in turn['tool_calls']]
    return message


def target_tokens(tokenizer, turn):
    dummy = [{'role': 'user', 'content': 'serialization-only'}]
    prefix = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(dummy + [assistant_message(turn)], tokenize=False,
                                         add_generation_prompt=False)
    if not full.startswith(prefix):
        raise ValueError('Target/prompt boundary differs')
    return tokenizer.encode(full[len(prefix):], add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    start = time.perf_counter()
    repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    src = GitInputs(repo)
    corpus, plan = src.json(CORPUS), src.json(PLAN)
    material_path = repo / 'runs/autodl/20261110_objective_source_review/material_manifest.json'
    material = json.loads(material_path.read_text())
    if digest(src.blob(CORPUS)) != material['source_corpus']['sha256']:
        raise ValueError('Wrong corpus version')
    local_tokenizers = {}
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        file = args.tokenizer / name
        wanted = {h for p, h in plan['inputs'].items() if p.endswith('/' + name)}
        got = digest(file.read_bytes())
        if wanted != {got}:
            raise ValueError('Tokenizer does not match published training plan')
        local_tokenizers[str(file)] = got
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    indexed = {(r['arm'], r['id']): r for r in material['records']}
    action_id = tokenizer.convert_tokens_to_ids('<tool_call>')
    decision_id = tokenizer.convert_tokens_to_ids('<think>')
    items, conflicts = [], []
    units = []
    for unit in corpus['rows']:
        ident = unit['id']
        documents = {}
        paths = {}
        for arm in ('action_only', 'decision_action'):
            documents[arm], paths[arm] = src.reference(unit['documents'][arm])
        request, request_path = src.reference(unit['request'])
        plain = target_tokens(tokenizer, documents['action_only']['turns'][0])
        joint = target_tokens(tokenizer, documents['decision_action']['turns'][0])
        if unit['target_kind'] == 'terminal_control':
            reason, action = [], joint
        else:
            boundary = joint.index(action_id)
            reason, action = joint[:boundary], joint[boundary:]
        action_ref = indexed['separate', ident + ':action']
        if plain != action or token_hash(action) != action_ref['target_sha256']:
            raise ValueError('Action token mismatch: ' + ident)
        if len(action) != action_ref['action_tokens']:
            raise ValueError('Action count mismatch')
        if reason:
            d = indexed['separate', ident + ':decision']
            if token_hash(reason) != d['target_sha256'] or len(reason) != d['decision_tokens']:
                raise ValueError('Reason target mismatch')
            if d['prompt_sha256'] != action_ref['prompt_sha256']:
                raise ValueError('This unit does not have the observed shared prefix')
            if (reason[0], action[0]) != (decision_id, action_id):
                raise ValueError('Unexpected first-token routing')
            tokens = material['totals']['separate']['family_component_tokens'][unit['family_id']]
            wd, wa = 1/(6*tokens['decision']), 1/(6*tokens['action'])
            probability = wd/(wd+wa)
            entropy = -probability*math.log(probability)-(1-probability)*math.log(1-probability)
            conflicts.append({'unit_id': ident, 'family_id': unit['family_id'],
                'shared_original_prompt_sha256': d['prompt_sha256'],
                'decision_first_token_id': reason[0], 'action_first_token_id': action[0],
                'decision_coefficient': wd, 'action_coefficient': wa,
                'isolated_optimal_decision_branch_probability': probability,
                'isolated_minimum_weighted_nll': (wd+wa)*entropy,
                'interpretation': 'Two output tasks at one prefix. Not measured model behavior or a proof that multiple valid outputs cannot be learned.'})
        # The targets retain the exact old token arrays. Task routing is added
        # to both arms. Image paths and all non-system messages stay identical.
        for component, ids in [('decision', reason), ('action', action)]:
            if not ids:
                continue
            messages = tagged_request(request, component)
            if messages[1:] != request[1:]:
                raise ValueError('Scene observations changed')
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if tokenizer.encode(decoded, add_special_tokens=False) != ids:
                raise ValueError('Target does not roundtrip')
            for arm in ('separate_tagged', 'paired_tagged'):
                draft = tokenizer.decode(reason, skip_special_tokens=False) if component == 'action' else ''
                prefix = action_prefix(arm, model_decision=draft, evaluation=False) if component == 'action' else ''
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                # Strict text tokenization check; image placeholders have not
                # yet been expanded by the multimodal processor.
                full_ids = tokenizer.encode(prompt_text+prefix+decoded, add_special_tokens=False)
                observed_ids = tokenizer.encode(prompt_text+prefix, add_special_tokens=False)
                if full_ids != observed_ids+ids:
                    raise ValueError('New target boundary merges across prefix')
                items.append({'unit_id': ident, 'family_id': unit['family_id'], 'arm': arm,
                    'component': component, 'target_kind': unit['target_kind'],
                    'source_document_path': paths['decision_action'], 'request_source_path': request_path,
                    'request_source_sha256': src.manifest[request_path], 'messages': messages,
                    'assistant_prefix': prefix, 'assistant_prefix_tokens': len(prefix_ids),
                    'target_text': decoded, 'target_token_ids': ids, 'target_sha256': token_hash(ids),
                    'common_prompt_sha256': digest(prompt_text.encode()),
                    'text_prefix_token_count_without_expanded_images': len(observed_ids),
                    'target_component_weight': 1/(6*material['totals']['separate']['family_component_tokens'][unit['family_id']][component]),
                    'full_multimodal_encoding_checked': False,
                    'decision_stage_stop': '</think>' if component == 'decision' else None})
        units.append({'id': ident, 'action_token_count': len(action), 'decision_token_count': len(reason),
                      'action_target_exact': True, 'reason_target_exact': True,
                      'images_unchanged_by_message_builder': True})
        print(ident, 'targets checked; routed material serialized', flush=True)
    totals = {}
    for arm in ('separate_tagged','paired_tagged'):
        chosen = [r for r in items if r['arm']==arm]
        totals[arm] = {'components':len(chosen), 'action_tokens':sum(len(r['target_token_ids']) for r in chosen if r['component']=='action'),
                      'decision_tokens':sum(len(r['target_token_ids']) for r in chosen if r['component']=='decision'),
                      'additional_reason_prefix_tokens':sum(r['assistant_prefix_tokens'] for r in chosen),
                      'same_prompt_different_task_targets':0}
        seen = {}
        for r in chosen:
            key = r['common_prompt_sha256']
            if key in seen and seen[key] != r['component']:
                raise ValueError('Routed tasks still collide')
            seen[key] = r['component']
    if totals['separate_tagged']['action_tokens'] != 1643 or totals['separate_tagged']['decision_tokens'] != 1332:
        raise ValueError('Material count differs')
    with (args.output/'prospective_components.jsonl').open('w') as f:
        for r in items:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(args.output/'routing_conflicts.json',conflicts)
    write(args.output/'source_manifest.json',{'pin':PIN,'git_sources':src.manifest,'local_tokenizer_hashes':local_tokenizers,
                                           'material_manifest_sha256':digest(material_path.read_bytes())})
    result={'source_units':len(units),'real_targets_reconstructed':len(units)*2,'original_routing_conflicts':len(conflicts),
            'prospective_totals':totals,'unit_checks':units,'model_calls':0,'training_updates':0,'new_decisions':0,
            'images_reprocessed':0,'training_ready':False,
            'isolated_branch_nll_lower_bound':sum(r['isolated_minimum_weighted_nll'] for r in conflicts),
            'elapsed_seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'pending':['Actual multimodal processor and label-mask integration',
                       'Deployment must use generated decisions, never held-out gold records',
                       'Fixed task-tagged deployment protocol and tool contract',
                       'Independent scenes and actual learned benefit; no efficacy predicted here'],
            'scope':'Ordinary task-routing control and source target audit, not a new research method or learning result.'}
    write(args.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='unit_checks'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
