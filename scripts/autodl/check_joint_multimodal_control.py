"""Encode available original/rebuilt views and exercise continuous policy plumbing.

All 24 units remain in the report. Missing views are not synthesized. The
generation probe explicitly replays target tokens; it is not model inference.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

from phase_deployment import deployment_input
from phase_objective import attach_objective_weights
from phase_policy import PhasePolicy, PhaseOutcomeError, TorchPhaseBackend, classify_stage
from routed_multimodal import encode_component, tensor_digest


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('material', 'equivalence', 'bindings', 'processor', 'scene-observation', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--reference-encoding', type=Path,
        default=Path('runs/autodl/20261110_stage_alignment_review/encoding_v2/encoding_records.json'),
        help='Actual split-arm encoding records for this version of rebuilt observations')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.generation.stopping_criteria import EosTokenCriteria
    torch.set_num_threads(1)
    source_hashes = {str(p): sha(p) for p in (a.material, a.equivalence, a.bindings, a.scene_observation, a.reference_encoding)}
    if sha(a.scene_observation) != '1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3':
        raise ValueError('Wrong upstream observation configuration')
    spec = importlib.util.spec_from_file_location('joint_observation_config', a.scene_observation)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    processor = AutoProcessor.from_pretrained(a.processor, local_files_only=True, trust_remote_code=False)
    native.configure_image_budget(processor, 65536)
    tok = processor.tokenizer; eos = [tok.eos_token_id]
    rows = [json.loads(s) for s in a.material.read_text().splitlines()]
    historical = {r['unit_id']: r for r in json.loads(a.equivalence.read_text())['material']}
    rebuilt = {r['unit_id']: r for r in json.loads(a.bindings.read_text())}
    old_vision = {r['unit_id']: r['transformed_sha256'] for r in json.loads(
        Path('runs/autodl/20261110_objective_source_review/material_manifest.json').read_text())['records']}
    rebuilt_vision = {r['unit_id']: r['tensor_sha256'] for r in json.loads(a.reference_encoding.read_text())
        if r['status'] == 'encoded'}
    records = []; classifications = []; observation = None; stream = None
    for c in rows:
        unit = c['unit_id']
        # Training chat serialization may contain a newline after native EOS.
        # Keep it supervised as in the source, but generation stops AT EOS.
        targets = c['target_token_ids']
        native_end = next((i+1 for i,t in enumerate(targets) if t in eos), len(targets))
        emitted_targets = targets[:native_end]
        kind = classify_stage(tok, emitted_targets, stage='joint', eos_ids=eos)
        if kind['kind'] not in ('tool_proposed', 'stop_proposed'):
            raise ValueError('Frozen joint target not executable: '+unit+': '+str(kind))
        classifications.append(dict(unit_id=unit, kind=kind['kind'], decision=kind.get('decision_raw') is not None,
            supervised_tokens_after_native_eos=targets[native_end:]))
        record = dict(unit_id=unit, target_tokens=len(c['target_token_ids']))
        messages = copy.deepcopy(c['messages']); images = []; paths = []; evidence = []
        if unit in rebuilt:
            source = rebuilt[unit]
            for name in ('export_receipt', 'annotation_receipt'):
                if sha(source[name]) != source[name+'_sha256']:
                    raise ValueError('Rebuilt observation receipt changed')
            refs = [dict(path=v['path'], local_path=v['path'], sha256=v['sha256']) for v in source['images']]
            expected = rebuilt_vision[unit]; record['observation_scope'] = 'versioned_rebuilt_not_historical_equivalent'
        elif historical[unit]['complete_five_views']:
            refs = []
            for v in historical[unit]['images']:
                eq = v.get('pixel_equivalent_recovery')
                refs.append(dict(path=v['path'], local_path=v['match']['local_path'] if v['match'] else eq['local_path'],
                    sha256=v['sha256'] if v['match'] else eq['replacement_file_sha256'],
                    rgb_sha256=None if v['match'] else eq['rgb_sha256']))
            expected = old_vision[unit]; record['observation_scope'] = 'original_or_verified_rgb_equivalent'
        else:
            records.append(dict(record, status='missing_original_views_retained'))
            continue
        for v in refs:
            p = Path(v['local_path'])
            if sha(p) != v['sha256']: raise ValueError('Image changed: '+str(p))
            with Image.open(p) as im: rgb = im.convert('RGB')
            if v.get('rgb_sha256') and hashlib.sha256(rgb.tobytes()).hexdigest() != v['rgb_sha256']:
                raise ValueError('Recovered RGB mismatch')
            images.append(rgb); paths.append(str(p)); evidence.append(dict(path=str(p), sha256=v['sha256']))
        slot = 0
        for m in messages:
            if not isinstance(m.get('content'), list): continue
            for part in m['content']:
                if part.get('type') == 'image':
                    if part['image'] != refs[slot]['path']: raise ValueError('Wrong image order')
                    part['image'] = paths[slot]; slot += 1
        if slot != 5: raise ValueError('Expected five views')
        cc = dict(c, component='joint')
        batch, boundary = encode_component(processor, cc, messages, images)
        attach_objective_weights(batch, cc)
        for name in ('pixel_values', 'image_grid_thw'):
            if tensor_digest(batch[name]) != expected[name]: raise ValueError('Vision differs from other arms')
        prefix = deployment_input(processor, messages, images, arm='joint_same_material_v1', stage='joint')
        if not torch.equal(prefix['input_ids'], batch['input_ids'][:, :boundary]):
            raise ValueError('Joint train/inference prefix differs')
        if any(not torch.equal(prefix[k], batch[k]) for k in ('pixel_values', 'image_grid_thw')):
            raise ValueError('Joint deployment vision differs')
        if batch['token_objective_weights'][batch['labels'] != -100].tolist() != c['token_objective_weights']:
            raise ValueError('Objective coefficients changed')
        record.update(status='encoded', input_tokens=boundary, sequence_tokens=batch['input_ids'].shape[1],
            objective_mass=float(batch['token_objective_weights'].sum()), images=evidence,
            decision_tokens=int(batch['decision_mask'].sum()), action_tokens=int(batch['action_mask'].sum()),
            tensors={k:tensor_digest(v) for k,v in batch.items()},
            common_vision_exact=True, joint_deployment_prefix_exact=True)
        records.append(record)
        if unit == 'waterloo_joined_host_stop':
            observation = dict(messages=messages, images=images,
                state_sha256=hashlib.sha256(b'encoding fixture, not a new scene state').hexdigest())
            stream = emitted_targets
        del batch, prefix
        print(unit, record['status'], flush=True)
    # The real production backend and policy with a clearly labelled generation
    # double. Tests one-call routing/EOS/accounting, not what a model would emit.
    class ArchivedOutput:
        training = False; device = 'cpu'
        generation_config = SimpleNamespace(eos_token_id=eos)
        def __init__(self, tokens): self.tokens = tokens; self.calls = []
        def generate(self, **kw):
            emitted = []; stop = EosTokenCriteria(eos_token_id=kw['eos_token_id'])
            for token in self.tokens[:kw['max_new_tokens']]:
                emitted.append(token)
                if bool(stop(torch.tensor([emitted]), None)[0]): break
            self.calls.append(dict(eos_ids=kw['eos_token_id'], emitted=len(emitted)))
            return torch.cat([kw['input_ids'], torch.tensor([emitted])], dim=1)
    probes = []
    for label, cap in (('continuous_stop', 1024), ('truncated_decision_is_not_stop', 1)):
        double = ArchivedOutput(stream)
        policy = PhasePolicy(TorchPhaseBackend(double, processor), arm='joint_same_material_v1',
            total_output_tokens=cap, decision_output_tokens=512, action_output_tokens=512,
            total_seconds=180, seed=17)
        try: result = policy.generate(observation)
        except PhaseOutcomeError as exc: result = exc.receipt
        assert result['generation_attempts'] == 1 and len(double.calls) == 1
        assert result['status'] == ('stop_proposed' if cap == 1024 else 'invalid_joint')
        if cap == 1024: assert result['output_tokens'] == len(stream)
        probes.append(dict(label=label, result=result, double_calls=double.calls, actual_model_calls=0))
    # Malformed control tokens must not turn into a free STOP or accepted tool.
    invalid = ['<think>open<|im_end|>', '<think>x</think>',
               '<think>x</think><think>y</think><|im_end|>',
               '<|im_end|><|im_end|>']
    for text in invalid:
        assert classify_stage(tok, tok.encode(text, add_special_tokens=False), stage='joint', eos_ids=eos)['kind'] == 'invalid'
    if {str(p):sha(p) for p in (a.material, a.equivalence, a.bindings, a.scene_observation, a.reference_encoding)} != source_hashes:
        raise ValueError('Inputs changed during execution')
    summary = dict(status='partial_multimodal_encoding_and_joint_host_replay_completed',
        units=len(records), encoded=sum(r['status']=='encoded' for r in records),
        missing_views=sum(r['status']!='encoded' for r in records), records=records,
        target_classifications=classifications, production_backend_replay=probes,
        malformed_outputs_rejected=len(invalid), source_sha256=source_hashes,
        actual_model_calls=0, optimizer_updates=0, paid_api_calls=0, new_complete_scenes=0,
        training_ready=False, full_runtime_ready=False,
        seconds=time.monotonic()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Real processor and unchanged view tensors; target-token replay only. No training or policy quality evidence. Missing units retained; no reduced-corpus training.')
    (a.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','target_classifications','production_backend_replay','source_sha256')}))


if __name__ == '__main__': main()
