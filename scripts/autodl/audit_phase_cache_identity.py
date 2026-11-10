"""Check exact phase-prefix identity against actual model input receipts.

CPU encoding only. Establishes whether whole decision KV could be reused for
the action system prompt; no teacher decision or invented valid D is inserted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    args = p.parse_args()
    output = args.stage / 'prefix_cache_audit.json'
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((args.stage / 'protocol.json').read_text())
    obs = json.loads(Path(protocol['phase_input']).read_text())
    def reject(*a, **k):
        raise RuntimeError('CPU input replay is offline')
    socket.socket.connect = reject; socket.create_connection = reject
    work = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    sys.path.insert(0, str(work / 'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine
    from phase_deployment import deployment_input
    from routed_multimodal import tensor_digest
    from PIL import Image
    engine = LocalQwenEngine(protocol['checkpoint'], args.stage / 'cpu_cache_identity',
        max_pixels=65536, max_input_tokens=16384, max_new_tokens=512)
    images = []
    try:
        for row in obs['images']:
            with Image.open(row['path']) as im:
                images.append(im.convert('RGB'))
        batches = {phase: deployment_input(engine.processor, obs['messages'], images,
                                          arm='separate_phase_v3', stage=phase)
                   for phase in ('action', 'decision')}
        ids = {phase: batch['input_ids'][0].tolist() for phase, batch in batches.items()}
        lcp = 0
        for left, right in zip(ids['action'], ids['decision']):
            if left != right:
                break
            lcp += 1
        actual = json.loads((args.stage / 'actual_v1/client_results.json').read_text())
        direct = actual[0]['result']['stages'][0]
        decision = actual[2]['result']['stages'][0]
        hashes = {phase: {k: tensor_digest(v) for k, v in batch.items()}
                  for phase, batch in batches.items()}
        assert hashes['action'] == direct['input_tensor_sha256']
        assert hashes['decision'] == decision['input_tensor_sha256']
        assert hashes['action']['pixel_values'] == hashes['decision']['pixel_values']
        assert 0 < lcp < min(len(x) for x in ids.values())
        assert engine._model is None
        result = {'status': 'exact_CPU_reencoding_matches_actual_forward_inputs',
            'input_lengths': {phase: len(values) for phase, values in ids.items()},
            'longest_common_prefix_tokens': lcp,
            'first_different_token': {phase: {'id': values[lcp],
                'decoded': engine.processor.tokenizer.decode([values[lcp]])} for phase, values in ids.items()},
            'input_tensor_sha256': hashes,
            'pixel_inputs_identical': True, 'whole_prefixes_identical': False,
            'entire_decision_KV_valid_for_action_without_recompute': False,
            'why': 'System mode text differs before later observation tokens. '
                   'Full D-stage causal KV is not the action-stage prefix. A valid shared-prefix '
                   'optimization would need a separate proof/implementation; current backend starts fresh.',
            'possible_reuse_not_measured': 'Identical vision inputs/common token prefix may support separate '
                                        'optimizations, but no vision-feature or KV reuse is implemented here.',
            'paired_action_actually_executed': False,
            'teacher_or_synthetic_D_inserted': False, 'model_loaded': False,
            'new_model_forwards': 0,
            'limits': 'Cost/serialization evidence only; does not prove learned decision benefit or full-scene quality.'}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'lcp_tokens': lcp, 'lengths': result['input_lengths'],
                          'whole_KV_reuse_valid': False, 'new_model_forwards': 0}))
    finally:
        for im in images:
            im.close()


if __name__ == '__main__':
    main()
