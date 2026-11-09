"""CPU prefix compatibility check; saved teacher text is a fixture, never inference.

Missing decisions and images remain missing. This cannot verify model generation,
the host's stop handling, or native scene execution.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import resource
import time

from audit_material_routing import digest, write
from audit_decision_phase_support import raw_messages
from phase_deployment import deployment_input, decision_stop_ids
from routed_multimodal import resolve_messages, encode_component, tensor_digest


def main():
    ap = argparse.ArgumentParser()
    for name in ('material', 'source', 'availability', 'processor', 'scene-observation', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier receipts')
    tick = time.perf_counter()
    material = [json.loads(x) for x in args.material.read_text().splitlines()]
    source = {r['unit_id']: r for r in
              [json.loads(x) for x in args.source.read_text().splitlines()]
              if r['arm'] == 'paired_tagged' and r['component'] == 'action'}
    views = {r['unit_id']: r for r in json.loads(args.availability.read_text())['rows']}
    if digest(args.scene_observation.read_bytes()) != '1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3':
        raise ValueError('Image budget source changed')
    spec = importlib.util.spec_from_file_location('original_image_contract', args.scene_observation)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True, trust_remote_code=False)
    official.configure_image_budget(processor, 65536)
    stop_ids = decision_stop_ids(processor.tokenizer)
    rows = []
    for c in material:
        row = {k:c[k] for k in ('unit_id', 'arm', 'stage')}
        view = views[c['unit_id']]
        if not view['complete_five_views']:
            row['status'] = 'missing_original_views'; rows.append(row); continue
        original = source[c['unit_id']]
        if original['request_source_sha256'] != c['source_request_sha256']:
            raise ValueError('Source request identity differs')
        raw, paths = resolve_messages(raw_messages(original), view['images'])
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        paired_action = c['arm'] == 'paired_phase_v3' and c['stage'] == 'action'
        fixture = c['assistant_prefix'].rstrip() if paired_action and c['assistant_prefix'] else None
        if paired_action and fixture is None:
            try:
                deployment_input(processor, raw, images, arm=c['arm'], stage=c['stage'])
            except ValueError as exc:
                if 'requires its actual generated decision' not in str(exc): raise
            else:
                raise AssertionError('Missing paired decision was silently accepted')
            row.update(status='no_paired_decision_target',
                       meaning='Action-only training example; deployed paired action needs a generated D. No teacher-prefix compatibility is asserted.')
            rows.append(row); continue
        deploy = deployment_input(processor, raw, images, arm=c['arm'], stage=c['stage'], generated_decision=fixture)
        train_messages, _ = resolve_messages(c['messages'], view['images'])
        full, boundary = encode_component(processor, dict(c, component=c['stage']), train_messages, images)
        assert torch.equal(deploy['input_ids'], full['input_ids'][:, :boundary])
        assert torch.equal(deploy['attention_mask'], full['attention_mask'][:, :boundary])
        assert all(torch.equal(deploy[k], full[k]) for k in ('pixel_values', 'image_grid_thw'))
        assert not any(k in deploy for k in ('labels', 'action_mask', 'decision_mask', 'token_objective_weights'))
        row.update(status='prefix_equal', prefix_tokens=boundary,
                   input_ids_sha256=tensor_digest(deploy['input_ids']),
                   teacher_decision_used_only_as_encoding_fixture=fixture is not None)
        rows.append(row)
    rejected = []
    # These checks must fail before invoking the processor or reading images.
    for name, kwargs in [
        ('decision_with_answer', dict(arm='paired_phase_v3', stage='decision', generated_decision='<think>x</think>')),
        ('separate_with_answer', dict(arm='separate_phase_v3', stage='action', generated_decision='<think>x</think>')),
        ('incomplete_decision', dict(arm='paired_phase_v3', stage='action', generated_decision='<think>x')),
        ('action_in_decision', dict(arm='paired_phase_v3', stage='action', generated_decision='<think>x</think><tool_call>{}</tool_call>')),
    ]:
        try: deployment_input(None, [], [], **kwargs)
        except ValueError: rejected.append(name)
        else: raise AssertionError(name)
    counts = dict(Counter(r['status'] for r in rows))
    assert counts == {'missing_original_views':52, 'prefix_equal':14, 'no_paired_decision_target':2}
    result = dict(status='partial_prefix_validation', rows=rows, counts=counts,
                  stop_token_ids=stop_ids, rejected_invalid_inputs=rejected,
                  full_corpus_training_ready=False, model_calls=0, optimizer_updates=0,
                  scene_edits=0, model_weights_loaded=False, cuda_initialized=torch.cuda.is_initialized(),
                  missing_decision_scope='14 of 24 states have no D target in the fixed full material; two are among the available image states.',
                  scope='Original-five-view prefix encoding only. Teacher D is a fixture, not a model output or evaluation result. Host generation, stopping and scene rollout are untested.',
                  source_sha256={str(p):digest(p.read_bytes()) for p in (args.material, args.source, args.availability,
                      args.scene_observation, Path(__file__), Path(__file__).with_name('phase_deployment.py'))},
                  wall_seconds=time.perf_counter()-tick, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','counts','wall_seconds','peak_rss_kib','model_calls')}))


if __name__ == '__main__':
    main()
