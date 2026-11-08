"""Build and validate finite equal-material training views on CPU, no optimizer."""
import argparse
import json
import os
from pathlib import Path

from materialize_native_completion import read, write, sha, ref, require, locked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args()
    plan = read(args.plan)
    for path, digest in plan['inputs'].items():
        require(sha(path) == digest, 'Frozen input changed: ' + path)
    require(plan['training_updates'] == plan['model_calls'] == 0, 'CPU preparation only')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    import torch
    from conditional_material_dataset import component_batch, tensor_digest, EqualMaterialExamples
    from train_edit_learning_study import normalization_check
    torch.set_num_threads(2)
    enc = locked(plan['encoding'])
    corpus = locked(plan['corpus'])
    indexed = {(r['arm'], r['id']): r for r in enc['records']}
    require(len(indexed) == len(enc['records']) == 48, 'Unexpected source records')
    records, equivalence, coverage = [], [], []
    print('Equal-material CPU preparation started; no model or optimizer', flush=True)
    for unit in corpus['rows']:
        ident = unit['id']
        originals = {arm: indexed[arm, ident] for arm in ('action_only', 'decision_action')}
        require(originals['action_only']['prompt_sha256'] == originals['decision_action']['prompt_sha256'], 'Prompt mismatch')
        doc = locked(unit['documents']['decision_action'])
        has_decision = originals['decision_action']['decision_tokens'] > 0
        content = doc['turns'][0]['content']
        source_id = json.loads(content)['principle_applicability']['source_decision_id'] if has_decision else None
        coverage.append(dict(id=ident, family_id=unit['family_id'], source_decision_id=source_id,
            target_kind=unit['target_kind'], professional_decision_position=has_decision,
            input=unit['input'], request=unit['request'], source_document=unit['documents']['decision_action']))
        for component in (['decision', 'action'] if has_decision else ['action']):
            pair = []
            for arm in ('separate', 'paired'):
                source_arm = 'decision_action' if component == 'decision' or arm == 'paired' else 'action_only'
                source = originals[source_arm]
                row = dict(id=ident + ':' + component, unit_id=ident, family_id=unit['family_id'],
                    arm=arm, component=component, target_kind=unit['target_kind'],
                    source_tensor=dict(path=source['tensor_path'], sha256=source['tensor_sha256']),
                    source_record_arm=source_arm, source_prompt_tokens=source['prompt_tokens'])
                batch = component_batch(row)
                boundary = source['prompt_tokens']
                require(bool((batch['labels'][0, :boundary] == -100).all()), 'Input supervision leak')
                row.update(input_tokens=int(batch['input_ids'].shape[1]),
                    action_tokens=int(batch['action_mask'].sum()), decision_tokens=int(batch['decision_mask'].sum()),
                    transformed_sha256={k: tensor_digest(batch[k]) for k in ('input_ids', 'labels', 'pixel_values', 'image_grid_thw')},
                    target_sha256=tensor_digest(batch['labels'][batch['labels'] != -100]),
                    prompt_sha256=tensor_digest(batch['input_ids'][0, :boundary]))
                records.append(row)
                pair.append(row)
            require(pair[0]['target_sha256'] == pair[1]['target_sha256'], 'Different supervised materials')
            require(pair[0]['prompt_sha256'] == pair[1]['prompt_sha256'], 'Different observed input')
            for k in ('pixel_values', 'image_grid_thw'):
                require(pair[0]['transformed_sha256'][k] == pair[1]['transformed_sha256'][k], 'Images differ')
            if component == 'decision' or not has_decision:
                require(pair[0]['transformed_sha256'] == pair[1]['transformed_sha256'], 'Unintended arm difference')
            else:
                require(pair[1]['input_tokens'] - pair[0]['input_tokens'] == originals['decision_action']['decision_tokens'], 'Unexpected action context difference')
            equivalence.append(dict(id=pair[0]['id'], exact_target_tokens_equal=True,
                exact_observed_prompt_equal=True, image_tensors_equal=True,
                paired_extra_context_tokens=pair[1]['input_tokens']-pair[0]['input_tokens']))
        print(ident + ': same material verified', flush=True)
    out = Path(plan['output'])
    totals, consumption = {}, {}
    for arm in ('separate', 'paired'):
        selected = [r for r in records if r['arm'] == arm]
        ds = EqualMaterialExamples(selected)
        totals[arm] = dict(components=len(selected), families=len(ds.families),
            **{k: sum(r[k] for r in selected) for k in ('input_tokens', 'action_tokens', 'decision_tokens')},
            family_component_tokens=ds.counts,
            attention_length_squared_proxy=sum(r['input_tokens']**2 for r in selected))
        consumption[arm] = normalization_check(ds, Path(plan['runtime']) / arm)
    require(totals['separate']['family_component_tokens'] == totals['paired']['family_component_tokens'], 'Family weights differ')
    require(not torch.cuda.is_initialized(), 'Unexpected GPU initialization')
    write(out / 'MATERIAL_MANIFEST.json', dict(schema='equal-material-components-v1', records=records,
        source_encoding=plan['encoding'], source_corpus=plan['corpus'], totals=totals,
        training_authorized=False, new_sources=0, optimizer_updates=0))
    write(out / 'SOURCE_COVERAGE.json', dict(rows=coverage,
        reason_positions=sum(r['professional_decision_position'] for r in coverage),
        unique_principles=len({r['source_decision_id'] for r in coverage if r['source_decision_id']}),
        reason_families=len({r['family_id'] for r in coverage if r['source_decision_id']}),
        all_families=len({r['family_id'] for r in coverage}),
        limits='Existing source-guided reconstructions, not new historical designer trajectories or independent quality labels.'))
    write(out / 'VALIDATION.json', dict(status='passed', equivalence=equivalence,
        trainer_consumption=consumption, totals=totals, model_loaded=False,
        cuda_initialized=False, optimizer_updates=0, backward_executed=False,
        serialization='Views of previously verified official-loader tensors, not a new tokenization.',
        limits='Equal target tokens/family weighting/forward counts do not equal exact compute. No performance measured.'))
    print(json.dumps(dict(status='passed', totals=totals)), flush=True)


if __name__ == '__main__':
    main()
