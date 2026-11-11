"""Recompute a closed diagnostic from saved outputs; no model imports or calls.

Decision accuracy and program accuracy are deliberately separate. The small
version-space calculation illustrates standard weak-supervision limits, not a
new learning algorithm or a measured training improvement.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import time
import zipfile

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from intent_geometry_diagnostic import upstream_functions


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def task_text(request):
    return request[-1]['content'][-1]['text'].split('## Task\n', 1)[1].split('\n\n## Memory\n', 1)[0]


def strict_program(args):
    people = args['people_by_phase']
    assert set(people) == {'work', 'meal'}
    for ids in people.values():
        assert isinstance(ids, list) and len(set(ids)) == len(ids)
        assert set(ids) <= {'resident', 'guest'}
    phases = args['materials_active_phases']
    assert phases is None or (isinstance(phases, list) and len(set(phases)) == len(phases)
                              and set(phases) <= {'work', 'meal'})


def bind(phases, facts):
    if phases == ['work']:
        return facts['one_work_region_fits'] and facts['two_meal_regions_fit']
    if set(phases or []) == {'work', 'meal'}:
        return facts['work_materials_plus_two_meal_regions_fit']
    return None


def mml(logits, accepted):
    norm = max(logits)
    exps = [math.exp(x - norm) for x in logits]
    p = [x / sum(exps) for x in exps]
    mass = sum(p[i] for i in accepted)
    if not mass:
        return None
    gradient = [v - (v / mass if i in accepted else 0.) for i, v in enumerate(p)]
    return {'loss': -math.log(mass), 'gradient': gradient}


def version_space(cases, raw_programs):
    # This grammar deliberately freezes work presence and people. Conclusions
    # are conditional on this two-hypothesis class and externally supplied labels.
    grammar = [['work'], ['work', 'meal']]
    groups = {}
    for c in cases:
        groups.setdefault(c['task_sha256'], []).append(c)
    result = []
    for task_hash, group in groups.items():
        by_world = {c['world']: c for c in group}
        ordered = [by_world['original_table'], by_world['smaller_table']]
        signatures = [[bind(g, c['common_geometry']['geometry_fact']) for c in ordered] for g in grammar]
        labels = [c['expected_shared_feasible'] for c in ordered]
        accepted_one = [i for i, sig in enumerate(signatures) if sig[0] is labels[0]]
        accepted_two = [i for i, sig in enumerate(signatures) if sig == labels]
        observed = {tuple(raw_programs[c['id']]['materials_active_phases']) for c in group}
        surviving_observed = [list(g) for g in observed
                              if [bind(list(g), c['common_geometry']['geometry_fact']) for c in ordered] == labels]
        loss_one = mml([0., 0.], accepted_one)
        loss_two = mml([0., 0.], accepted_two)
        assert accepted_one == [0, 1] and len(accepted_two) == 1
        assert max(map(abs, loss_one['gradient'])) < 1e-15
        # Finite differences check only the analytic expression, not a model.
        eps = 1e-5
        errors = []
        for i in range(2):
            plus = [0., 0.]; minus = [0., 0.]
            plus[i] += eps; minus[i] -= eps
            fd = (mml(plus, accepted_two)['loss'] - mml(minus, accepted_two)['loss']) / (2 * eps)
            errors.append(abs(fd - loss_two['gradient'][i]))
        assert max(errors) < 1e-9
        result.append(dict(task_sha256=task_hash, grammar_material_phases=grammar,
                           world_order=[c['world'] for c in ordered], signatures=signatures,
                           external_gold_labels=labels, original_only_survivors=accepted_one,
                           two_world_survivors=accepted_two, observed_pool_survivors=surviving_observed,
                           original_only_MML_at_uniform_logits=loss_one,
                           two_world_MML_at_uniform_logits=loss_two,
                           finite_difference_max_error=max(errors),
                           labels_generated_from_current_parser=False,
                           general_semantic_identification_proved=False))
    return dict(groups=result, gradient_updates=0, model_calls=0,
                meaning='Standard MML on a finite grammar with externally labelled worlds; not new theory or a training result.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); root = a.root.resolve(); start = time.monotonic()
    assert root.parent == R / 'runs/autodl'
    protocol = read(root / 'protocol.json'); result = read(root / 'model_result.json')
    cases = read(root / 'cases_and_gold.json'); by_case = {c['id']: c for c in cases}
    assert result['status'] == 'completed' and result['model_calls'] == 8
    assert len(result['attempts']) == 8 and result['program_arm_ran']
    checks = []
    for path, expected in {**protocol['requests'], **protocol['source_hashes']}.items():
        assert sha(path) == expected, path
        checks.append(dict(path=path, sha256=expected, category='frozen_request_or_source'))
    for row in read(root / 'preparation_v1/archive_note.json')['original_files']:
        assert sha(root / 'preparation_v1' / row['path']) == row['sha256']
    assert read(root / 'preparation_v1/archive_note.json')['model_calls_before_edit'] == 0
    build, parse = upstream_functions()
    preflight = {(r['case'], r['arm']): r['encoding'] for r in read(root / 'encoding_preflight.json')['rows']}
    rows = []; raw_programs = {}; reviews = []; total_in = total_out = 0
    for item in result['attempts']:
        case = by_case[item['case']]; arm = item['arm']
        request_path = root / f"{case['id']}_{arm}_request.json"
        request = read(request_path); evidence = Path(item['receipt_dir'])
        assert evidence.is_relative_to(root / 'model_calls')
        assert read(evidence / 'request.json') == request
        text = task_text(request)
        assert hashlib.sha256(text.encode()).hexdigest() == case['task_sha256']
        assert request == build(request[0]['content'], text,
                                json.dumps(case['common_geometry'], ensure_ascii=False, allow_nan=False), [])
        opposite = read(root / f"{case['id']}_{'program' if arm == 'direct' else 'direct'}_request.json")
        assert request[1:] == opposite[1:]
        think, calls = parse((evidence / 'raw_output.txt').read_text())
        assert not think and calls == item['parsed'] and len(calls) == 1
        _, text_calls = parse((evidence / 'text_output.txt').read_text())
        assert text_calls == calls
        args = calls[0]['arguments']; quotes = args['evidence_quotes']
        assert quotes and all(isinstance(q, str) and q and q in text for q in quotes)
        assert calls[0]['name'] == ('assess_shared_surface' if arm == 'direct' else 'compile_activity_program')
        semantic = None; people_correct = None; materials_correct = None
        if arm == 'direct':
            assert set(args) == {'feasible', 'evidence_quotes'}
            pred = args['feasible']; assert type(pred) is bool
        else:
            assert set(args) == {'people_by_phase', 'materials_active_phases', 'evidence_quotes'}
            strict_program(args)
            people_correct = args['people_by_phase'] == case['expected_people']
            materials_correct = set(args['materials_active_phases'] or []) == set(case['expected_material_phases'])
            semantic = people_correct and materials_correct
            pred = bind(args['materials_active_phases'], case['common_geometry']['geometry_fact']) if people_correct else None
            raw_programs[case['id']] = args
            reviews.append(dict(case=case['id'], source_quotes=quotes,
                                emitted_material_phases=args['materials_active_phases'],
                                expected_material_phases=case['expected_material_phases'],
                                quote_program_contradiction=(case['intent'] == 'keep' and not materials_correct),
                                reviewer='Author self-review against explicit task clauses, not independent human annotation',
                                semantic_entailment_metric='No learned entailment checker or external judge used'))
        decision_correct = pred is case['expected_shared_feasible']
        # The old scorer's 'correct' includes semantic fields for program arm.
        assert item['score']['correct'] == (decision_correct and (semantic if semantic is not None else True))
        rec = read(evidence / 'receipt.json')
        assert rec['status'] == 'completed' and rec['ended_with_eos'] and rec['generation_started']
        assert rec['output_tokens'] < rec['max_new_tokens'] and not rec['sampling']
        assert rec['fallbacks'] == rec['paid_api_calls'] == 0 and rec['images'] == []
        enc = preflight[(case['id'], arm)]
        assert all(rec[k] == enc[k] for k in ['input_tokens', 'input_shapes', 'chat_template_sha256'])
        assert rec['input_tokens'] <= protocol['max_input_tokens']
        assert rec['metadata'] == {'case': case['id'], 'arm': arm}
        total_in += rec['input_tokens']; total_out += rec['output_tokens']
        rows.append(dict(case=case['id'], arm=arm, intent=case['intent'], world=case['world'],
                         expected=case['expected_shared_feasible'], prediction=pred,
                         decision_correct=decision_correct, semantic_program_correct=semantic,
                         people_fields_correct=people_correct, material_fields_correct=materials_correct,
                         literal_quotes=True, input_tokens=rec['input_tokens'], output_tokens=rec['output_tokens'],
                         elapsed_seconds=rec['elapsed_seconds'], peak_cuda_allocated_bytes=rec['peak_cuda_allocated_bytes'],
                         exact_request_sha256=sha(request_path), raw_output_sha256=sha(evidence / 'raw_output.txt')))
    assert len(list((root / 'model_calls').glob('*/receipt.json'))) == 8
    assert total_in == 5872 and total_out == 628
    aggregate = {}
    for arm in ['direct', 'program']:
        arm_rows = [r for r in rows if r['arm'] == arm]
        aggregate[arm] = dict(cases=len(arm_rows), decision_correct=sum(r['decision_correct'] for r in arm_rows),
                              program_exact_correct=sum(r['semantic_program_correct'] is True for r in arm_rows) if arm == 'program' else None,
                              input_tokens=sum(r['input_tokens'] for r in arm_rows), output_tokens=sum(r['output_tokens'] for r in arm_rows))
    assert [aggregate[k]['decision_correct'] for k in ['direct', 'program']] == [3, 3]
    assert aggregate['program']['program_exact_correct'] == 2
    dump(root / 'source_entailment_review.json', dict(rows=reviews, independent_annotation=False))
    dump(root / 'training_identifiability.json', version_space(cases, raw_programs))
    schema_path = R.parent.parent / 'professional_data_ingest_20261104/native_src/stage_schemas.py'
    schema_module = runpy.run_path(str(schema_path))
    targets = read(root / 'repaired_program_teacher_records.json')['records']
    admission = []
    for target in targets:
        call = target['corrected_target']
        errors = schema_module['validate_tool_call']('furniture', call['name'], call['arguments'])
        assert errors and any('unknown tool' in e for e in errors)
        admission.append(dict(case=target['case'], native_stage='furniture', errors=errors,
                              custom_component_schema_only=True, admitted_to_native_action_SFT=False))
    dump(root / 'native_training_admission.json', dict(rows=admission, schema_source=str(schema_path),
         schema_sha256=sha(schema_path), original_validator_executed=True,
         original_validator_changed=False, model_calls=0, training_updates=0))
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    protected = read(R / 'runs/autodl/20261111_feedback_information_control/source_before_after.json')
    protected_rows = []
    for r in protected['original_inputs']:
        actual = sha(r['path']); assert actual == r['before'], r['path']
        protected_rows.append(dict(path=r['path'], before=r['before'], after=actual, same=True))
    assert len(protected_rows) == 579
    dump(root / 'source_before_after.json', dict(count=579, all_originals_unchanged=True, original_inputs=protected_rows))
    audit = dict(status='raw_outputs_recomputed', rows=rows, aggregate=aggregate,
                 source_hashes=checks, protected_sources_rehashed=579,
                 decision_metric_not_same_as_program_metric=True,
                 model_generate_calls=8, input_tokens=total_in, output_tokens=total_out,
                 all_tokens=total_in + total_out, all_run_seconds=result['elapsed_seconds'],
                 peak_host_rss_kib=result['peak_rss_kib'],
                 peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
                 no_per_method_speed_claim='First direct call includes checkpoint loading; later calls warm',
                 models_called_by_this_audit=0, training_updates=0, paid_API_calls=0,
                 independent_rooms=0, exposed_source_families=1, new_whole_scenes=0,
                 previous_eight_generation_budget='closed', audit_wall_seconds=time.monotonic() - start)
    dump(root / 'independent_recompute.json', audit)
    required = ['protocol.json', 'cases_and_gold.json', 'model_result.json', 'encoding_preflight.json',
                'ordinary_repair_controls.json', 'repaired_program_teacher_records.json',
                'source_entailment_review.json', 'training_identifiability.json', 'summary.csv',
                'independent_recompute.json', 'source_before_after.json', 'algorithm_and_training.txt',
                'reading_ledger.json', 'method_difference.csv', 'reproduce.txt', 'stage_report.html',
                'native_training_admission.json']
    for name in required:
        assert (root / name).is_file(), name
    payloads = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in
                      {'evidence.zip', 'delivery_manifest.json', 'delivery_verification.json'})
    scripts = [R / 'scripts/autodl' / s for s in
               ['intent_geometry_diagnostic.py', 'repair_intent_programs.py', 'audit_intent_geometry_gate.py']]
    manifest = [dict(path=str(p.relative_to(R)), sha256=sha(p), bytes=p.stat().st_size) for p in payloads + scripts]
    dump(root / 'delivery_manifest.json', dict(files=manifest, original_and_corrected_outputs_both_preserved=True))
    with zipfile.ZipFile(root / 'evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in payloads + scripts + [root / 'delivery_manifest.json']:
            z.write(p, str(p.relative_to(R)))
    with zipfile.ZipFile(root / 'evidence.zip') as z:
        assert z.testzip() is None
        for row in manifest:
            assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
    dump(root / 'delivery_verification.json', dict(status='verified', manifest_files=len(manifest),
         evidence_zip_sha256=sha(root / 'evidence.zip'), evidence_zip_bytes=(root / 'evidence.zip').stat().st_size,
         zip_crc_and_member_hashes_checked=True, all_579_protected_sources_unchanged=True,
         scientific_success_claim='None: component diagnostic and ordinary offline controls only'))
    print(json.dumps(dict(status='verified', direct_decisions='3/4', program_decisions='3/4',
                          program_semantics='2/4', calls_in_this_audit=0, manifest_files=len(manifest)), ensure_ascii=False))


if __name__ == '__main__':
    main()
