"""Exercise the full-stage wrapper's evidence routing on archived observations.

The policy is an observation recorder that returns STOP, never a model or a
method score. Image references are compared, not loaded/rendered. Temporary
host logs live in tmpfs and only comparison receipts are persisted.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from check_host_observation_alignment import visible_images
from materialize_host_evidence import public_facts
from phase_host import StageBudget, digest, furniture_driver
from scene_evidence import (STATE_MARKER, append_host_evidence,
                            canonical_host_evidence, text_slots, visible_scene_state)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ObservationRecorder:
    arm = 'separate_direct_v3'
    total_tokens = 1
    seconds = 10
    seed = 0

    def __init__(self, captured):
        self.captured = captured

    def generate(self, obs):
        self.captured.append(copy.deepcopy(obs))
        return dict(status='stop_proposed', tool_calls=[],
                    state_sha256=obs['state_sha256'], generation_attempts=0,
                    input_tokens=0, output_tokens=0,
                    output_token_accounting_complete=True)


async def check(a):
    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)
    source = [json.loads(s) for s in a.source.read_text().splitlines()]
    material = [json.loads(s) for s in a.material.read_text().splitlines()]
    assert len(source) == len(material) == 68
    hashes = {str(p): sha(p) for p in (a.source, a.material)}
    driver_factory = furniture_driver
    if a.legacy_host_source:
        spec = importlib.util.spec_from_file_location('archived_phase_host', a.legacy_host_source)
        legacy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = legacy
        spec.loader.exec_module(legacy)
        driver_factory = legacy.furniture_driver
        hashes[str(a.legacy_host_source)] = sha(a.legacy_host_source)
    bindings = {}
    if a.bindings:
        hashes[str(a.bindings)] = sha(a.bindings)
        bindings = {r['unit_id']: r for r in json.loads(a.bindings.read_text())}
    rows = []
    delegated = []
    budget = StageBudget(1, 1, 1, 1, 30)

    async def original_driver(**kwargs):
        delegated.append(kwargs['env'].stage)
        return ['original_stage']

    for old, expected in zip(source, material):
        assert {k:v for k,v in old.items() if k != 'messages'} == {
            k:v for k,v in expected.items() if k != 'messages'}
        # The newer material only substitutes independently versioned views.
        # Preserve source prose/state/schemas; do not derive them from targets.
        base = copy.deepcopy(old['messages'])
        images = iter(visible_images(expected['messages']))
        for m in base:
            if isinstance(m.get('content'), list):
                for part in m['content']:
                    if part.get('type') == 'image':
                        part['image'] = next(images)
        assert next(images, None) is None
        schemas, state = public_facts(base)
        if old['unit_id'] in bindings:
            binding = bindings[old['unit_id']]
            path = Path(binding['export_receipt']).parent / 'actual_room_state.json'
            from native_state_override import canonical as runtime_digest
            runtime_state = json.loads(path.read_text())
            assert runtime_digest(runtime_state) == binding['runtime_state_sha256']
            hashes[str(path)] = sha(path)
            # Rebase only the visible part of the embedded archived state onto
            # the measured runtime state. Preserve extra fields by OBJECT ID.
            objects = state['objects']
            assert isinstance(objects, dict)
            by_id = {o['object_id']: (k,o) for k,o in objects.items()}
            projected = visible_scene_state(runtime_state)['objects']
            assert set(by_id) == {o['object_id'] for o in projected}
            state['objects'] = {by_id[o['object_id']][0]: dict(by_id[o['object_id']][1], **o)
                                for o in projected}
            for slot, key in text_slots(base):
                if STATE_MARKER in slot[key]:
                    pre, post = slot[key].split(STATE_MARKER)
                    _, end = json.JSONDecoder().raw_decode(post)
                    slot[key] = pre + STATE_MARKER + json.dumps(state) + post[end:]
            state = runtime_state  # Port observes the actual runtime record.
        assert canonical_host_evidence(base, schemas=schemas, before_state=state,
                                       host_history=[]) == expected['messages']
        for fmt in (('canonical_v2',) if a.legacy_route else ('canonical_v2', 'append_v1')):
            captured = []
            with tempfile.TemporaryDirectory(prefix='stage_route_', dir='/dev/shm') as td:
                env = SimpleNamespace(stage='furniture', steps_dir=Path(td), mem={},
                    scene=SimpleNamespace(to_state_dict=lambda: copy.deepcopy(state)),
                    tool_map={k:SimpleNamespace(params_json_schema=v) for k,v in schemas.items()})
                env.observation = lambda: {'messages': copy.deepcopy(base), 'step_idx': 0}
                # A dispatch is a test failure: this regression has no actions.
                async def no_step(*args, **kwargs):
                    raise AssertionError('Unexpected native dispatch')
                env.step = no_step
                module = SimpleNamespace(driver_loop=original_driver)
                before = digest(state)
                kwargs = dict(policy=ObservationRecorder(captured), budget=budget,
                    allowed_tools=tuple(schemas), load_images=visible_images,
                    acceptance=lambda _: dict(state_sha256=digest(state), full_scene_success=None))
                if not a.legacy_route:
                    kwargs['evidence_format'] = fmt
                with driver_factory(module, **kwargs):
                    if not rows:
                        for stage in ('floor_plan', 'wall', 'ceiling'):
                            assert await module.driver_loop(env=SimpleNamespace(stage=stage),
                                policy=object(), max_steps=1) == ['original_stage']
                    await module.driver_loop(env=env, policy=object(), max_steps=1)
                assert module.driver_loop is original_driver
                receipt = json.loads((Path(td)/'phase_host/result.json').read_text())
                assert len(captured) == 1 and digest(state) == before
                assert receipt['native_calls'] == receipt['generation_attempts'] == 0
                target = (expected['messages'] if fmt == 'canonical_v2' else
                          append_host_evidence(base, schemas=schemas, before_state=state, host_history=[]))
                matches = captured[0]['messages'] == target
                logged = (receipt.get('evidence_format') == fmt and
                          receipt['rounds'][0].get('evidence_format') == fmt and
                          captured[0].get('evidence_format') == fmt)
                rows.append(dict(unit_id=old['unit_id'], arm=old['arm'], stage=old['stage'],
                    requested_format=fmt, messages_match=matches, format_logged=logged,
                    expected_messages_sha256=digest(target),
                    observed_messages_sha256=digest(captured[0]['messages']),
                    state_unchanged=True, native_calls=receipt['native_calls']))
                if not a.legacy_route:
                    assert matches and logged
    invalid = None
    if not a.legacy_route:
        module = SimpleNamespace(driver_loop=original_driver)
        try:
            with furniture_driver(module, policy=None, budget=budget, allowed_tools=('x',),
                                  load_images=None, acceptance=None, evidence_format='typo'):
                raise AssertionError('Unknown format accepted')
        except ValueError:
            invalid = 'rejected before replacing native driver'
        assert module.driver_loop is original_driver
    assert hashes == {p: sha(p) for p in hashes}
    summary = dict(status='legacy_gap_reproduced' if a.legacy_route else 'passed',
        components=len(source), units=len({x['unit_id'] for x in source}),
        wrapper_runs=len(rows), mismatches=sum(not r['messages_match'] for r in rows),
        missing_format_receipts=sum(not r['format_logged'] for r in rows),
        delegated_stages=delegated, invalid_format=invalid, source_hashes_unchanged=hashes,
        seconds=time.monotonic()-started, model_calls=0, render_calls=0,
        native_tool_calls=0, training_updates=0, full_generation_runs=0,
        scope='Actual furniture_driver + run_phase_stage on archived observations; recording STOP policy only. No image decode, model, SDK, rendering or generation.',
        rows=rows)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','source_hashes_unchanged')}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--material', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--legacy-route', action='store_true')
    ap.add_argument('--legacy-host-source', type=Path)
    ap.add_argument('--bindings', type=Path)
    asyncio.run(check(ap.parse_args()))
