"""Host failure accounting; no native mutation, model, renderer or training."""
import argparse
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from check_native_phase_host import ScriptedBackend, Tokenizer, read
from phase_host import NativeRoomPort, StageBudget, digest, run_phase_stage, write
from phase_policy import PhasePolicy


async def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    w = args.work_base.resolve()
    old = w/'paris_local_run_results_20261104'
    state = read(old/'B_mesh_feedback_sdpa_kv/tools/checkpoint.json')['payload']['placed_instances']
    views = read(old/'B_after_feedback_views/receipt.json')['views']
    from PIL import Image
    loaded = [Image.open(v['path']).convert('RGB') for v in views]
    tokenizer = Tokenizer(w/'inference_autodl_20261103/processor_only/tokenizer.json')
    cases = []
    for mode in ('invalid_decision', 'backend_error', 'unadvertised_tool', 'stale_context', 'schema_fault'):
        mutable = {'objects': copy.deepcopy(state)}
        env = SimpleNamespace(stage='furniture', scene=SimpleNamespace(to_state_dict=lambda: mutable),
                              mem={}, tool_map={'permitted': SimpleNamespace(params_json_schema={
                                  'type': 'object', 'properties': {}, 'additionalProperties': False})})
        async def forbidden(calls):
            raise AssertionError('Failure fixture must not execute a native tool')
        env.step = forbidden
        env.observation = lambda: {'messages': [{'role': 'system', 'content': 'Host failure fixture'}], 'step_idx': 0}
        port = NativeRoomPort(env, allowed_tools=('permitted',), load_images=lambda _: loaded,
                              acceptance=lambda _: {'state_sha256': digest(mutable), 'full_scene_success': None})

        class FailureBackend(ScriptedBackend):
            def generate(self, messages, images, **kw):
                if mode == 'backend_error':
                    raise RuntimeError('Deliberate fixture backend failure')
                if mode == 'invalid_decision':
                    return {'output_ids': self.eos_ids, 'input_tokens': 1}
                output = super().generate(messages, images, **kw)
                if kw['stage'] == 'action':
                    if mode == 'stale_context':
                        env.mem['concurrent_change'] = True
                    if mode == 'schema_fault':
                        # A bad live schema is an infrastructure fault, not a
                        # model's invalid argument. Keep digest stable so the
                        # schema validator is actually reached in this fixture.
                        pass
                return output

        if mode == 'schema_fault':
            env.tool_map['permitted'].params_json_schema = {'type': 'not-a-JSON-schema-type'}
        call = {'name': 'not_advertised' if mode == 'unadvertised_tool' else 'permitted', 'arguments': {}}
        backend = FailureBackend(tokenizer, [call])
        policy = PhasePolicy(backend, arm='paired_phase_v3', total_output_tokens=256,
                             decision_output_tokens=96, action_output_tokens=160, total_seconds=20)
        # Three allowed generation calls cannot buy two D+A turns.
        budget = StageBudget(2 if mode == 'unadvertised_tool' else 1, 3, 2, 512, 60)
        thrown = None
        try:
            result = await run_phase_stage(port=port, policy=policy, budget=budget, output=root/mode)
        except Exception as exc:
            thrown = type(exc).__name__
            result = read(root/mode/'result.json')
        assert result['native_calls'] == 0 and result['full_scene_success'] is None
        assert result['status'] != 'model_stop_proposed'
        expected = {'invalid_decision': ('turn_budget_exhausted', 1),
                    'backend_error': ('backend_failure_accounting_incomplete', 1),
                    'unadvertised_tool': ('cumulative_budget_exhausted', 2),
                    'stale_context': ('stale_observation_no_dispatch', 2),
                    'schema_fault': ('host_failure', 2)}
        assert (result['status'], result['generation_attempts']) == expected[mode]
        if mode == 'schema_fault':
            assert thrown == 'SchemaError'
        cases.append({'case': mode, 'status': result['status'],
                      'generation_attempts': result['generation_attempts'],
                      'tool_attempts': result['tool_attempts'], 'native_calls': 0,
                      'accounting_complete': result['accounting_complete'], 'exception': thrown})
    write(root/'summary.json', {'cases': cases, 'real_model_calls': 0, 'native_calls': 0,
                               'new_scene_generations': 0, 'scope': 'failure fixtures only'})
    print(json.dumps(cases), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-base', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    asyncio.run(run(ap.parse_args()))
