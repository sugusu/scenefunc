"""Exercise the original SFT reader on diagnostic-only partial trajectories.

No processor, tokenizer, neural model, optimizer or Trainer is constructed.
Fixtures stay in RAM and are explicitly NOT admitted training examples.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

from audit_recorded_trajectory_compression import (
    ARMS, PILOT, SOURCE, START, dump, read, sha, source_history,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ram-output', type=Path, required=True)
    args = ap.parse_args()
    assert not (args.output / 'sft_reader_results.json').exists()
    args.ram_output.mkdir(exist_ok=False)
    path = SOURCE / 'src/sft_dataset.py'
    assert sha(path) == '9e4d18f67e746af16f830d7e8d0da714964a43bfa320ef091a0092b3e9bb2569'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('source_sft_reader', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import memory
    initial = read(START)['payload']
    rows = []
    prefix = memory.empty_memory('furniture')
    for event in initial['tool_history']:
        call = event['call']
        memory.update_furniture(prefix, call['name'], call['arguments'], event['response'])
    # This is an actual recorded prefix, not a guessed synthetic placement.
    prefix_report = {'recorded_prefix_calls': len(initial['tool_history']),
                     'scene_instances': len(initial['placed_instances']),
                     'memory_assets': len(prefix['assets']),
                     'memory_placed_assets': sum(bool(x.get('placed')) for x in prefix['assets'].values()),
                     'prefix_rendered_memory': memory.render_memory('furniture', prefix),
                     'not_injected_into_loader': True}
    for arm in ARMS:
        episode = read(args.output / f'{arm}_projected_episode.json')
        request = read(PILOT / arm / 'turn_00/request.json')
        episode['system'] = next(m['content'] for m in request if m['role'] == 'system')
        episode['task_context'] = {'scene_description': read(PILOT / 'PAIR_PROTOCOL.json')['task']['brief']}
        episode['training_admitted'] = False
        episode['reason'] = 'Partial failed development trace; missing construction prefix and original four-stage evidence'
        for turn in episode['turns']:
            if turn.get('role') == 'assistant':
                raw = turn['content']
                match = re.search(r'<think>\s*(.*?)\s*</think>', raw, flags=re.S)
                # Six saved decisions are tool-only. Absence is not permission
                # to manufacture a rationale or reject otherwise valid calls.
                assert match or '<think>' not in raw, 'Malformed captured reasoning; preserve failure'
                turn['content'] = match.group(1) if match else ''
        fixture = args.ram_output / f'{arm}_diagnostic_not_for_training.json'
        dump(fixture, episode)
        samples = module.build_step_samples(fixture, SOURCE)
        sample_rows = []
        for i, sample in enumerate(samples):
            default, _ = module._build_messages(sample, no_thinking=False)
            actions_only, _ = module._build_messages(sample, no_thinking=True)
            assert default[-1]['tool_calls'] == actions_only[-1]['tool_calls']
            assert len(sample.scene_images) == 5
            sample_rows.append({'step_idx': sample.step_idx,
                'scene_images': len(sample.scene_images),
                'memory_str': sample.memory_str,
                'captured_original_reasoning_chars': len(sample.assistant_content),
                'default_target_contains_original_reasoning': sample.assistant_content in default[-1]['content'],
                'explicit_actions_only_target_content': actions_only[-1]['content'],
                'tool_calls': sample.tool_calls,
                'source_original_tool_success': source_history(arm)[1][i]['feedback'][0]['response']['success']})
        assert len(samples) == 4
        assert samples[0].memory_str == '# Memory snapshot (empty)'
        rows.append({'arm': arm, 'structurally_built_step_samples': len(samples),
                     'quality_gate_in_reader': False,
                     'training_admitted_by_this_audit': False,
                     'fixture_sha256': sha(fixture),
                     'fixtures_persisted_in_git': False,
                     'first_step_inventory_context_missing': True,
                     'sample_rows': sample_rows})
    dump(args.output / 'sft_reader_results.json', {
        'scope': 'Original reader/message builder executed; no tokenization or gradient computation',
        'model_constructions': 0, 'training_steps': 0, 'new_API_calls': 0,
        'original_reader_sha256': sha(path),
        'original_memory_sha256': sha(SOURCE / 'src/memory.py'),
        'interpretation': 'The loader is structural, not a whole-room quality validator. This does not show bad samples occur in OTA-35K. Our four-step excerpts omit their real construction prefix and must not be trained as standalone episodes.',
        'recorded_prefix_recovery': prefix_report, 'rows': rows})
    print(json.dumps({'loader_step_samples': sum(x['structurally_built_step_samples'] for x in rows),
                      'training_admitted': 0, 'training_steps': 0}))


if __name__ == '__main__':
    main()
