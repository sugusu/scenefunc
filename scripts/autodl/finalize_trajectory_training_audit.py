"""Package actual source/trajectory findings without promoting them to successes."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

from audit_recorded_trajectory_compression import R, SOURCE, W, dump, read, sha


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    o = a.output.resolve()
    result = read(o / 'replay_results.json')
    assert result['status'] == 'completed' and result['native_actions'] == 24
    assert len(result['rows']) == 8
    assert sum(r['world_and_response_reusable'] for r in result['rows']) == 6
    assert all(r['final_world_preserved_exactly'] and not r['training_admitted'] for r in result['rows'])
    assert read(o / 'sft_reader_v2_receipt.json')['exit_code'] == 0
    sft = read(o / 'sft_reader_results.json')
    assert sum(x['structurally_built_step_samples'] for x in sft['rows']) == 8
    assert not any(x['training_admitted_by_this_audit'] for x in sft['rows'])
    assert sft['recorded_prefix_recovery']['memory_placed_assets'] == 9
    ordinary = read(o / 'ordinary_selector_results.json')['rows']
    assert all(x['ordinary_rule_matches_minimum_replay_steps'] for x in ordinary)
    arithmetic = read(o / 'official_local_arithmetic_results.json')['rows']
    assert len(arithmetic) == 8
    assert sum(r['status'] == 'calculated' for r in arithmetic) == 3
    before = read(o / 'protected_sources_before.json')
    protected = [{'path': x['path'], 'before': x['before'], 'after': sha(x['path'])} for x in before]
    assert len(protected) == 579 and all(x['before'] == x['after'] for x in protected)
    dump(o / 'protected_sources_before_after.json', protected)
    originals = read(o / 'source_input_hashes.json')
    assert all(sha(x['path']) == x['sha256'] for x in originals)
    manifest = read(o / 'public_source_manifest.json')
    # Root-project license absence is separate from the bundled upstream notice.
    manifest.pop('license_file_found_in_recursive_tree', None)
    tree = read(SOURCE / 'tree.json')['tree']
    manifest['root_project_license_found'] = any(x['path'].lower() in ('license', 'license.md', 'license.txt', 'copying') for x in tree)
    manifest['third_party_scenesmith_license_notice_listed'] = any(x['path'] == 'integration/SceneSmith-LICENSE' for x in tree)
    for rel in ('src/memory.py', 'src/scene_observation.py', 'trajectory_compression/offline_scene.py'):
        p = SOURCE / rel
        manifest['files'].append({'path': str(p), 'relative_path': rel,
            'bytes': p.stat().st_size, 'sha256': sha(p),
            'url': f'https://github.com/ZJU-OmniAI/Scene-Designer/blob/{manifest["pin"]}/{rel}'})
    assert all(sha(x['path']) == x['sha256'] for x in manifest['files'])
    dump(o / 'public_source_manifest.json', manifest)
    reading = []
    full = {'trajectory_compression/candidates.py', 'trajectory_compression/engine.py',
            'trajectory_compression/quality.py', 'trajectory_compression/review.py',
            'trajectory_compression/offline_check.py', 'trajectory_compression/offline_evidence.py',
            'trajectory_compression/offline_scene.py', 'trajectory_compression/model.py',
            'src/train_sft.py', 'src/sft_dataset.py', 'src/scene_observation.py', 'USAGE.md', 'VALIDATION.md'}
    for x in manifest['files']:
        reading.append({**x, 'coverage': 'Full text read' if x['relative_path'] in full else
            'Targeted room update368–595, renderer621–771, incremental memory772–794; helper/track ranges previously reviewed. Not full995-line module.',
            'execution': ('candidates/deletable_turns unchanged; eight native deletion replays are our audit' if x['relative_path'].endswith('candidates.py') else
                          'validate_input extracted unchanged; full Compressor NOT executed' if x['relative_path'].endswith('engine.py') else
                          'snapshot_state/calculate unchanged on eight saved transitions' if x['relative_path'].endswith('offline_scene.py') else
                          'build_step_samples and _build_messages unchanged; eight samples, no tokenizer/weights' if x['relative_path'].endswith('sft_dataset.py') else
                          'Real prefix replay into update_furniture; memory_before_turns via original reader' if x['relative_path'].endswith('memory.py') else
                          'Imported transitively' if x['relative_path'].endswith(('offline_check.py', 'model.py', 'scene_observation.py')) else
                          'Read only; no API/model/training calls')})
    book = Path('/root/autodl-tmp/research_agent3d_20261013/book_pages.json')
    dump(o / 'reading_ledger.json', {'public_source_pin': manifest['pin'], 'entries': reading,
        'book': {'source': str(book), 'sha256': sha(book),
                 'new_since_01b8d09': [[190, 222]],
                 'current_direct_text_coverage': [[1, 8], [110, 312]],
                 'current_direct_text_pages': 211, 'total_pages': 312,
                 'still_unread_in_current_direct_pass': [[9, 109]],
                 'coverage_note': '190–201 read during the preceding unarchived training review; 202–222 read this continuation in seven three-page groups. Chapter7 extracted text now complete. Earlier historical full-text ledger is separate.',
                 'not_claimed': ['Fresh full-book reread', 'Visual figure inspection', 'Book benchmark reruns', 'Verification of model/product versions or numeric examples'],
                 'applied': ['Mechanism metrics versus task outcomes', 'Failure prefixes as acceptable action sets', 'Matched task/state and costs', 'Reset and observation provenance'],
                 'cautions': ['Different model families do not ensure independent judge errors', 'Deterministic geometry tools can report causes, not just pass/fail', 'Small illustrative benchmark results do not prove full-system performance', 'Rubric reuse as a reward still needs coverage and anti-shortcut checks']},
        'papers': [
            {'url': 'https://arxiv.org/html/2312.06585v2', 'title': 'Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models', 'coverage': 'Sections2–3 including EM and algorithm1, section4 related work; not a full experimental reread', 'limit': 'No self-training run and no monotonic improvement claim for our finite data'},
            {'url': 'https://arxiv.org/html/2203.14465v2', 'title': 'STaR: Bootstrapping Reasoning With Reasoning', 'coverage': 'Sections3.1–3.2 and algorithm1; not all experiments/appendices', 'limit': 'Correct final answer filtering does not independently prove faithful reasoning'},
            {'url': 'https://aclanthology.org/2024.emnlp-main.861.pdf', 'title': 'Re-ReST: Reflection-Reinforced Self-Training for Language Agents', 'coverage': 'Sections2.1–2.3, PDF pages2–4; not all experiments/appendices', 'limit': 'Feedback-based refinement and ordinary MLE are existing methods, not proposed here'}],
        'original_OTA_35K': {'downloaded': False, 'trajectory_level_census': False,
                            'negative_inference_forbidden': 'No public raw corpus located in inspected release is not a claim that no author data exist.'}})
    csv_write(o / 'summary.csv', ordinary)
    csv_write(o / 'replay_matrix.csv', [{k: v for k, v in x.items() if k in (
        'arm', 'candidate_index', 'retained_actions', 'final_world_preserved_exactly',
        'all_retained_input_worlds_preserved', 'all_retained_tool_responses_preserved',
        'world_and_response_reusable', 'training_admitted')} | {
            'dropped_original_model_turns': json.dumps(x['dropped_original_model_turns'])}
        for x in result['rows']])
    csv_write(o / 'sft_step_audit.csv', [{
        'arm': x['arm'], 'step_idx': s['step_idx'], 'input_view_count': s['scene_images'],
        'captured_reasoning_present': bool(s['captured_original_reasoning_chars']),
        'original_tool_success': s['source_original_tool_success'],
        'structurally_loadable': True, 'whole_room_training_admitted': False}
        for x in sft['rows'] for s in x['sample_rows']])
    dump(o / 'result_field_clarifications.json', {
        'old_reasoning_admitted': 'Legacy replay-row name means unchanged original noncompressed reasoning only. It does NOT override training_admitted=false for every row.',
        'default_target_contains_original_reasoning': 'For empty original content the initial adapter uses Python empty-substring inclusion. Use captured_original_reasoning_chars>0 (sft_step_audit.csv) to determine actual presence; two have reasoning and six do not.',
        'world_and_response_reusable': 'Exact checked world projection and tool response only; not equality of complete policy history, pixels, physical state or training readiness.',
        'not_found': 'No solution in the original complete finite candidate pool; continuous-space feasibility unknown.'})
    cleanup = read(o / 'verified_redundant_archive_cleanup.json')
    assert not Path(cleanup['removed_path']).exists()
    assert all(sha(x['retained_path']) == x['sha256'] for x in cleanup['members'])
    summary = {'development_rooms': 1, 'previous_neural_trajectories': 2,
        'new_neural_calls': 0, 'new_API_calls': 0, 'training_updates': 0,
        'native_move_calls': 24, 'native_checkpoint_restores': 8,
        'deletion_subsets': 8, 'exact_endpoints_preserved': 8,
        'retained_world_and_response_preserved': 6,
        'original_local_arithmetic_calls': 8, 'original_arithmetic_calculated': 3,
        'structurally_readable_sft_steps': 8, 'new_whole_room_positive_examples': 0,
        'new_scenes_generated': 0, 'whole_room_success_demonstrated': False,
        'native_replay_elapsed_seconds': result['elapsed_seconds'],
        'corrected_sft_reader_elapsed_seconds': read(o / 'sft_reader_v2_receipt.json')['elapsed_seconds'],
        'timing_limit': 'No efficiency claim. Totals exclude reading/writing/packaging, initial failed adapter and missing time-wrapper overhead; failures separately retained.',
        'protected_files_unchanged': len(protected),
        'deleted_redundant_archive_bytes': cleanup['removed_bytes'],
        'unique_original_files_deleted': 0,
        'image_portability_limit': '40 original input references,35 distinct files (~36.05 MB) verified in RAM; not duplicated into this package. Native replay needs the original local assets/runtime and these source observations. No portable train-ready dataset claimed.',
        'scope': 'Source reading and exact native edit replays, not full official generation/CTC/physics or learned policy improvement'}
    dump(o / 'summary.json', summary)
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(x[k])) + '</td>' for k in (
        'arm', 'original_steps', 'endpoint_only_selected_steps', 'endpoint_only_feedback_compatible',
        'exact_replay_selector_steps', 'ordinary_original_noop_rule_steps')) + '</tr>' for x in ordinary)
    theory = html.escape((o / 'algorithm_and_theory.txt').read_text())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实际轨迹与训练目标审计</title>
<style>body{{font:17px/1.7 system-ui;max-width:1050px;margin:36px auto;padding:0 24px;color:#17202a}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:8px;border:1px solid #ccd4dd;text-align:left}}pre{{white-space:pre-wrap;font:15px/1.8 system-ui}}.box{{padding:18px;background:#eef4fa;border-left:5px solid #4278ac}}</style>
<h1>压缩可保住错误终点，不能因此获得正确的场景示范</h1>
<p>2026-11-11｜固定来源、两条真实模型轨迹、24次原生移动回放。新增模型/API调用与训练均为0。</p>
<div class="box">8个删减子集均保留最终世界投影，其中6个同时保留每个动作的输入世界和工具反馈。普通 no_movement 规则已解释本例全部安全压缩收益。房间末态仍有6/5对家具网格相交，未证明完整生成成功。</div>
<h2>同权限下的普通对照</h2><p>同一个开发房间的两种已有模型提示条件；不是独立测试集。规则选择是事后解释对照，后续才可冻结检验。</p>
<table><tr><th>旧模型轨迹</th><th>原步数</th><th>仅终点筛选</th><th>所选反馈仍一致</th><th>原生验证最短</th><th>普通无需移动规则</th></tr>{table}</table>
<h2>训练读取器的实际结果</h2><p>原读取器产生8条结构样本，但我们准入的完整房间正例为0。四步摘录省略了真实生成前缀；直接加载首步记忆为空。恢复12条真实前缀事件后可得到9件已放家具。原始OTA-35K未取得，不能据此统计或批评它的坏样本比例。</p>
<p>源数据包含2段原模型解释和6次无解释工具调用；没有补写思考。原适配器误要求每步有think、另有time包装缺失，两项失败均保留，未用于挑选样本。原始SFT目标是普通assistant-token交叉熵；本轮未做优化。</p>
<h2>论文与书的阅读边界</h2><p>公开压缩器已有状态/图像对应、保留反馈校验和删除后解释重标注。ReST-EM、STaR及Re-ReST已有成功过滤或反馈修正训练，不能重新命名为创新。对应方法章节已读，未重跑作者实验。书第七章提取文字已读完，当前直接复读211/312页，仍余9–109页；没有逐页看图或核实书中所有数字。</p>
<p>代码与权重的版本要分开：当前公开使用文档明确说五视图代码更新并未重新训练已发布SFT权重。不能由本地五视图开发失败推断论文配置能力上限，也不能宣称已经复现其训练。</p>
<h2>交付与限制</h2><p><a href="summary.csv">两组对照CSV</a> · <a href="replay_matrix.csv">八次回放</a> · <a href="sft_step_audit.csv">逐步训练审阅</a> · <a href="reading_ledger.json">实际阅读记录</a> · <a href="public_source_manifest.json">固定源码</a> · <a href="evidence.zip">证据包</a></p>
<p>579个受保护输入哈希未变。仅删除一个152个成员均已逐字节验证有原文件的旧重复压缩包，约5.96MB，全部成员原文件保留，清理清单在目录内。40个原输入视图引用在RAM核验，没有复制35个独立图像约36MB；因此本包不是可迁移SFT数据集。大模型、资产和当前运行环境未删除。</p>
<p>本轮核查完成，研究目标仍未完成。尚未证明CVPR主创新，不申请新资源，不把未通过的完整房间改称局部成功。</p>
<h2>完整算法与理论条件</h2><pre>{theory}</pre></html>'''
    (o / 'stage_report.html').write_text(page)
    (o / 'REPRODUCE.txt').write_text(
        'No model/API/training commands are included. Original local assets and runtime required.\n'
        'Read public_source_manifest.json; retrieve selected pinned source text into SOURCE RAM paths if absent. Source files are not redistributed.\n'
        'Use a NEW output directory and NEW RAM directory for any rerun; copy PROTOCOL.json there. Preserve originals.\n'
        'Native command (old run already completed):\n'
        'env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
        f'PYTHONPATH={R}/runs/autodl/20261110_corpus_state_recovery/dependency/import:'
        '/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps '
        '/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python '
        'scripts/autodl/audit_recorded_trajectory_compression.py --output NEW_OUTPUT --ram-output NEW_RAM\n'
        f'{W}/v100_runtime_20261104/runtime/bin/python scripts/autodl/check_trajectory_sft_boundary.py --output NEW_OUTPUT --ram-output NEW_SFT_RAM\n'
        'python scripts/autodl/compare_trajectory_selectors.py --output NEW_OUTPUT\n'
        'The full official Compressor is NOT run. Four-stage evidence must not be fabricated to bypass that boundary.\n')
    scripts = [R / 'scripts/autodl' / name for name in (
        'audit_recorded_trajectory_compression.py', 'check_trajectory_sft_boundary.py',
        'compare_trajectory_selectors.py', 'finalize_trajectory_training_audit.py')]
    payloads = [p for p in sorted(o.rglob('*')) if p.is_file()
                and p.name not in ('evidence.zip', 'artifact_manifest.json', 'delivery_receipt.json')]
    entries = [{'path': str(p.relative_to(R)), 'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in [*payloads, *scripts]]
    dump(o / 'artifact_manifest.json', {'files': entries, 'zip_exclusions': ['evidence.zip', 'delivery_receipt.json'],
        'original_source_files_redistributed': False})
    free = shutil.disk_usage(o).free
    needed_upper = sum(p.stat().st_size for p in payloads) + sum(p.stat().st_size for p in scripts) + 65536
    assert free - needed_upper > 5 * 1024**3, 'Preserve 5GiB data reserve; do not truncate scientific payloads'
    with zipfile.ZipFile(o / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in [*payloads, *scripts, o / 'artifact_manifest.json']:
            z.write(p, str(p.relative_to(R)))
    with zipfile.ZipFile(o / 'evidence.zip') as z:
        assert z.testzip() is None
        for entry in entries:
            assert hashlib.sha256(z.read(entry['path'])).hexdigest() == entry['sha256']
    dump(o / 'delivery_receipt.json', {'status': 'verified', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'manifest_payloads': len(entries), 'zip_sha256': sha(o / 'evidence.zip'),
        'protected_inputs_unchanged': len(protected), 'source_trace_inputs_unchanged': len(originals),
        'full_room_success': False, 'new_model_calls': 0, 'new_API_calls': 0, 'new_training_updates': 0,
        'free_after_bytes': shutil.disk_usage(o).free})
    print(json.dumps({'status': 'verified', 'report': str(o / 'stage_report.html'),
                      'payloads': len(entries), 'free_bytes': shutil.disk_usage(o).free}))


if __name__ == '__main__':
    main()
