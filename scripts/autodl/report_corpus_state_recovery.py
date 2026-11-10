"""Record recovered source assets/states separately from missing observations.

The availability cohort is described, never silently substituted for the
24-unit training corpus. No model, renderer, training or new label is used.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import subprocess


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); root = a.root
    states = read(root/'state_audit_v2/units.json')
    state_summary = read(root/'state_audit_v2/summary.json')
    assets = read(root/'asset_recovery_v4/asset_checks.json')
    asset_summary = read(root/'asset_recovery_v4/summary.json')
    availability = read(root/'family_images_v1/result.json')
    dependency = read(root/'dependency/receipt.json')
    wheel = root/'dependency/trimesh-4.7.4-py3-none-any.whl'
    assert sha(wheel) == dependency['wheel_sha256']
    source_verification = []
    for r in read(root/'source_manifest.json')['files']:
        p = root/'source'/Path(r['path']).name
        assert sha(p) == r['sha256']
        raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', PIN+':'+r['path']])
        assert hashlib.sha256(raw).hexdigest() == r['sha256']
        source_verification.append(dict(r, local_path=str(p), unchanged=True))
    recovered = []
    for asset in assets:
        for record in asset['checks']:
            p = root/'asset_recovery_v4/canonical'/asset['asset_id']/Path(record['relative']).name
            assert sha(p) == record['expected_sha256'] == record['actual_sha256']
            recovered.append({'manifest_relative_path': record['relative'],
                              'local_path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size})
    for state in states:
        p = Path(state['local_state'])
        value = read(p)
        canonical = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False).encode()).hexdigest()
        assert canonical == state['state_canonical_sha256']
        assert state['status'] == 'resolved' and not state['context_mismatches']
    assert len(states) == state_summary['resolved_states'] == 15
    assert len(recovered) == 9 and asset_summary['all_geometry_and_material_files_exact']

    material = Path('runs/autodl/20261110_observation_alignment/material_v2/components.jsonl')
    components = [json.loads(line) for line in material.read_text().splitlines()]
    paired = [c for c in components if c['arm'] == 'paired_phase_v3']
    has_d = {c['unit_id'] for c in paired if c['stage'] == 'decision'}
    image_units = {u['unit_id']: u for u in availability['material']}
    cohorts = []
    for c in paired:
        if c['stage'] != 'action':
            continue
        ident = c['unit_id']; image_unit = image_units[ident]
        cohorts.append({'unit_id': ident, 'family_id': c['family_id'],
                        'five_views_available': image_unit['complete_five_views'],
                        'has_supervised_decision': ident in has_d,
                        'training_action_reads_teacher_decision': bool(c['assistant_prefix']),
                        'proposed_paired_deployment_reads_generated_decision': True,
                        'target_kind': c['target_kind']})
    complete = [r for r in cohorts if r['five_views_available']]
    cohort = {
        'source_units': len(cohorts), 'complete_view_units': len(complete),
        'complete_view_families': dict(Counter(r['family_id'] for r in complete)),
        'complete_view_units_with_D': sum(r['has_supervised_decision'] for r in complete),
        'complete_view_units_without_D': sum(not r['has_supervised_decision'] for r in complete),
        'complete_view_components_per_arm': len(complete)+sum(r['has_supervised_decision'] for r in complete),
        'rows': cohorts, 'material_sha256': sha(material),
        'selection_rule': 'Existing five-view availability only; no outcome filtering.',
        'training_subset_created': False,
        'interpretation': 'A possible explicitly declared development smoke cohort, not a replacement training set or a new independent test.',
        'no_test_time_gold_routing': True,
        'missing_D_policy': 'Retain unannotated action supervision. Always-generated-D deployment is a declared distribution change; never route by whether an evaluation teacher label exists.',
        'scientific_next_step': 'Preserve equal original target mass across auxiliary and reason-conditioned arms; measure own-generated decisions through real scene execution. No further latent loss is selected here.'}
    write(root/'availability_cohort.json', cohort)
    recovery = {'pin': PIN, 'exact_asset_files': recovered,
                'state_records': [{'unit_id': r['unit_id'], 'state_path': r['local_state'],
                                   'state_canonical_sha256': r['state_canonical_sha256'],
                                   'proof': r['state_proof'],
                                   'geometry_dependencies': r['geometry_dependencies'],
                                   'room_sdf_dependency': r['room_sdf_dependency']}
                                  for r in states],
                'exact_assets_scope': 'The three assets in the Paper reconstruction; not all assets in fifteen states.',
                'images_exporter': 'scripts/autodl/export_missing_training_views.py',
                'native_bundle_restored': False, 'missing_original_views_recovered': 0,
                'fresh_render_equals_original': False}
    write(root/'recovery_manifest.json', recovery)
    summary = {
        'previous_goal_turn': 'no_progress: Git status review; no new implementation or experiment result',
        'this_goal_turn': 'progress: all fifteen frozen states resolved and three asset closures reproduced byte-exactly',
        'pin': PIN, 'resolved_states': 15, 'state_object_records': state_summary['state_object_records'],
        'context_fields_compared': sum(r['context_fields_compared'] for r in states),
        'context_mismatches': 0, 'exact_assets': 3, 'exact_asset_files': 9,
        'asset_reconstruction_seconds': asset_summary['seconds'],
        'asset_peak_rss_kib': asset_summary['peak_rss_kib'],
        'state_audit_seconds': state_summary['seconds'],
        'original_sources_verified_unchanged': len(source_verification),
        'complete_original_or_equivalent_view_units': len(complete), 'source_units': len(cohorts),
        'failed_asset_variants_preserved_as_receipts': ['asset_recovery_v1', 'asset_recovery_v2', 'asset_recovery_v3'],
        'new_model_calls': 0, 'optimizer_updates': 0, 'new_scenes': 0, 'new_renders': 0,
        'native_bundle_restored': False, 'training_ready': False, 'gpu_requested': False,
        'method_gain_established': False, 'full_goal_complete': False}
    write(root/'summary.json', summary)
    write(root/'verification.json', {'sources': source_verification, 'asset_files_verified': 9,
                                   'states_canonical_verified': 15,
                                   'not_verified': ['native resource closure', 'missing original image pixels',
                                                    'free-model rollout', 'training or full-scene quality']})
    rows = ''.join('<tr><td>'+html.escape(s['unit_id'])+'</td><td>'+str(s['object_count'])+
                   '</td><td>'+html.escape(s['state_proof'])+'</td></tr>' for s in states)
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>原始状态与资产恢复：实际结果</title>
<style>body{{max-width:1040px;margin:40px auto;padding:0 20px;font:17px/1.65 system-ui;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #bbb;padding:8px}}code{{overflow-wrap:anywhere}}</style>
<h1>15 个状态已核对，3 个资产完成精确恢复</h1>
<p>这轮修复的是下一次实验的输入可用性。没有训练、模型调用、新渲染或新场景生成，不能据此宣称算法收益。</p>
<p>15 个缺失视图单元均找到唯一来源状态，共 {summary['state_object_records']} 条对象记录、
{summary['context_fields_compared']} 项可见字段比较，无不一致。证明强度见下表；通过历史哈希链接的记录与直接五视图绑定分别注明。</p>
<p>采用原构建函数和隔离的 trimesh 4.7.4 后，沙发、移动桌、凳子三个资产的 canonical GLB、glTF 描述及缓冲区共 9 个文件，与冻结清单逐字节哈希一致。
前面三次非精确结果保留。4.5.3 加载器会添加文件路径元数据，4.7.4 的本次原始加载结果为空；人为补回路径会改变归档。
最终成功运行没有改源函数或补写路径元数据。此次实际用时 {summary['asset_reconstruction_seconds']:.2f} 秒，峰值约 {summary['asset_peak_rss_kib']/1024:.1f} MiB。</p>
<p>此前追加检查的 60 个同来源图片文件没有新增精确匹配。仍只有 9/24 个单元的五视图齐全；状态和部分几何相同不能代替原始视图。
所有目标状态的完整原生资源包尚未恢复，本文不把局部资产闭合等同于整场可运行。</p>
<h2>对下一轮方法比较的影响</h2>
<p>现有完整视图单元中，{cohort['complete_view_units_with_D']} 个有决策说明监督，{cohort['complete_view_units_without_D']} 个没有，
覆盖 {len(cohort['complete_view_families'])} 个来源家族。可用于明确标注的运行诊断，不能悄悄替代原 24 单元训练或宣称泛化。</p>
<p>候选机制仍是：相同场景、知识和监督下，动作读取模型自己生成的设计判断是否有额外执行收益。
辅助说明监督与说明条件动作应保留同一目标权重；没有说明标签的步骤仍保留动作监督。部署不能根据测试标签是否存在决定走哪条路径。
本轮没有选用新的潜变量损失，也没有证明缺少说明标签必然导致失败。</p>
<p>精确原图恢复与统一新渲染必须成为两种明确的数据版本。若后续改用新渲染，所有方法共用同一版本，历史模型输出不能接在改变后的观察后面当作真实新轨迹。</p>
<table><tr><th>单元</th><th>对象记录</th><th>状态来源证据</th></tr>{rows}</table>
<h2>可核查文件</h2><ul>
<li><a href="recovery_manifest.json">状态与资产恢复清单</a></li>
<li><a href="availability_cohort.json">24 单元可用性与监督覆盖</a></li>
<li><a href="asset_recovery_v4/asset_checks.json">9 个文件的原始与实际哈希</a></li>
<li><a href="state_audit_v2/units.json">逐项状态和字段核对</a></li>
<li><a href="reading_ledger.json">源码实际阅读范围</a></li>
<li><a href="COMMANDS.txt">执行命令</a></li>
<li><a href="temporary_cleanup.json">仅本轮临时二进制清理</a></li>
<li><a href="summary.json">结果摘要</a> · <a href="verification.json">核验范围</a></li></ul>
<p>许可资产保存在本地，不随公开 Git 提交。旧实验和原始资产只读。全场功能、语义、物理与独立场景方法收益均未在本轮验证。</p></html>'''
    (root/'stage_report.html').write_text(report)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
