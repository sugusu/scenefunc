"""Write the bounded CPU result and a source-host image export manifest."""
import csv
import html
import json
from pathlib import Path

from audit_material_routing import digest, write


def main():
    root = Path('runs/autodl/20261110_multimodal_material')
    availability = json.loads((root / 'source_image_lookup/image_availability.json').read_text())
    result = json.loads((root / 'encoded/encoding.json').read_text())
    missing = {}
    for row in availability['rows']:
        for image in row['images']:
            if image['match']:
                continue
            item = missing.setdefault(image['sha256'], {'sha256': image['sha256'], 'source_path': image['path'], 'required_by': []})
            item['required_by'].append(row['unit_id'])
    write(root / 'REQUIRED_REMOTE_EXPORT.json', {
        'pin': availability['pin'], 'corpus_sha256': availability['corpus_sha256'],
        'scope': 'Exact original observation images, not model weights or raw scene assets.',
        'search_limit': availability['scope'], 'images': list(missing.values()),
        'no_claim_images_absent_from_remote_runtime': True,
        'no_substitution_or_reduced_training_set': True})
    summary = []
    for unit in availability['rows']:
        rec = [r for r in result['records'] if r['unit_id'] == unit['unit_id']]
        summary.append({'unit_id': unit['unit_id'], 'family_id': unit['family_id'],
                        'exact_views': sum(i['match'] is not None for i in unit['images']),
                        'required_views': 5, 'components': len(rec),
                        'encoded_components': sum(r['status'] == 'encoded' for r in rec),
                        'status': ';'.join(sorted({r['status'] for r in rec}))})
    with (root / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    usable = [r for r in result['records'] if r['status'] == 'encoded']
    historical = result['archived_output_prefix_audit']
    prefix_counts = {s: sum(r['status'] == s for r in historical) for s in sorted({r['status'] for r in historical})}
    write(root / 'continuation_decision.json', {
        'decision': 'Continue a bounded conditional-decision study, not an expanded training or claimed novel method.',
        'new_result': 'Actual original-image multimodal encoding and exact archived tensor comparisons, with fixed full-corpus missingness retained.',
        'all_fixed_material_encoded': result['all_fixed_material_encoded'],
        'training_ready': False, 'gpu_requested': False,
        'reasons_not_ready': ['Source-host image export remains incomplete locally.',
                              'No training or deployment inference under the new task-routed protocol has been run.',
                              'Existing four-start Waterloo rollouts do not establish held-out scene improvement.'],
        'positive_research_basis': 'Real activity-conditioned scene decisions and stop/edit failures are observable, and causal inputs/targets can be isolated for a small falsification study.',
        'strong_controls': ['frozen base and explicit task prompt',
                            'ordinary bounded geometry/search with the same observations/actions',
                            'task-routed separate rationale/action multitask learning with identical target material',
                            'paired rationale-to-action with generated, never gold, deployment rationale'],
        'method_claim_boundary': 'Task routing, image-path remapping and mask correction are engineering controls. Rationale multitask learning already has direct prior art.',
        'next_independent_evidence': 'Real improvement of unmet activity requirements, retention of satisfied requirements, and total cost on independent room families; no image substitution or injected failures.',
        'not_a_goal_completion': True})
    unit_rows = ''.join('<tr>' + ''.join('<td>' + html.escape(str(row[k])) + '</td>' for k in ('unit_id','exact_views','encoded_components','components','status')) + '</tr>' for row in summary)
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Git 进度复核与真实多模态材料检查</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 22px;color:#17212b}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ccd5de;padding:8px;text-align:left}}code{{background:#eef2f5;padding:2px 5px}}.note{{padding:15px;background:#eef5fb}}a{{color:#175a99}}</style>
<h1>值得继续有限验证；还不值得扩大训练或宣布方法成立</h1>
<p>2026-11-10。本轮读取并固定 runs/920 的 <code>{availability['pin']}</code>，以及 runs/autodl 的 a1e4cd4。没有新增模型调用、参数更新或场景生成。旧结果保持只读。</p>
<p class="note">Git 已积累真实场景、实际模型轨迹和可核验的对照，但目前没有稳定的独立 Agent 收益证据。此次新增的是可执行多模态材料检查，不能算新的场景生成成绩或论文创新。</p>
<h2>已有证据支持什么</h2>
<ul><li>Waterloo 的四个已见起点上，较早 EVALFIX 的限定活动代理通过数为基座 3/4、action-only 2/4、decision-action 2/4，三者均没有达到“改善且不新增损坏”的案例。后续 MATCHEDROLL 的两个缺陷起点也均未修好；已满足起点上的停止行为有所不同。这些不是独立房间的泛化结果。</li>
<li>最近公开训练快照只有 5 次 forward、5 次 backward、0 次 optimizer update；不能当作这项训练已完成或有效，也不能据此推断远端当前进程状态。</li>
<li>真实网格上的 21 个双任务候选，12 个存在“分别能做、同时不能做”的冲突；普通精确几何判定与枚举 21/21 一致。因此该局部现象不要求新学习器才能解决。</li>
<li>拟议 separate 材料有 8 处 D/A 共用同一前缀、首目标不同的输出路由混杂。它不曾实际训练，因此不是旧负面结果的解释。任务标签是建立强对照所需的普通修正，不是贡献。</li></ul>
<h2>本轮实际跑了什么</h2>
<p>先有界读取公开图像，再完成原始来源运行的 134 条图像路径核查（132 个唯一 Git blob）；文件内容以训练清单 SHA256 为准。共定位 {availability['unique_exact_matches']}/{availability['unique_required_images']} 张唯一原图；{availability['complete_units']}/24 个样本的五视图齐全。其余 19 个样本的原图可能仍存在源主机，不能说数据集丢失。未扫描所有其他运行中的重复图片，也没有用重渲染或相似预览替代。</p>
<p>本机实际使用 Qwen3VLProcessor、冻结的图像预算函数和 65,536 像素预算。成功编码 {result['encoded_components']}/64 个固定监督组件，涉及 {result['encoded_units']} 个完整样本。每个成功组件的 pixel_values 与 image_grid_thw 均和原编码记录哈希完全相同；目标 token 也保持原值。所有提示、图像 token 和 paired 动作输入中的说明前缀均被掩码，保存后重载张量一致。失败或缺图的组件继续计入 64 的分母，没有改成小子集训练。</p>
<p>实际编码程序耗时 {result['wall_seconds']:.2f} 秒，峰值 RSS {result['peak_rss_kib']/1024:.1f} MiB；此耗时包含读取归档输出和初始化，不是模型推理速度。CUDA 初始化：{result['cuda_initialized']}。两阶段图像检索分别耗时 253.58 和 178.10 秒，合计读取约 175.63 MiB Git 图像内容；这不是网络传输量。张量缓存保存在本机 work/，Git 保存可重建代码、路径、内容哈希和结果。</p>
<p>还核对了 8 条既有 D 目标的结束边界：目标在 <code>&lt;/think&gt;</code> 后有一个换行。若未来推理在闭合标记处停止，宿主需补回这一序列化分隔符；不能补写语义内容或用标准答案替代生成结果。35 条旧模型全文的边界审计为 {html.escape(json.dumps(prefix_counts, ensure_ascii=False))}。这只是旧输出的格式适配检查，不是按新协议运行过 35 次，更不能拼接旧的后续响应当新轨迹。</p>
<table><tr><th>固定样本</th><th>找到的原视图 / 5</th><th>成功组件</th><th>组件总数</th><th>状态</th></tr>{unit_rows}</table>
<h2>下一步应检验的具体差异</h2>
<p>共同输入 I 是当前五视图、活动要求与真实工具记录。D 是拟议决定，A 是动作或停止。两组都学习同一批 D 与 A，且任务前缀明确区分；separate 学 D|I 和 A|I，paired 学 D|I 和 A|I,D。训练时 D 可用参考文本，部署时 paired 必须使用实际生成 D，不能使用参考答案。比较须同时报告额外说明生成与读取成本；“先生成再丢弃”的 separate 可以用于匹配预算，但更便宜的直接动作基线也应保留。</p>
<p>这是一项可否定的机制实验，不能凭写出该分解声称新算法。普通 rationale 多任务学习已有 Distilling Step-by-Step 等直接前作；Scene-Designer 已有说明监督与工具反馈，SceneTeract 已有功能性标注和学习。必须证明条件与真实空间的交互带来普通提示、搜索、已有学习机制不能解释的完整任务收益。</p>
<p>若普通基线同样好，就降低这项机制投入；若只让说明更像标准答案而场景无改善，也不能扩大。只有在独立房间里改善实际未完成要求、保住已满足要求且成本合理，才考虑扩大。当前 24 个材料单元和 6 个来源族只够开发，不能代替新房间评测。</p>
<h2>当前还缺什么，是否开资源</h2>
<p><strong>本轮不请求开 GPU。</strong>先在源主机按 REQUIRED_REMOTE_EXPORT.json 导出缺失的 {len(missing)} 张哈希锁定原图，并在完整材料上执行同一编码器；再核验新的两阶段部署与完整训练消费路径。当前 CPU 编码通过的是部分材料，未运行新训练器、模型 forward/backward 或完整部署。本文不宣称全部准备完成。</p>
<p>远端导出命令（尚未执行）：<code>python scripts/autodl/export_missing_training_views.py --manifest runs/autodl/20261110_multimodal_material/REQUIRED_REMOTE_EXPORT.json --output /tmp/scenefunc_missing_views.tar --check-only</code>。确认后去掉 check-only 才创建归档；不读取模型权重、不生成替代图、不启动付费任务。</p>
<h2>可核验证据</h2><ul>
<li><a href="encoded/encoding.json">实际编码、监督掩码、张量哈希与计时</a>；<a href="summary.csv">完整 24 样本审阅表</a></li>
<li><a href="source_image_lookup/image_availability.json">原始五视图可用性</a>；<a href="REQUIRED_REMOTE_EXPORT.json">缺失原图清单</a></li>
<li><a href="../20261110_material_routing/stage_report.html">前一轮输出任务路由审计</a>；<a href="../20261110_objective_source_review/research_review.html">学习结果与论文阅读边界</a>；<a href="../20261110_joint_representation/stage_report.html">真实网格联合用途与普通算法对照</a></li></ul>
</html>'''
    (root / 'stage_report.html').write_text(text)
    source_files = [Path('scripts/autodl') / name for name in (
        'resolve_training_views.py','routed_multimodal.py','encode_routed_material.py',
        'export_missing_training_views.py','report_multimodal_material.py','verify_multimodal_material.py')]
    source_files += [Path('runs/autodl/20261110_material_routing/actual/prospective_components.jsonl'),
                     Path('runs/autodl/20261110_objective_source_review/material_manifest.json')]
    sources = {str(p): digest(p.read_bytes()) for p in source_files}
    output_files = {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json','verification_receipt.json')}
    write(root / 'manifest.json', {'source_sha256': sources, 'output_sha256': output_files})
    print(json.dumps({'encoded_components':len(usable),'missing_unique_images':len(missing),'new_model_calls':0,'report':str(root / 'stage_report.html')}))


if __name__ == '__main__':
    main()
