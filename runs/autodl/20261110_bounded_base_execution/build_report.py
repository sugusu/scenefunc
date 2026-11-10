"""CPU-only audit/report of the completed bounded trajectory; no new inference."""
import csv
import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
READY = REPO / 'runs/autodl/20261110_bounded_base_readiness'
WORK = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
RUN = OUT / 'recovery_v1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(name, d):
    (OUT/name).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def pairs(d):
    return {tuple(sorted(x)) for x in d['furniture_surface_intersection_pairs']}


def shape_hash(payload):
    d = {k: payload[k] for k in ['placed_instances', 'frozen_shell']}
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def main():
    result = read(RUN/'result.json')
    parent = read(READY/'model_run_v1/result.json')
    wrapper = read(OUT/'process_receipt.json')
    old_wrapper = read(READY/'model_run_v1_process_receipt.json')
    assert wrapper['exit_code'] == 0 and old_wrapper['exit_code'] == 1
    assert result['cumulative_model_subprocess_attempts'] == 4
    assert result['completed_model_outputs'] == 3
    assert result['cumulative_model_subprocess_seconds'] < 1200
    initial_cp = Path(read(RUN/'configuration.json')['source_checkpoint'])
    initial_payload = read(initial_cp)['payload']
    initial_objects = {x['object_id']: x for x in initial_payload['placed_instances']}
    initial_audit = read(READY/'initial_audit.json')
    before_payload, before_cp, before_views = initial_payload, initial_cp, read(READY/'initial_views_receipt.json')
    rows, evidence, local_media = [], [], []
    first_request = read(READY/'model_run_v1/turn_00/request.json')
    total_model_seconds = parent['model_subprocess_seconds']
    stage_seconds = {'dispatch': 0.0, 'mesh_audit': 0.0, 'render': 0.0}
    for turn in sorted(RUN.glob('turn_*')):
        req = read(turn/'request.json')
        receipt = read(turn/'model/receipt.json')
        process = read(turn/'model_process_receipt.json')
        engine = read(Path(receipt['engine_receipt'])/'receipt.json')
        assert receipt['request_sha256'] == sha(turn/'request.json')
        assert receipt['status'] == 'completed' and engine['generation_started']
        assert engine['ended_with_eos'] and receipt['network_connection_attempts'] == 0
        assert process['exit_code'] == 0 and not process['timeout']
        assert process['elapsed_seconds'] < 200
        obs = json.loads(req[-1]['content'][-1]['text'].split('\nCurrent evidence:\n', 1)[1])
        assert len(obs['previous_attempts']) == len(evidence)
        expected_objects = [{k: o[k] for k in ['object_id', 'name', 'transform'] if k in o} for o in before_payload['placed_instances']]
        assert obs['current_objects'] == expected_objects
        assert obs['mesh_check']['furniture_surface_intersection_pairs'] == (initial_audit if not rows else read(RUN/rows[-1]['turn']/'audit.json'))['furniture_surface_intersection_pairs']
        if not evidence:
            assert req == first_request
        for previous, saved in zip(obs['previous_attempts'], evidence):
            assert previous['raw_model_output'] == saved['raw']
            assert previous['feedback'] == saved['responses']
        images = engine['images']
        assert len(images) == 5
        assert sha(before_cp.parent/'partial_scene.glb') == (before_views.get('source_sha256') or read(READY/'initial_audit.json')['scene_sha256'])
        for actual, expected in zip(images, before_views['views']):
            assert actual['sha256'] == expected['sha256'] == sha(actual['origin'])
        audit = read(turn/'audit.json')
        cp = turn/'tools/checkpoint.json'
        payload = read(cp)['payload']
        current = {x['object_id']: x for x in payload['placed_instances']}
        assert set(current) == set(initial_objects)
        assert payload['frozen_shell'] == initial_payload['frozen_shell']
        changed = [k for k in current if current[k] != initial_objects[k]]
        assert changed == ['dining_chair_0']
        assert all(current[k] == initial_objects[k] for k in current if k != 'dining_chair_0')
        responses = read(turn/'tools/tool_responses.json')
        calls = read(turn/'calls.json')
        assert len(responses) == len(calls) == 1
        assert responses[0]['call'] == calls[0]
        raw = (turn/'model/raw_output.txt').read_text()
        evidence.append({'raw': raw, 'responses': responses})
        total_model_seconds += process['elapsed_seconds']
        for key in stage_seconds:
            cost = read(turn/(key+'_receipt.json'))
            assert cost['exit_code'] == 0 and not cost['timeout']
            stage_seconds[key] += cost['elapsed_seconds']
        row = {'turn': turn.name, 'initial_checkpoint_sha256': sha(initial_cp),
            'request_sha256': sha(turn/'request.json'), 'input_tokens': engine['input_tokens'],
            'output_tokens': engine['output_tokens'], 'real_model_attempts': 1,
            'completed_model_outputs': 1, 'tool_calls': len(calls),
            'tool_success': responses[0]['response']['success'],
            'tool_error': responses[0]['response'].get('error_type'),
            'action': json.dumps(calls[0], ensure_ascii=False, separators=(',', ':')),
            'objects': len(current), 'objects_added': 0, 'objects_removed': 0,
            'changed_objects_vs_initial': len(changed), 'non_target_objects_preserved': 8,
            'intersection_pairs': len(pairs(audit)),
            'new_pairs_vs_initial': sorted(pairs(audit)-pairs(initial_audit)),
            'removed_pairs_vs_initial': sorted(pairs(initial_audit)-pairs(audit)),
            'outside_floor': len(audit['out_of_rectangular_floor_objects']),
            'separate_planar': read(turn/'measurements.json')['results']['scene_section_free']['separate_use']['status'],
            'simultaneous_planar': read(turn/'joint.json')['status'],
            'terminal_checkpoint': str(cp), 'terminal_checkpoint_sha256': sha(cp),
            'scene_glb': str(cp.parent/'partial_scene.glb'), 'scene_sha256': sha(cp.parent/'partial_scene.glb'),
            'geometry_state_sha256': shape_hash(payload),
            'model_seconds': process['elapsed_seconds'],
            'peak_cuda_allocated_bytes': engine['peak_cuda_allocated_bytes'],
            'peak_cuda_reserved_bytes': engine['peak_cuda_reserved_bytes'],
            'model_process_peak_rss_mib': receipt['peak_rss_mib'],
            'whole_room_success': 'unknown'}
        rows.append(row)
        views = read(turn/'views/receipt.json')
        assert views['source_sha256'] == row['scene_sha256']
        for v in views['views']:
            assert sha(v['path']) == v['sha256']
        for q in [cp.parent/'partial_scene.glb']+[Path(v['path']) for v in views['views']]:
            local_media.append({'path': str(q), 'sha256': sha(q), 'bytes': q.stat().st_size})
        before_cp, before_payload, before_views = cp, payload, views
    assert abs(total_model_seconds-result['cumulative_model_subprocess_seconds']) < 1e-6
    assert len({x['request_sha256'] for x in rows}) == 3
    assert len({x['action'] for x in rows}) == 1
    assert len({x['geometry_state_sha256'] for x in rows}) == 1
    original_hashes = read(WORK/'autodl_continuation_20261110/cpu_replay/source_hashes_before.json')
    assert all(sha(path) == h for path, h in original_hashes.items())
    warm = read(RUN/'cache_warmup.json')
    end = datetime.datetime.fromisoformat(wrapper['ended_at_utc'])
    begin = datetime.datetime.fromisoformat(old_wrapper['started_at_utc'])
    costs = {'cold_wrapper_seconds': old_wrapper['elapsed_seconds'],
        'cold_failed_model_subprocess_seconds': parent['model_subprocess_seconds'],
        'cache_warmup_seconds': warm['seconds'], 'recovery_wrapper_seconds': wrapper['elapsed_seconds'],
        'cumulative_model_subprocess_seconds': total_model_seconds, **stage_seconds,
        'pipeline_wrappers_sum_seconds': wrapper['elapsed_seconds']+old_wrapper['elapsed_seconds'],
        'first_launch_to_terminal_calendar_seconds': (end-begin).total_seconds(),
        'calendar_scope': 'Includes intervening implementation/inspection; not all GPU compute. No account billing measured.'}
    ordinary = read(READY/'ordinary_reference_result.json')
    summary = {'status': 'bounded_base_trajectory_executed', 'model_revision': '63e235a0c0d6e5bc2f8f3485214df4457fd2e672',
        'model_subprocess_attempts': 4, 'load_timeouts': 1, 'completed_model_outputs': 3,
        'tool_calls': 3, 'tool_successes': 1, 'no_movement_rejections': 2,
        'initial_furniture_intersection_pairs': len(pairs(initial_audit)),
        'final_furniture_intersection_pairs': rows[-1]['intersection_pairs'],
        'new_pairs': rows[-1]['new_pairs_vs_initial'], 'removed_pairs': rows[-1]['removed_pairs_vs_initial'],
        'original_objects_preserved': 9, 'non_target_objects_preserved': 8,
        'simultaneous_planar': result['condition_B_simultaneous_planar'],
        'unknowns': result['unknowns'], 'costs': costs,
        'peak_cuda_allocated_bytes': max(x['peak_cuda_allocated_bytes'] for x in rows),
        'cuda_peak_scope': 'Completed generation receipts; engine resets CUDA statistics after model loading. Not a cold-load or whole-job VRAM peak.',
        'completed_input_tokens': sum(x['input_tokens'] for x in rows),
        'completed_output_tokens': sum(x['output_tokens'] for x in rows),
        'sampled_peak_container_memory_bytes': wrapper['sampled_peak_cgroup_memory_bytes'],
        'sampled_peak_includes_file_cache_and_other_processes': True,
        'original_source_files_unchanged': len(original_hashes),
        'input_feedback_binding_verified': True, 'new_training_updates': 0, 'new_paid_api_calls': 0,
        'new_complete_scene_generations': 0, 'whole_scene_success': None,
        'ordinary_reference': {'kind': 'saved_post_hoc_development_reference_not_equal_budget_comparison',
            'model_calls': 0, 'tool_calls': 1, 'intersection_pairs': ordinary['intersection_pairs'],
            'simultaneous_planar': ordinary['condition_B_simultaneous_planar_status']},
        'rows': rows}
    dump('summary.json', summary)
    dump('local_only_media.json', {'redistributed_in_git_or_zip': False, 'files': local_media})
    dump('delivery_check.json', {'status': 'passed', 'checks': ['Actual distinct input hashes and growing input token counts',
        'Initial request identical to the frozen parent request', 'Tool feedback included verbatim in each later observation',
        'Updated object transforms and actual scene-bound image hashes used in next input',
        'Actual EOS-completed generation, local model receipts and three native dispatches',
        'All nine objects, eight non-target object records, shell and 21 original source hashes preserved',
        'All15rendered images match their scene-bound receipts', 'Four cumulative attempts and281.21seconds charged including the initial timeout'],
        'not_checked': 'No independent rerun of FCL, human use or preference evaluation; primary FCL checks were executed once per actual terminal.'})
    with (OUT/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    tr = ''.join(f'<tr><td>{x["turn"]}</td><td>{x["tool_success"]} / {html.escape(str(x["tool_error"]))}</td><td>{x["intersection_pairs"]}</td><td>{x["simultaneous_planar"]}</td><td>{x["model_seconds"]:.2f}</td></tr>' for x in rows)
    pic = 'recovery_v1/turn_03/views/current_state.topdown.png'
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定8B底座真实闭环结果</title><style>body{{font:16px/1.75 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#17212a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4dc;padding:8px}}img{{max-width:700px;width:100%}}code{{overflow-wrap:anywhere}}</style>
<h1>8B底座已完成真实工具闭环；本开发案例没有改善</h1>
<p>2026年11月10日。一个预先冻结的巴黎开发房间，9件现有真实网格家具，移动工具和自定义受限提示。不是新房间生成、官方完整服务复现或独立项目测试。全程零付费API、零训练。真实模型输出、工具反馈、各终态GLB和15张新视图已经保存。</p>
<h2>实际结果</h2><p>共4次模型子进程尝试，其中首次在权重载入时超时，计入预算；后3次生成均以EOS结束，且都提出同一餐椅位姿。工具第一次执行成功，后两次以no_movement拒绝。9件家具均保留，8件非目标对象与建筑记录不变；家具表面相交由5对变6对，新增餐椅—床一对，旧相交没有消除，越界仍为0。没有计算穿透深度。</p>
<table><tr><th>完成回合</th><th>工具成功/错误</th><th>相交对</th><th>同时使用有限见证</th><th>模型子进程秒</th></tr>{tr}</table>
<p>原始要求中的80×50厘米工作与60×40厘米用餐同时使用检查仍为not_found；这不证明连续空间绝对无解。分开使用仍有有限见证。完整人体通行、舒适度、收纳、材质、动态稳定等仍未知；落地参考与源地板顶面差2厘米，不能声称物理支撑通过。全部场景成功未建立。</p>
<h2>输入/反馈有没有真正更新</h2><p>已独立核对：三次请求哈希不同，输入token为3261→3600→3851；后续输入包含对应更新后的对象变换、最新相交表、真实场景绑定五视图，以及原文工具反馈。第三次输入明确含no_movement。原始输出字节不同，但解析后的动作完全相同。因此这是这次受限配置中的重复决策，不能解释为脚本回放、漏传反馈或重复读取同一请求。见<a href="delivery_check.json">核验范围</a>与逐回合原始记录。</p>
<h2>冷启动与成本</h2><p>现有V10032GB、25GiB系统内存、6核CPU。首次200秒门槛在载入2/4分片后结束，未形成输出；原日志保留。其后唯一显式恢复先顺序读取锁定权重预热OS文件缓存，耗时{warm['seconds']:.2f}秒，不复制权重、不加载张量。恢复时每次4个权重分片约7秒载入；没有提高200秒子进程或1200秒累计上限，首轮仍计第1次，新增最多3次。缓存预热属于运行工程修正，不是研究方法。</p>
<p>累计模型子进程{total_model_seconds:.2f}秒，原生dispatch共{stage_seconds['dispatch']:.2f}秒，网格检查{stage_seconds['mesh_audit']:.2f}秒，渲染{stage_seconds['render']:.2f}秒。两次运行wrapper共{costs['pipeline_wrappers_sum_seconds']:.2f}秒；首次启动至最终完成日历时间{costs['first_launch_to_terminal_calendar_seconds']:.2f}秒，包含中间实现和检查，不能称作纯GPU计算。三次完成生成记录的GPU最大已分配约{summary['peak_cuda_allocated_bytes']/1024**3:.2f}GiB；统计在权重载入后重置，不覆盖冷载入峰值。采样容器内存峰值约{summary['sampled_peak_container_memory_bytes']/1024**3:.2f}GiB（包含文件缓存及其他进程）。未获取账户扣费账单。</p>
<h2>普通参考与可以支持的判断</h2><p>此前同初态普通参考用一次沙发移动得到同时使用有限见证，相交5→4；它是开发后选定的事后参考，总计算预算不同，不能宣称本轮公平胜负。但它足以表明这个已知案例存在普通可用操作，本轮不应再围绕它扩训。当前结果也没有比较专业知识、决定监督或两阶段策略，不能由失败推断哪项学习机制有效。</p>
<p>本轮采用自定义移动菜单和JSON式历史观察；作者原生rollout使用阶段系统/工具模板和Memory构造。不能据此推断官方完整生成能力或知识缺失。下一项CPU工作应先固定原生输入格式与相同信息的普通反馈/搜索对照，再在已有独立来源的任务里检验经验是否改善方案提出。新模型调用或训练不由本报告自动启动。24个训练单元仍19个可编码、5个缺视图；不把补齐材料当作方法收益。</p>
<h2>产物与退出状态</h2><p><a href="summary.csv">逐回合结果</a> · <a href="summary.json">完整摘要与成本</a> · <a href="recovery_v1/result.json">真实运行终态</a> · <a href="recovery_v1/turn_03/tools/checkpoint.json">最终可编辑检查点</a> · <a href="recovery_v1/turn_03/tools/partial_scene.glb">最终GLB（本地）</a> · <a href="local_only_media.json">全部本地GLB/视图哈希</a> · <a href="evidence.zip">不含资产/图像/权重的证据包</a>。</p>
<p>恢复wrapper退出0；模型/渲染进程均已结束。后续整理可用CPU，当前无需继续保持GPU计费资源。Git和压缩包只保存文本/元数据，实际网格和视图留在本机并给出哈希，未伪装为可在无资产新机直接运行的独立包。</p><img src="{pic}" alt="实际最终俯视图，已有相交未解决"></html>'''
    (OUT/'stage_report.html').write_text(report)
    dump('handoff.json', {'status': 'bounded_real_model_execution_complete_not_research_complete',
        'next_action': 'CPU source/input attribution and frozen strong controls before any new model run or training',
        'new_model_outputs': 3, 'all_model_attempts': 4, 'new_training_updates': 0,
        'whole_room_success': None, 'do_not_retry_this_case_automatically': True,
        'gpu_run_and_render_terminal': True, 'ordinary_reference_is_post_hoc': True})
    files = [p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.png', '.glb', '.blend', '.bin', '.pyc']
             and p.name not in ['artifact_manifest.json', 'evidence.zip']]
    files += [p for p in (READY/'model_run_v1').rglob('*') if p.is_file()]
    files += [READY/'model_run_v1_process_receipt.json', READY/'allocation_before_model_run_v1.json',
              READY/'model_run_v1.stdout', READY/'model_run_v1.stderr',
              REPO/'scripts/autodl/continue_bounded_base_after_load_timeout.py']
    manifest = {'files': [{'path': str(p.relative_to(REPO)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(files))],
                'local_only_media': 'local_only_media.json', 'scope': 'Raw input/output, receipts and code; excludes weights, raw images and meshes'}
    dump('artifact_manifest.json', manifest)
    with zipfile.ZipFile(OUT/'evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files+[OUT/'artifact_manifest.json'])):
            z.write(p, str(p.relative_to(REPO)))
    print(json.dumps({k: summary[k] for k in ['status', 'model_subprocess_attempts', 'completed_model_outputs', 'tool_calls', 'new_pairs', 'simultaneous_planar']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
