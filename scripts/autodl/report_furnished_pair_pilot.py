"""Persist and audit actual pilot terminals; retain censored infrastructure arm."""
import argparse
import csv
import gzip
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
from cpu_continuation import read, dump, sha, execute

W=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
REPO=W/'autodl_continuation_20261110/repository'
NATIVE='/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python'


def pairs(audit):
    return {tuple(p) for p in audit['furniture_surface_intersection_pairs']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--artifacts',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();artifacts=a.artifacts.resolve();artifacts.mkdir(parents=True,exist_ok=False)
    states={arm:read(root/arm/'state.json') for arm in ['B0','B1','B1_repaired','B2']}
    assert all(s['status']!='active' for s in states.values()),'Actual trajectories still running'
    initial=read(root/'initial_freeze.json')
    base=read(W/'autodl_continuation_20261110/cpu_replay/keep/audit.json')
    rows=[];files=[];traces=[];unique_images={}
    for arm,state in states.items():
        receipts=[read(p) for p in sorted((root/arm).glob('turn_*/model_process_receipt.json'))]
        assert all(r['status']=='terminal' for r in receipts)
        engines=[read(p) for p in sorted((root/arm).glob('turn_*/model/calls/*/receipt.json'))]
        row=dict(project_id='19_remote_studio_researcher_reconstruction',source='prior selected HSSD mesh development state; historical project identity unverified',
                 split='development',condition='B_work_materials_remain_while_meal_surface_available',arm=arm,seed=0,
                 start_hash=initial['checkpoint_sha256'],code_sha='see source_manifest.json',
                 weight_revision='63e235a0c0d6e5bc2f8f3485214df4457fd2e672',run_id=initial['run_id'],
                 termination=state['status'],model_calls=sum(bool(e.get('generation_started')) for e in engines),
                 model_subprocess_attempts=len(receipts),tool_calls=len(list((root/arm).glob('turn_*/calls.json'))),
                 input_tokens=sum(e.get('input_tokens',0) for e in engines),output_tokens=sum(e.get('output_tokens',0) for e in engines),
                 gpu_seconds=sum(p['elapsed_seconds'] for p in receipts),cpu_seconds=0.,required_pass='not_established',
                 new_damage=None,initial_violations=len(pairs(base)),unknown_count=len(base['not_checked']),
                 final_scene_path='',complete_scene_success=None)
        cpu_receipts=[read(p) for p in (root/arm).glob('turn_*/*_receipt.json') if p.name in ['dispatch_receipt.json','mesh_audit_receipt.json','render_receipt.json']]
        row['cpu_seconds']=sum(p['elapsed_seconds'] for p in cpu_receipts)
        row['cpu_seconds']+=sum(read(p).get('seconds',0) for p in (root/arm).glob('turn_*/measurements.json'))
        row['cpu_seconds']+=sum(read(p).get('seconds',0) for p in (root/arm).glob('turn_*/proxy_candidates.json'))
        if arm=='B2':row['proxy_candidates']=state['proxy_candidates']
        if arm=='B1':
            row.update(termination='censored_preforward_scheduling_failure',required_pass='not_evaluated')
            rows.append(row);continue
        cp=Path(state['checkpoint']);glb=cp.parent/'partial_scene.glb'
        dest=artifacts/arm;dest.mkdir()
        packed=dest/'terminal_scene.glb.gz'
        if shutil.disk_usage(root).free-glb.stat().st_size < 5*1024**3:
            raise RuntimeError('Insufficient reserve for durable terminal; no old output deletion')
        with glb.open('rb') as f,gzip.open(packed,'wb',compresslevel=3) as g:shutil.copyfileobj(f,g)
        # Decompress to fresh RAM and hash exact bytes, not merely gzip validity.
        restored=Path('/dev/shm/FRP1-PARIS-B-20261111')/f'{arm}_durability_check.glb'
        with gzip.open(packed,'rb') as f,restored.open('wb') as g:shutil.copyfileobj(f,g)
        assert sha(restored)==sha(glb)
        shutil.copyfile(cp,dest/'checkpoint.json')
        audit_path=root/arm/'terminal_reloaded_audit.json'
        execute([NATIVE,str(W/'paris_local_run_results_20261104/audit_scene.py'),'--checkpoint',str(cp),'--out',str(audit_path)],
                root/arm,'terminal_reloaded_audit',120)
        audit=read(audit_path);stored=read(state['audit'])
        for key in ['scene_sha256','furniture_surface_intersection_pairs','out_of_rectangular_floor_objects','architecture_surface_intersection_pairs']:
            assert audit[key]==stored[key],key
        preserved={o['object_id'] for o in read(cp)['payload']['placed_instances']}
        original={o['object_id'] for o in read(initial['source_checkpoint'])['payload']['placed_instances']}
        assert preserved==original
        joint=read(state['joint']);measured=read(state['measured']);old=pairs(base);now=pairs(audit)
        floor_diff=audit['out_of_rectangular_floor_objects']
        oldwalls={tuple(p) for p in base['architecture_surface_intersection_pairs']['walls_doors_windows']}
        walls={tuple(p) for p in audit['architecture_surface_intersection_pairs']['walls_doors_windows']}
        row.update(final_scene_path=str(glb),durable_scene_gzip=str(packed),scene_sha256=sha(glb),
                   placed_objects=len(preserved),remaining_intersection_pairs=len(now),new_damage=len(now-old),
                   eliminated_initial_intersection_pairs=len(old-now),new_architecture_intersections=len(walls-oldwalls),
                   outside_floor=len(floor_diff),planar_A=measured['results']['scene_section_free']['separate_use']['status'],
                   planar_B=joint['status'],required_planar_subcheck_pass=joint['status']=='witness')
        files.append(dict(path=str(packed),sha256=sha(packed),uncompressed_sha256=sha(glb),bytes=packed.stat().st_size))
        # Persist every post-action observation exactly; requests are immutable.
        for view_receipt in sorted((root/arm).glob('turn_*/views_receipt.json')):
            vr=read(view_receipt);target=dest/view_receipt.parent.name/'views';target.mkdir(parents=True)
            for view in vr['views']:
                src=Path(view['path']);dst=target/src.name
                if view['sha256'] in unique_images:
                    # Deduplicate only these NEW artifacts, never old outputs.
                    os.link(unique_images[view['sha256']],dst)
                else:
                    if shutil.disk_usage(root).free-src.stat().st_size < 5*1024**3:raise RuntimeError('View persistence would cross reserve')
                    shutil.copyfile(src,dst);unique_images[view['sha256']]=str(dst)
                assert sha(dst)==view['sha256']
                files.append(dict(path=str(dst),original_request_image=str(src),sha256=view['sha256'],bytes=dst.stat().st_size))
        for p in sorted((root/arm).glob('turn_*/calls.json')):
            stage=p.parent;j=read(stage/'joint.json') if (stage/'joint.json').exists() else {}
            aa=read(stage/'audit.json') if (stage/'audit.json').exists() else {}
            traces.append(dict(arm=arm,turn=stage.name,calls=read(p),planar_B=j.get('status'),
                               intersection_pairs=aa.get('furniture_surface_intersection_pairs'),
                               outside_floor=aa.get('out_of_rectangular_floor_objects')))
        rows.append(row)
    allreceipts=[read(p) for p in root.glob('B*/turn_*/model_process_receipt.json')]
    assert len(allreceipts)<=12 and sum(p['elapsed_seconds'] for p in allreceipts)<=1200
    # The recovered request is byte-identical to the failed pre-forward request.
    assert sha(root/'B1/turn_00/request.json')==sha(root/'B1_repaired/turn_00/request.json')
    protected=read(REPO/'runs/autodl/20261111_feedback_information_control/source_before_after.json')['original_inputs']
    check=[dict(path=x['path'],expected=x['before'],actual=sha(x['path'])) for x in protected]
    assert all(x['expected']==x['actual'] for x in check)
    dump(root/'protected_source_recheck.json',dict(count=len(check),all_unchanged=True,rows=check))
    for p,h in initial['asset_and_state_hashes'].items():assert sha(p)==h
    dump(root/'artifact_manifest.json',dict(files=files,all_frozen_scene_inputs_unchanged=True,
        restoration='gzip -dc terminal_scene.glb.gz > /dev/shm/restored_scene.glb; verify uncompressed_sha256',
        artifacts_not_duplicated_into_git=True))
    dump(root/'action_trace.json',traces)
    dump(root/'summary.json',dict(rows=rows,actual_model_subprocess_attempts=len(allreceipts),
         all_model_process_seconds=sum(p['elapsed_seconds'] for p in allreceipts),paid_api_calls=0,training=False,
         gpu_seconds_definition='GPU-enabled model subprocess wall lifetime, including load and failed attempt; not CUDA kernel time',
         cpu_seconds_definition='Recorded CPU subprocess wall durations plus planar/proxy function times; concurrent branches mean not end-to-end elapsed time',
         unknown_count_definition='Unverified audit scope categories, not a count of unsatisfied user requirements',
         scope='One furnished development reconstruction; no independent-project generalization or new complete-scene generation',
         complete_scene_successes_established=0,protocol_deviation='Retained B1 pre-forward concurrency OOM; unchanged request restored with exclusive model lock; not a confirmatory comparison'))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    codefiles=[REPO/'scripts/autodl'/name for name in ['furnished_pair_pilot.py','furnished_pilot_process.py','furnished_ordinary_search.py','report_furnished_pair_pilot.py','furnished_shared_pool.py','model_step.py','bounded_trajectory.py']]
    dump(root/'source_manifest.json',{str(p.relative_to(REPO)):sha(p) for p in codefiles})
    tab=''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in ['arm','termination','model_calls','tool_calls','planar_A','planar_B','remaining_intersection_pairs','new_damage','new_architecture_intersections','gpu_seconds','cpu_seconds'])+'</tr>' for row in rows)
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实际配对试跑</title><style>body{font:16px/1.6 sans-serif;margin:36px;max-width:1400px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}pre{white-space:pre-wrap}</style><h1>已有家具场景：实际配对试跑</h1>
<p>这是一项研究者重建开发场景上的 move-only 编辑诊断，不是从文字开始的新完整生成，也未建立论文方法优势。原项目身份未核实；跨项目 Waterloo 说明是两组共同检索材料，不是本房间的专家答案。</p>
<p>B0 为原有联合输出；B1_repaired 增加适用条件核查提示，资料、模型和动作相同；B2 为普通有限 AABB 代理搜索。B1 原尝试在权重加载时因本次调度错误与 B0 重叠而 OOM，尚未开始生成。原失败保留并计费；修复只增加独占锁，重新使用逐字节相同请求。此执行偏离阻止将其写成严格预注册结果。</p>
<p>planar_A/B 只验证零厚度矩形在现有网格截面中的有限位置见证；not_found 不是不可行证明。家具相交、边界、原有损害与未知项分别报告。工作材料与用餐面同时保留，不意味着两人同时活动。</p>
<table><tr><th>方法</th><th>终止</th><th>实际forward</th><th>原生动作</th><th>平面A</th><th>平面B</th><th>剩余相交对</th><th>新增家具相交对</th><th>新增建筑相交对</th><th>模型进程秒</th><th>CPU阶段墙钟秒</th></tr>'''+tab+'''</table>
<p>B2 的全部代理候选均记录；代理不是完整 FCL 或人体功能优化。它的最后状态新增了床与浴室墙面的相交，不能只看家具相交对减少就称为整体改善。早期已保存的状态也全部保留，共享候选诊断会明确计入建筑相交。它与模型的 FLOPs、工具前内部计算不同，不能只凭原生调用数声称公平效率。各方法最多4次原生动作，模型总启动最多12次、累计1200秒，没有新API或训练。</p>
<p>所有9个初始家具ID、资产与建筑保持。保留未使用的注册资产事实，不以改名当作语义完整。动态稳定、通行、功能朝向、收纳、窗景、美学与真实偏好均未知，整场成功未建立。</p>
<p>完整逐步动作、原始请求与模型回答见本目录；summary.csv、summary.json、artifact_manifest.json 给出费用口径、持久场景与全部视图哈希。场景以 gzip 保存，解压后字节哈希已复核。579项保护输入重新核对不变。仅删除一份无关的旧公开演示视频下载缓存，保留官方URL与哈希；没有删除旧实验或当前模型。</p>
<p><a href="summary.csv">逐组CSV</a> · <a href="action_trace.json">动作与结果</a> · <a href="artifact_manifest.json">持久场景和视图</a> · <a href="PAIR_PROTOCOL.json">协议与明确偏离</a></p></html>'''
    (root/'stage_report.html').write_text(doc)
    with zipfile.ZipFile(root/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name not in ['evidence.zip','exclusive_model.lock'] and '__pycache__' not in p.parts:
                z.write(p,str(p.relative_to(root)))
        for p in codefiles:z.write(p,'code/'+p.name)
    dump(root/'delivery_receipt.json',dict(report=str(root/'stage_report.html'),csv=str(root/'summary.csv'),
         evidence_zip=str(root/'evidence.zip'),zip_sha256=sha(root/'evidence.zip'),
         source_files_unchanged=len(check),disk_free_bytes=shutil.disk_usage(root).free,
         archive_large_assets_referenced_via_manifest=True))
    print(json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
