"""Report completed input recovery and encoding without policy-effect claims."""
import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args();root=a.run
    read=lambda p:json.loads(Path(p).read_text())
    pipeline=read(root/'material_pipeline.json');verification=read(root/'verification.json')
    if pipeline['status']!='completed' or verification['status']!='passed':raise ValueError('Incomplete pipeline')
    for p,h in read(root/'execution_source_hashes.json').items():
        if sha(p)!=h:raise ValueError('Execution source changed: '+p)
    batch=read(root/'views_v2/result.json');rows=[]
    for r in batch['rows']:
        ex=read(r['export']);an=read(r['annotation']);view=Path(r['annotation']).parent
        render=read(view/'result.json')
        rows.append({'unit_id':r['unit_id'],'objects':ex['objects'],'images':len(an['images']),
                     'render_seconds':render['renderer_seconds'],'annotation_seconds':an['seconds'],
                     'peak_render_rss_bytes':render['sampled_process_tree_peak_rss_bytes'],
                     'source_runtime_json_identical':ex['source_room_sha256']==ex['runtime_room_sha256'],
                     'topdown':str((view/'step_006.topdown.png').relative_to(root))})
    joint=read(root/'joint_multimodal_v4/summary.json');material=read(root/'material_v4/summary.json')
    cleanup=read(root/'archive_cleanup_receipt.json')
    summary={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'previous_goal_turn':'progress: exact native serialization cause verified and two observation states completed',
             'this_goal_turn':'progress: complete fixed five-state native observation batch, unchanged full-corpus encoding and joint control',
             'fixed_training_units':24,'encoded_before':14,'encoded_after':joint['encoded'],'missing_views':joint['missing_views'],
             'fixed_states_recovered':len(rows),'new_images':sum(r['images'] for r in rows),
             'native_resource_files_byte_exact':24,'native_objects_per_state':26,
             'mesh_objects':2,'proxy_objects':15,'runtime_furniture_objects':15,'ceiling_light_objects':2,'architectural_objects':9,
             'render_wall_seconds':sum(r['render_seconds'] for r in rows),
             'render_sampled_peak_rss_bytes':max(r['peak_render_rss_bytes'] for r in rows),
             'annotation_seconds':sum(r['annotation_seconds'] for r in rows),
             'split_encoding_seconds':material['seconds'],'joint_encoding_seconds':joint['seconds'],
             'rows':rows,'verification_checks':verification['count'],
             'new_model_calls':0,'training_updates':0,'new_complete_scenes':0,'new_paid_api_calls':0,
             'training_ready':False,'full_runtime_ready':False,'method_gain_established':False,'goal_complete':False,
             'gpu_requested':False,'historical_pixel_equivalence':False}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    table=''.join('<tr><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.2f}</td><td>{}</td></tr>'.format(
        html.escape(r['unit_id']),r['images'],r['render_seconds'],r['annotation_seconds'],
        '逐字一致' if r['source_runtime_json_identical'] else '原生四元数序列化一致，原始哈希不同') for r in rows)
    text='''<!doctype html><meta charset="utf-8"><title>固定Storage观察恢复与共同材料</title>
<style>body{max-width:1050px;margin:40px auto;font:17px/1.7 sans-serif}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}img{max-width:48%}</style>
<h1>五个固定状态已恢复；完整生成与学习收益仍未建立</h1>
<p>这是同一开发房间的五个已记录状态，不是五个新生成房间。原构建器运行后，24个资源文件与归档逐字匹配。每个状态包含26个对象：2个真实网格物体、15个代理物体（含2盏顶灯）和9个建筑对象。没有替换冻结状态或监督答案。</p>
<table><tr><th>状态</th><th>视图</th><th>渲染秒</th><th>标注秒</th><th>状态核验</th></tr>'''+table+'''</table>
<p>首次构建失败来自路径混用；第一次批次在第二状态因四元数数值读写差异停止。失败和额外耗时保留。修订只重新计算原生Quaternion→RotationMatrix→Quaternion序列化，逐字核对其预期输出，没有加入位姿容差，也没有放宽质量阈值。成功的第一状态和第二状态导出复用，其余按原顺序继续。</p>
<p>五视图使用已固定的CPU渲染设置。当前Drake与历史运行版本不同；没有建立历史像素等价。各组共同获得这一版本的实际图像，不能把新的观察与旧适配器历史结果混作同一实验。</p>
<h2>实际编码与强对照</h2><p>'''+str(joint['encoded'])+'''/24个固定单元完成实际多模态processor编码，仍有'''+str(joint['missing_views'])+'''个缺视图。68条分组组件、24个联合序列以及2975个原始目标token全部保留；没有改目标、权重或补写决定。普通联合对照与分阶段方案的视图张量一致，联合推理前缀也经实际编码核对。生成接口检查使用显式目标回放，不是模型推理。</p>
<p>剩余air_edit_01、waterloo_edit_02/04、lifecycle_02/03仍保留。缺决定监督的14个单元也全部保留。不能以当前可编码子集开始训练并称为原协议的完整实验。</p>
<h2>方法与工具边界</h2><p>新增最近邻和现有代码的解释见 <a href="method_review.html">方法复核</a>、<a href="reading_ledger.json">实际阅读记录</a>。理由先行、加权交叉熵、反馈编辑与正确性监控都有普通方法或直接前作；本轮没有形成新的学习效果证据。真实检索网格不等于原生VLM朝向/材质/质量分析和完整资产路径均已执行。</p>
<h2>成本与证据</h2><p>本轮新增模型调用、训练更新、付费API和完整生成均为0。原生成记录、原材料和来源文件只读；已完成BLEND中间件在图像校验后清理。旧交付压缩包仅在所有成员与展开文件哈希一致时删除，未通过该条件的包保留；详见清理回执。</p>
<ul><li><a href="summary.json">逐状态耗时和峰值内存</a></li><li><a href="verification.json">最终核验及适用范围</a></li><li><a href="material_pipeline.json">实际编码命令、退出码、耗时</a></li><li><a href="views_v1/result.json">原批次失败</a> / <a href="views_v2/result.json">完成批次</a></li><li><a href="serialization_audit.json">序列化原因与错误状态拒绝检查</a></li><li><a href="archive_cleanup_receipt.json">旧包装清理</a> / <a href="transient_cleanup.json">新构建重复中间件清理</a></li></ul>
<h2>实际观察预览</h2><p>下图用于检查恢复内容，代理与低采样噪声可见，不充当功能金标准。</p>'''
    for r in (rows[0],rows[-1]):
        text+='<a href="'+html.escape(r['topdown'])+'"><img src="'+html.escape(r['topdown'])+'" alt="'+r['unit_id']+'"></a>'
    (root/'stage_report.html').write_text(text)
    local=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in root.rglob('*') if p.is_file() and p.suffix in ['.glb','.bin','.blend']]
    (root/'local_only_resources.json').write_text(json.dumps({'note':'Binary resources remain local; no raw mesh redistribution in Git. Recovery uses original licensed assets.','files':local},indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False))


if __name__=='__main__':main()
