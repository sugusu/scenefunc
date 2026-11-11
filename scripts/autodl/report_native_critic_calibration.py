"""Report three actual contract probes and the ordinary geometric control."""
import argparse
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path
import shutil
import zipfile
from cpu_continuation import read,dump,sha


def pair_agreement(pred,ref):
    if pred is None:return None
    pos={x:i for i,x in enumerate(pred)}
    pairs=list(itertools.combinations(ref,2))
    return dict(correct=sum(pos[x]<pos[y] for x,y in pairs),total=len(pairs))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    if (root/'summary.json').exists():raise FileExistsError('Report exists; do not rerun experiments')
    refs=read(root/'labels_quarantine.json')
    dev=read(root/'development_pool_reference_quarantine.json')
    refs.append(dict(arm='B_native_development_pool',id='same_room_four_actual_states',numimages=4,
                     reference=dev['ordinary_rule_reference']))
    rows=[]
    for ref in refs:
        folder=root/ref['arm']/'turn_00'
        proc=read(folder/'model_process_receipt.json');assert proc['status']=='terminal' and proc['exit_code']==0
        model=read(folder/'model/receipt.json');assert model['network_connection_attempts']==0
        eng=read(next((folder/'model/calls').glob('*/receipt.json')))
        pred=read(folder/'model/parsed_ranking.json')['ranking']
        n=ref['numimages'];assert eng['input_shapes']['image_grid_thw']==[n,3]
        pixels=eng['input_shapes']['pixel_values'][0]
        assert pixels==n*64*64,'Native full-resolution patch grid changed'
        rank=ref['reference']
        rows.append(dict(arm=ref['arm'],id=ref['id'],role='public_benchmark_qualification' if ref['arm']!='B_native_development_pool' else 'correlated_development_pool',
                         images=n,full_input_resolution='1024x1024 per image',max_pixels=model['max_pixels'],
                         prediction='>'.join(pred) if pred else None,reference='>'.join(rank),
                         reference_kind='author_published_reference' if ref['arm']!='B_native_development_pool' else 'declared_ordinary_geometric_priority',
                         parse_valid=pred is not None,exact_match=pred==rank,best_choice_match=bool(pred and pred[0]==rank[0]),
                         pairwise_agreement=pair_agreement(pred,rank),input_tokens=eng['input_tokens'],output_tokens=eng['output_tokens'],
                         model_process_seconds=proc['elapsed_seconds'],peak_gpu_gib=eng['peak_cuda_allocated_bytes']/1024**3,
                         ended_with_eos=eng['ended_with_eos'],paid_model_API_calls=0,training=False,new_scenes=0))
    total=sum(x['model_process_seconds'] for x in rows);assert len(rows)<=3 and total<=600
    acquired=read(root/'acquisition.json')['files']
    for x in acquired:assert sha(x['path'])==x['sha256']
    for path,digest in read(root/'input_freeze_sha256.json').items():assert sha(path)==digest
    repo=root.parents[2]
    protected=read(repo/'runs/autodl/20261111_feedback_information_control/source_before_after.json')['original_inputs']
    check=[dict(path=x['path'],expected=x['before'],actual=sha(x['path'])) for x in protected]
    assert all(x['expected']==x['actual'] for x in check)
    dump(root/'original_input_preservation.json',dict(count=len(check),all_unchanged=True,rows=check))
    result=dict(rows=rows,model_process_seconds=total,actual_model_forward_calls=3,paid_model_API_calls=0,optimizer_updates=0,new_scenes=0,
                interpretation='Native-format 2-row qualification, not complete benchmark accuracy or single-factor input ablation; no novel method established',
                source_role='Public joint-SFT checkpoint, no v10 LoRA adapter or RL checkpoint',
                old_results_preserved=True,full_scene_successes_established=0)
    dump(root/'summary.json',result)
    fields=list(dict.fromkeys(k for x in rows for k in x))
    with (root/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');wr.writeheader();wr.writerows(rows)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>' for k in ['id','prediction','reference','exact_match','best_choice_match','model_process_seconds'])+'</tr>' for x in rows)
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原生Critic输入校准</title><style>body{font:16px/1.7 sans-serif;margin:36px;max-width:1100px}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>原生输入校准：修正上一轮失败结论的范围</h1>
<p>公开权重是Scene-Builder联合SFT阶段，不能称为作者v10 Critic或Scene-GRPO最终模型。源码的Critic输入通常为2或4张俯视图、无system消息，回答使用reasoning和answer标签；上轮25图、5候选、C0标签与JSON回答不等价。原生策略的Task/Memory与标注图也不同于当前有限移动桥。不能把旧配置的失败直接推广为原生模型普遍无能。</p>
<p>在模型运行前固定公开标注顺序的首道二选题scene_otc_0001与首道四选题scene_otc_0095。原始PNG按官方清单逐字节核验，参考标签独立保存，未进入模型请求，也未用于训练。使用保存processor的原始像素上限，实际各图1024×1024，没有压到上轮的256像素规模；仍采用V100适用FP16，不声称数值等价于作者BF16示例。</p>
<table><tr><th>问题</th><th>实际排序</th><th>参考</th><th>完全匹配</th><th>首选匹配</th><th>模型进程秒</th></tr>'''+table+'''</table>
<p>两道公开题正确不能推导149题准确率或泛化。第三题是已用开发房间的三个方法末态加普通搜索较早保留状态，仍是相关开发数据；它选对首项，但将有新增墙面相交的B排在C、A之前，违反明确优先级。普通规则投影得到D&gt;C&gt;A&gt;B，且无需模型确定这个顺序，属于必要基线，并非新方法。所有候选仍有已知家具相交，所以不能将选到D当作完整场景生成成功。</p>
<p>第三次调用是事先声明预算内的探索性范围调整：原拟原生Actor探测需要尚未复现的标注图及全工具，不能虚称原生对齐，因此改做原生Critic模式的已有开发池选择。四图、完整分辨率、输出格式与显式物体清单同时改变；不能把差异单独归因于提示或像素数。旧失败记录和调度错误均保留，无训练，无付费模型API，无新场景。</p>
<p>本轮还阅读iARCS v3的方法、实验与指定附录，以及固定公开代码的反思记忆与奖励加载部分。它已使用任务分解、执行奖励和历史程序记忆；这些不能单独作为新创新。论文每项任务适配LoRA，不能当作已证明一个零样本多任务策略。没有运行iARCS、下载其新模型或重建3D-FUTURE资产。</p>
<p>历史证据另有更新：2026-10-14记忆试验其实已有六段锁定八轮会话，旧决策记录未显示追加方案独立收益，也没有进入三维阶段；本轮只复查原记录，未重新调用其模型，不应继续将该文本试验描述为“从未做过”。</p>
<p><a href="summary.csv">实际调用CSV</a> · <a href="ordinary_projection.json">普通规则控制</a> · <a href="reading_ledger.json">阅读范围</a> · <a href="algorithm_and_evidence.txt">算法边界与证据</a> · <a href="evidence.zip">证据包</a></p></html>'''
    (root/'stage_report.html').write_text(report)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
