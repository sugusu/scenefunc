"""Build a claim ledger and pilot results directly from completed source artifacts."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).resolve().parents[2];out=r/'research/920/paper_v1'
paths={'training':'runs/920/P920-OFFLOAD-v1_20261105T102528Z/TRAINING_RESULT.json','comparison':'runs/920/P920-CASA-COMP-v1_20261105T123924Z/COMPARISON.json','coverage':'runs/920/P920-CASA-COMP-v1_20261105T123924Z/TRAINING_TASK_COVERAGE.json','construction':'runs/920/P920-SBUILD-v1_20261105T143623Z/SUMMARY.json','mask':'scripts/920/decision_pairing.py','trainer':'scripts/920/decision_pilot_offload.py'}
refs={k:{'path':v,'sha256':hashlib.sha256((r/v).read_bytes()).hexdigest()} for k,v in paths.items()}
a=json.loads((r/paths['training']).read_text());b=json.loads((r/paths['comparison']).read_text());c=json.loads((r/paths['coverage']).read_text());d=json.loads((r/paths['construction']).read_text())
assert a['total_optimizer_updates']==4 and len(b['arms'])==3 and d['new_training_updates']==0
claims=[
 {'id':'M1','claim':'The matched four-update LoRA mechanism pilot completed with the frozen base/vision and source-balanced target loss.','status':'supported_mechanism_only','evidence':['training','trainer'],'not_supported':'Convergence, final held-out loss reduction, design learning, cold optimizer resume.'},
 {'id':'M2','claim':'The action-context treatment supplies the teacher-forced reason prefix only in paired action examples; both arms supervise the same reason and action targets.','status':'supported_specific_operationalization','evidence':['mask','trainer'],'not_supported':'A general proof that all forms of pairing outperform all forms of separate learning. Reason-only examples in both groups already see the scene.'},
 {'id':'R1','claim':'The Casa base/separate/paired development attempts produced 7/6/6 furniture instances and two detected contact pairs each.','status':'supported_single_exposed_development_project','evidence':['comparison'],'not_supported':'Independent project population effect, calibrated professional quality, or a formal 0/3 binary success rate.'},
 {'id':'D1','claim':'All ten pilot demonstration actions were moves of already populated rooms; Casa evaluation began with no furniture and required construction.','status':'supported_post_hoc_task_coverage_diagnosis','evidence':['coverage'],'not_supported':'That this mismatch caused the observed failures or that correcting it will produce learning gains.'},
 {'id':'D2','claim':'A separate untrained candidate now provides nine native additions with one reused source reason and fifty bound views.','status':'supported_new_data_candidate','evidence':['construction'],'not_supported':'Nine independent professional decisions, source diversity growth, complete fixture installation, human-use acceptance or training improvement.'},
 {'id':'H1','claim':'Pairing professional reasons with editable scene decisions improves unseen whole-room design beyond same-material separate learning and strong non-training baselines.','status':'unproven_research_hypothesis','evidence':[],'missing':['Adequate independent source families and construction coverage','Training beyond a mechanism smoke test with separate approval','Same-knowledge and strong feedback controls on matched tasks','Sealed independent project evaluation','Independent professional whole-room judgments and uncertainty analysis']}
]
ledger={'created_at':datetime.now(timezone.utc).isoformat(),'status':'working_paper_evidence_not_submission','sources':refs,'claims':claims,'new_model_calls':0,'new_training_updates':0}
(out/'CLAIM_EVIDENCE.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
lines=['# 已完成试运行的结果草稿','','本节仅写已结束并可核对的结果；强提示/同知识的已完成开发对照单列。论文主假设未得到验证。全部数据来源及哈希见 CLAIM_EVIDENCE.json。','','## 四更新机制试运行','','两组各两次更新、每组40次微样本反向，共4次更新、80次反向。每个更新循环监督982个目标token。以下目标值在对应更新之前记录；不是最终适配器的独立评价损失。','','| 组 | 更新前循环1目标 | 更新前循环2目标 | 峰值显存 GiB | 更新次数 |','|---|---:|---:|---:|---:|']
for name,x in a['arms'].items():lines.append(f"| {name} | {x['cycles'][0]['source_balanced_objective']:.9f} | {x['cycles'][1]['source_balanced_objective']:.9f} | {x['peak_allocated_GiB']:.6f} | {x['updates']} |")
lines+=['','两组的第二循环目标均略升。配对组在第一次更新前已经较低，因此两组原始目标差不能解释为训练收益。保存/清零/同进程重载验证通过，冷优化器恢复未验证。','','## 同一 Casa 开发房间的整屋输出','','| 组 | 模型/工具调用 | 末态家具数 | 三角接触对数 | 总运行秒 |','|---|---:|---:|---:|---:|']
for x in b['arms']:lines.append(f"| {x['arm']} | {x['model_calls']}/{x['tool_calls']} | {x['furniture_count']} | {len(x['triangle_contacts'])} | {x['actual_seconds']:.3f} |")
lines+=['','三组均未补齐原任务所需角色，完整功能仍未验收。配对组在0.5/0.6米方形身体假设下另有露台接近失败。这些是一个已暴露项目的观察，不作显著性检验，也不把三组当三个独立项目。家具更多不意味着质量更高。','','## 数据覆盖解释及后续候选','','训练材料为三来源的十个相关移动步骤，输入已有13–16件家具；Casa从零家具开始，要求添加与整屋闭合。该差异在结果产生后分析，不能当作预注册的因果解释。','','后续SBUILD建设候选包含六件预装设施和九次添加，十次观察共五十张图，仅一步配真实来源原则及标注的研究者解释。两组目标585 token相同，输入88078/88160 token不同。它复用一个已有来源，未进行新训练，不计新增独立决定。条件接近量测通过仍不能证明容量、坐席使用、安装、隐私或专业品质。','','## 本轮尚不能写入的结论','','不能写“专业配对学习已提升设计能力”“证明优于充分提示/检索”“具有跨项目泛化”或“达到专业整屋品质”。需要更多独立项目、足够且审定的训练、公平基线、封存测试和真实独立评审。']
(out/'PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n')

q=r/'runs/920/P920-KBASE-v1_20261105T151743Z/COMPARISON.json'
if q.exists():
    k=json.loads(q.read_text());assert k['status']=='two_development_baselines_completed_and_reviewed'
    ledger['sources']['knowledge_baseline']={'path':str(q.relative_to(r)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
    ledger['claims'].append({'id':'R2','claim':'On the exposed Casa development task, strong guidance yielded an empty room; the same guidance plus all ten pilot reason texts yielded only a bed and shower.','status':'supported_single_project_negative_outcomes','evidence':['knowledge_baseline'],'not_supported':'A professional quality gain from knowledge, necessity of training, population generalization or pairing benefit.'})
    (out/'CLAIM_EVIDENCE.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    text='\n## 强提示与同知识的完成结果\n\n两条件均48次总模型/工具尝试中的各24次，0训练。强提示组24次全是查询，末态0家具；同知识组末态仅床和淋浴，厨房/收纳/餐位等缺失。查询schema异常分别24与13；引擎仍接受。两组零检测接触不能抵消缺失功能，知识组还有条件通路损失。\n\n'
    text+='| 条件 | 家具 | 实际运行秒 | 输出token |\n|---|---:|---:|---:|\n'
    for x in k['arms']:text+=f"| {x['arm']} | {x['furniture_count']} | {x['actual_seconds']:.3f} | {x['output_tokens']} |\n"
    text+='\n本对照使用共同16384输入上限，旧三组为12288且没有此强提示；跨批结果不是严格同计算比较。固定提供全部历史理由不等于实现检索排序；其中包含研究者操作解释。新三家族七知识未进入此冻结批次。只有一个暴露开发项目，不做显著性或跨项目推断。\n'
    with (out/'PILOT_RESULTS.md').open('a') as f:f.write(text)

print(json.dumps({'claims':len(ledger['claims']),'source_hashes':len(ledger['sources']),'model_calls':0,'training_updates':0}))
