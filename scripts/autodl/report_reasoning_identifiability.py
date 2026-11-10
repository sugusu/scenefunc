"""Summarize actual public samples, source execution and identification limits."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main(repo,s):
    samples=json.loads((s/'il3d_static_layout_sample_v2/result.json').read_text())
    audit=json.loads((s/'actual_input_audit_v2/result.json').read_text())
    support=json.loads((s/'actual_input_audit_v2/empirical_support.json').read_text())
    mask=json.loads((s/'mask_boundary_v1/result.json').read_text())
    original=json.loads((repo/'runs/autodl/20261111_supervision_and_branch_state/source_before_after.json').read_text())
    verified=[]
    for x in original['checks']:
        p=Path(x['path']);actual=sha(p) if p.exists() else None
        verified.append(dict(path=str(p),before=x['baseline_sha256'],after=actual,same=actual==x['baseline_sha256']))
    assert all(x['same'] for x in verified)
    for path,expected in audit['input_hashes'].items():assert sha(repo/path)==expected
    dump(s/'source_before_after.json',dict(original_inputs=verified,all_same=True,
         actual_component_input_hashes_rechecked=audit['input_hashes'],source_or_evidence_deleted=0))
    reading=[
      dict(title='Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought Reasoning',
           URL='https://arxiv.org/html/2605.24286v1',
           read='HTML abstract, introduction/related work, sections3-5,6.1-6.4; AppendixA-D including implementation costs and limitations. Figure captions read; no PDF/figure visual audit.',
           source_pin=mask['source_pin'],source_scope='cot_masking.py lines1-331 completely; dp_actor.py targeted mask dispatch, old-log-prob override and policy-update lines379-448,748-838,944-1077. README overview. Not every actor function or transformer patch read.',
           executed='Original three standalone mask helpers via AST extraction on actual archived assistant-prefix/target IDs with an explicitly synthetic leading prompt. No model, backward or training.',
           consequence='A preexisting structural-training baseline. Cannot rename prompt/CoT routing as a new mechanism. Forward masks and backward restrictions make different claims; no generic deployment independence follows.',
           code_distribution='Targeted unmodified source remains a read-only RAM research copy; only URLs/hashes and original analysis redistributed. Reacquisition helper provided.'),
      dict(title='The Observability Gap: Why Output-Level Human Feedback Fails for LLM Coding Agents',
           URL='https://arxiv.org/html/2603.26942v1',
           read='Complete short HTML main text through limitations. Source code link found; code not run or independently audited. No figure visual audit.',
           scope='CHI2026 workshop per arXiv comments; single Blender house/trees/cars task, not an indoor generation benchmark or general theorem.',
           consequence='Existing evidence that output symptoms need not identify code/state failure causes; diagnostic code intervention is not itself a learned solution. Our native registry failure remains separately measured evidence.'),
      dict(title='IL3D: A Large-Scale Indoor Layout Dataset for LLM-Driven3D Scene Generation',
           URL='https://arxiv.org/abs/2510.12095',project='https://wenxuzhou.github.io/project/IL3D/',
           dataset='https://huggingface.co/datasets/WenxuZhou/IL3D',
           read='Paper abstract, official project and dataset card; no new full-paper claim. Complete official sft_dataset.py read; targeted create_scene/coordinate/export paths in get_usda_rooms.py read.',
           actual_acquisition='Fixed first8lexicographic static layouts, HTTP-range ZIP extraction, CRC checked. One SSL EOF after3samples retained; same8continued with a bounded transport retry and fixed traffic cap. No full archive, HSSD,3D-FRONT geometry or model downloaded.',
           empirical_scope='All8are Synthetic Data / RecreationRoom by original tag.69instances,1has an ID matching a currently local HSSD mesh. No full description, rationale, tool trajectory or established training split in these raw8files.',
           source_coordinate_contract='Y-up and metres from official SFT prompt; original USD export multiplies translations by100. Do not silently mix with native Z-up.',
           consequence='Original static dataset evidence replaces speculation about availability, but does not supply the missing reason/action supervision or whole-room acceptance labels.'),
      dict(title='SCOTT: Self-Consistent Chain-of-Thought Distillation',
           URL='https://aclanthology.org/2023.acl-long.304/',
           read='This stage: primary abstract and method3.2 excerpt reconfirmed; earlier exact reading recorded separately. No new full implementation reproduction.',
           consequence='Counterfactual rationale-conditioning is existing work. Following a deliberately wrong rationale is not a desirable3D task-completion target.'),
      dict(title='Measuring Faithfulness in Chain-of-Thought Reasoning',
           URL='https://arxiv.org/abs/2307.13702',
           read='This stage: primary abstract and intervention overview only. No new full-paper claim or execution.',
           consequence='Reasoning perturbation tests are prior art; distinguish sensitivity from correctness and monitorability from full3D success.'),
      dict(title='ThinkBLOX public-data lookup',URL='https://arxiv.org/abs/2607.13539',
           read='Existing method reading retained from previous stage; this stage checked paper/project links and public dataset searches.',
           empirical_scope='HFsearches ThinkBLOX and ThinkBLOX-Data-200K returned empty lists at access time. This is not proof that no release exists elsewhere. No original ThinkBLOX reasoning rows acquired.')]
    dump(s/'reading_ledger.json',reading)
    theory='''三维决策学习：哪些作用可识别，哪些结构不应直接移植

一、训练理由不等于理由被使用
设X为原始任务、真实状态、历史及可用工具，D为生成的理由/决策，A为原生动作。
现有联合目标仍是标准加权CE：L_J= -sum_i w_A log pi(A_i|X_i,D_i) -lambda sum_{i:D原来存在} w_D log pi(D_i|X_i)。
AUX预测同样的D，但动作仅条件于X；J允许动作读取D；两者资料和权重相同。P拆成两次调用，其额外前缀、输出token和耗时必须入账。
本轮实际重算：24条动作组件有24个不同的序列化X；去除所有图像引用后，文字输入也有24个不同的哈希。
这不是基于图像文件名区分输入，但也没有声称运行了模型或比对其token/pixel隐藏激活。没有同一X下多个已验证的D/A组合。
因此，有限训练表可以被一个只按X查表的假想函数表示，也可以被读取X,D的函数表示。
这是可识别性反例，不是说实际8B模型已经记忆，也不是一个训练过的查表基线。

形式化例：X等概率取0/1，训练记录恒有D=X、A=X。
机制M1令A=D；机制M2令A=X。二者产生完全相同的训练联合分布，均可使H(A|D)=0、I(X;A|D)=0。
但干预do(D=1-X)后，M1改变动作，M2不改变。仅观察训练记录或teacher-forced损失无法区分。
这个常见的观测等价例不是新定理；它说明不能用更低损失或流畅理由，证明三维决策机制有效。
闭环使用自己生成的D并执行到整场，是必要的效果验证。对D进行诊断干预可以辅助理解，不能把故意错误的D当正确设计标签。

二、完整信息通过隐藏表示，与通过可见文字，不是同一个保证
普通因果注意力允许后面的D隐藏表示读取X，后面的A读取D隐藏表示。
屏蔽A直接读取X，仍保留X -> D_hidden -> A这一多层路径；本轮运行原版掩码函数验证了这一允许的拓扑。
这本来就是表示层面的中介路径，不是论文实现错误；但它不能单独证明可见D文字已完整表达或正确解释动作依据。
若要检验字面D是否足够，需在新上下文重新编码D，避免沿用携带X的KV/隐藏状态；还要明确有无其他输入路径。
当前P动作阶段仍读取原始X与D，因此不能宣称已实现这种严格文字瓶颈。
反过来，梯度限制前向不变时，也不能被当成部署时的条件独立约束。

三、三维动作不能为了理由可见性而失去真实状态
设两个实际几何状态G1/G2使用相同的抽象设计D，但可执行动作集合F(G1)和F(G2)互不相交。
如果动作器只读取相同D，它在两个状态下只能给出同一分布；两状态等概率时，该次动作的平均合法率不超过1/2。
证明：任一动作至多属于一个可行集合，对两个互斥集合的概率质量之和不超过1，再乘各状态1/2。
这一通用状态混叠论证没有证明我们的模型已发生该失败；它说明纯文字瓶颈可能丢掉必要的物体身份、尺寸和坐标。
保留q(A|G,C_hard,D)是一个有用的结构候选：G提供真实几何，C_hard保留不可违反的要求，D表达可调整的设计选择。
若希望只有某部分软意图U_soft经D影响动作，必须在新动作上下文显式排除U_soft的其他通道，同时避免C_hard或检索字段暗中携带它。
这种设计本质上接近已有分层规划器/控制器；不自动构成新颖性，也没有比普通规划+求解更好的保证。
当前没有实现或训练该结构，不把它当已完成方法。先保留原始状态可见的A/AUX/J/P及已有完整轨迹T作为强对照。

四、边界定义必须适配已有两阶段接口
本轮读取现有68个源组件，取24个paired动作组件；10个含前填D，14个没有D。
按动作target长度直接调用通用substring掩码时，10个前填D都位于prompt区域，A的查询行会被禁止直接读取它们。
special-token模式又会搜索整个输入；在显式含结束标记的合成格式提示下，24例出现prompt/answer区域重叠。
该论文实际实验采用substring模式，所以这不是其公开结果失效的结论。
这些是未适配边界的反例；本轮没有修改旧训练器，没有启动掩码训练，也没有执行模型来测量因果效果。
注意：预测首个A token用前一个位置的logit；本轮测的是查询区域和访问关系，不是假称实测了整段概率或梯度。
后续若真的比较掩码，须用已知的组件角色与真实前填边界，不凭输入中的标签猜测；没有D的14例不能被悄悄改成新增理由任务。

五、原始数据的可用范围
固定8个IL3D布局已经取得，但其原始标签全部为Synthetic Data/RecreationRoom，不是8个独立真人任务或8种房间。
八个原始布局共有69个物体实例；仅1个实例的资产ID与当前缓存匹配。未核验USDZ/GLB转换、朝向或功能等价，不宣称整场可回放。
原版SFT脚本从布局和外部assets.json描述构造JSON-only答案。原始8个布局没有完整用户任务、理由或操作轨迹。
因此不能把这些静态场景自动转称为已公开专家推理监督，更不能在无几何和人工/独立证据时全部标为设计成功。
保留固定选择和所有样本，不因为同属活动室或缺资产换样本；没有全量下载布局包、HSSD或3D-FRONT。

六、下一步可以支持的研究主张
仍检验同等资料和计算下，匹配的设计决策监督能否改善新房间完整生成，而非仅增加额外文字监督。
第一优先是把真实输入、理由来源、动作和执行结果对齐，保留普通工具、反馈、检索、联合reason-action及完整轨迹强基线。
不能仅追求“动作更听理由的话”；错误理由得到忠实执行可能使几何或用户要求更差。
需要同时报告原始任务完成、共同几何/功能有效性、误改、未知项与总成本。可靠工具反馈的普通修正对所有方法共享。
若AUX、J或T已解释收益，修订主张；若只有teacher-forced文本指标改善，不扩大三维效果结论。
本轮无新模型前向、梯度、训练或付费API；新增的是原始数据、真实组件支持分析及原版函数执行证据。研究目标仍未达成。
'''
    (s/'algorithm_theory.txt').write_text(theory)
    algorithm=dict(status='Empirical conditional-supervision hypothesis; no novel learned benefit established',
      goal='Complete editable indoor3D scene generation under the unchanged original task',
      original_controls='A action-only; AUX sameDindependent auxiliary target; J original joint reason-action; P two-pass generatedD; F ordinary retrieval/feedback/search; T existing full-trajectory learning',
      new_evidence=dict(actual_components=24,unique_observed_X=24,prefilled_D=10,no_D=14,
        original_static_layouts=8,object_instances=69,original_reasoning_trajectories_acquired=0,
        unique_text_only_X=support['unique_text_only_X'],
          generic_mask_changes_two_pass_semantics=True),
      theory_boundaries=['Finite observed likelihood does not identify causal use ofD.',
        'Representation-level mediation does not certify complete/correct visible reasoning.',
        'Blindly removing geometry may reduce achievable success under state aliasing.',
        'Dfaithfulness is not equivalent to correct original-task execution.'],
      concrete_candidate='Retain current direct-state action interfaces. A future role-typed q(A|G,C_hard,D) would require a normal hierarchical planner/controller control and fresh context. It is not implemented or claimed novel here.',
      implemented=['Pinned original mask-helper execution on archived component IDs, no model.',
        'Bounded fixed8public-layout acquisition with saved SSL failure and identical transport continuation.',
        'Original69instance input/asset/reference coverage table and exact24context support audit.'],
      not_implemented=['New information-flow training','New role-typed policy','Learned complete-room comparison','Exact geometry replay of8IL3D rooms'],
      model_forwards=0,training_updates=0,paid_API_calls=0,goal_achieved=False)
    dump(s/'algorithm_contract.json',algorithm)
    rows=[['author_layout_payloads',8,'fixed lexical first8, not sampled by quality'],
          ['raw_object_instances',69,'all8same Synthetic Data/RecreationRoom tag'],
          ['local_HSSD_ID_matches',audit['matching_available_local_HSSD_ID_instances'],'ID match only, not converted-geometry proof'],
          ['reasoning_trajectories_acquired',0,'static layouts are not trajectories'],
          ['actual_component_contexts',24,'24uniqueX; no matched-input intervention truth'],
          ['actual_prefilled_D_components',10,'14others unchanged and missingD'],
          ['generic_substring_mask_blocks_prefilled_D',10,'attention-query diagnostic, not learned effect'],
          ['special_mode_overlap_cases',24,'synthetic leading prompt; paper uses substring'],
          ['new_model_forwards',0,'no model loaded'],['new_training_updates',0,'no optimizer'],['paid_API_calls',0,'none']]
    with (s/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','value','scope']);w.writerows(rows)
    reproduce='''Run from REPOSITORY; all output paths must be fresh.

1. Reacquire the pinned small prior source into a local research cache (no install):
python scripts/autodl/fetch_pinned_reasoning_helpers.py --manifest runs/autodl/20261111_reasoning_identifiability/faithful_cot_source/targeted_source_manifest.json --out /dev/shm/faithful_cot_reproduction_NEW

2. Mask probe (CPU torch only, no model). Use a fresh stage containing the recorded faithful_cot_source manifest, and the exact original components under the repository:
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python scripts/autodl/probe_reasoning_mask_boundaries.py --repo . --stage NEW_STAGE --source /dev/shm/faithful_cot_reproduction_NEW/train/verl/utils/cot_masking.py

3. Static layout range reader: NEW_STAGE must contain public_data/il3d_minimal.json and il3d_tree.json copied unchanged. Never reuse existing results or replace preselected samples:
PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/read_public_layout_sample.py --stage NEW_STAGE
The saved first attempt had a terminal SSL EOF after3of8. Its same8transport continuation used:
PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/read_public_layout_sample.py --stage runs/autodl/20261111_reasoning_identifiability --resume-from runs/autodl/20261111_reasoning_identifiability/il3d_static_layout_sample_v1
This command already completed; do not rerun into that stage.

4. The input/support analyzer consumes the saved complete8sample receipt plus the original components and asset inventory:
PYTHONDONTWRITEBYTECODE=1 python scripts/autodl/audit_layout_schema_and_support.py --repo . --stage NEW_STAGE_WITH_SAVED_INPUTS

No command starts model inference, training, a paidAPI or a billed instance.
'''
    (s/'reproduce.txt').write_text(reproduce)
    links=['algorithm_theory.txt','algorithm_contract.json','reading_ledger.json','summary.csv',
           'actual_input_audit_v2/result.json','actual_input_audit_v2/empirical_support.json',
           'actual_input_audit_v2/sample_rooms.json','actual_input_audit_v2/sample_objects.csv',
           'mask_boundary_v1/result.json','faithful_cot_source/targeted_source_manifest.json',
           'il3d_static_layout_sample_v1/result.json','il3d_static_layout_sample_v2/result.json',
           'il3d_static_layout_sample_v2/SELECTION.json','TRANSPORT_CONTINUATION.json',
           'source_before_after.json','reproduce.txt','evidence.zip']
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    main='''<h1>三维决策监督：原始数据、边界与可识别性</h1>
<p>本轮完成了固定8个原始公开布局的有限读取、69个实例的来源核对、24个真实组件的输入支持分析，以及公开原版掩码函数的实际执行。没有训练或新增模型调用；没有把工程适配、可解释文字或静态布局说成完整生成成功。</p>
<h2>新增证据</h2><p>IL3D路径已跑通，没有全量下载。但固定8个布局均标为同类合成活动室，缺少原始完整用户任务及推理轨迹。单个当前HSSD资产ID匹配不等于可重放整场。SSL失败后的继续使用相同8个名字，未按内容替换。</p>
<p>现有24个动作组件的X全不同，故训练拟合不能识别模型是否使用D。实际执行公开原版掩码函数后，10个前填D在未适配动作分段时会被当作prompt而屏蔽。保留14个无D样本；未更改旧训练器或纸面提高分数。</p>
<h2>方法判断</h2><p>“训练模型依赖理由”已有直接前作；三维动作又必须读取真实几何。继续用A/AUX/J/P、普通工具反馈与完整轨迹学习作强对照；不盲目引入纯理由瓶颈，不把服从错误理由奖励成正确场景。未来若采用分层意图/几何接口，需要与普通分层规划控制器比较，目前没有训练这一结构或证明新颖性。</p>'''
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>三维决策学习与可识别性</title><style>body{max-width:1000px;margin:40px auto;padding:20px;font:16px/1.65 system-ui}pre{white-space:pre-wrap;background:#f6f6f6;padding:18px}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}a{overflow-wrap:anywhere}</style>'+main+'<h2>实际范围</h2><table><tr><th>指标</th><th>值</th><th>范围</th></tr>'+table+'</table><h2>推导与执行边界</h2><pre>'+html.escape(theory)+'</pre><h2>文件</h2><ul>'+''.join('<li><a href="'+x+'">'+x+'</a></li>' for x in links)+'</ul><p>一手来源：<a href="https://arxiv.org/html/2605.24286v1">Faithfulness as Information Flow</a>；<a href="https://arxiv.org/html/2603.26942v1">Observability Gap</a>；<a href="https://wenxuzhou.github.io/project/IL3D/">IL3D作者项目</a>。精确阅读与未验证事项见reading_ledger.json。</p></html>'
    (s/'stage_report.html').write_text(page)
    scripts=['read_public_layout_sample.py','probe_reasoning_mask_boundaries.py',
             'audit_layout_schema_and_support.py','fetch_pinned_reasoning_helpers.py',
             'report_reasoning_identifiability.py']
    bundle=s/'evidence.zip';files=[p for p in sorted(s.rglob('*'))
        if p.is_file() and p.name not in ('evidence.zip','delivery_verification.json')]
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(s)))
        for name in scripts:z.write(repo/'scripts/autodl'/name,'code/'+name)
    delivery=dict(files={str(p.relative_to(s)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
        scripts={n:sha(repo/'scripts/autodl'/n) for n in scripts},
        evidence_zip=dict(bytes=bundle.stat().st_size,sha256=sha(bundle)),
        original_input_hashes_unchanged=len(verified),
        delivered_hash_checks_not_model_quality_tests=True,
        data_free_GiB=shutil.disk_usage(s).free/2**30,goal_achieved=False,
        model_forwards=0,training_updates=0,paid_API_calls=0)
    dump(s/'delivery_verification.json',delivery)
    print(json.dumps(dict(stage=str(s),payloads=len(samples['samples']),
        source_inputs_unchanged=len(verified),evidence_zip_bytes=bundle.stat().st_size,
        data_free_GiB=delivery['data_free_GiB'],goal_achieved=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
