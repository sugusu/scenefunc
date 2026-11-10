"""Report actual source provenance and equal-supervision field controls."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main(repo,stage):
    controls=json.loads((stage/'field_controls_v1/result.json').read_text())
    boundary=json.loads((stage/'annotation_boundary_v1/result.json').read_text())
    presence=json.loads((stage/'source_information_presence.json').read_text())
    assert controls['fixed_units']==24 and controls['structured_sourced_D_components']==8
    assert controls['source_targets_and_coefficients_identical'] and controls['all_action_causal_token_boundaries_verified']
    assert boundary['recorded_native_user_text_matches_published_SFT_wrapper']==8
    assert presence['exact_principle_sentences_present']==8
    original=json.loads((repo/'runs/autodl/20261111_closed_loop_data/source_before_after.json').read_text())
    protected=[]
    for row in original['original_inputs']:
        current=sha(Path(row['path']));assert current==row['after']
        protected.append(dict(path=row['path'],before=row['after'],after=current,same=True))
    assert len(protected)==579
    external=json.loads((stage/'SceneDesigner_read_sources.json').read_text())
    for row in external['source_files']:
        assert sha(Path(row['local_internal_read_path']))==row['sha256']
    dump(stage/'source_before_after.json',dict(original_inputs=protected,original_new_author_source_unchanged=True))
    dump(stage/'NETWORK_SCOPE_CLARIFICATION.json',dict(
        protocol_field='new_model_or_API_calls refers to local/generative-model API calls, not all HTTP access',
        generative_model_calls=0,paid_model_API_calls=0,training_updates=0,
        actual_public_network_reads='GitHub commit/tree metadata, bounded original source files and primary web research were read',
        no_claim_of_zero_all_network_API_requests=True,no_credentials_submitted=True,
        no_gated_data_acquired=True,no_dataset_license_accepted=True))
    cases=[
        dict(decision_id='airbnp_open_clothing_access_for_short_stays',
             units=['clothing_build_07'],
             URL='https://position-concept.com/portfolio/airbnp-2/',
             source_kind='Architect-owned first-person project explanation',
             located='Full story, short-stay travelers and express clothing-use paragraph',
             supported='Short-stay visitors can use an open clothing rail for quick access, with other storage retained.',
             not_supported='The reconstructed asset ID, exact placement, load capacity or a correct native action trajectory.'),
        dict(decision_id='guilbault_consolidated_storage_for_continuous_living_space',
             units=['storage_build_07','storage_edit_01'],
             URL='https://www.archdaily.com/623445/30m2-flat-in-paris-richard-guilbault',
             source_kind='Project text explicitly provided by architect',
             located='Storage-wall paragraph and reorganization of living/private space',
             supported='A continuous integrated storage/service wall consolidates practical functions in a compact dwelling.',
             not_supported='Our three separate cabinet assets, their poses, integrated building services or final room usability.'),
        dict(decision_id='paper_table_mobility_for_overnight_guest',
             units=['paper_table_modes_build_08','paper_table_modes_build_09'],
             URL='https://www.nevertoosmall.com/post/how-two-graphic-designers-live-and-work-in-45sqm-484sqft-home',
             source_kind='Original project/homeowner interview reporting',
             located='Dining/living section, wheeled table and sleeper sofa',
             supported='The same dining/living zone changes for meals, hosting and overnight guests using a movable table.',
             not_supported='The reconstructed temporary/daily coordinates, valid continuous movement or our table/stool action sequence.'),
        dict(decision_id='waterloo_separable_tables_for_changing_activities',
             units=['waterloo_edit_02','waterloo_edit_03'],
             URL='https://www.nevertoosmall.com/post/waterloo-street-singapore',
             source_kind='Original designer/project reporting',
             located='Client flexibility requirement and two wheeled tables',
             supported='Clients wanted flexible arrangements; two wheeled tables can be joined or separated.',
             not_supported='Our object identities, joining coordinate0.12m, separate coordinate1.1m, seams, chair access or stability.'),
        dict(decision_id='michelet_adapt_family_lifecycle',units=['lifecycle_01'],
             URL='https://www.nevertoosmall.com/post/designing-small-spaces-for-families',
             source_kind='Original designer interview with described lived change',
             located='Michelet passage, eldest child leaving and former bedroom becoming office',
             supported='Home organization can change with family life stage, including a former child bedroom becoming an office.',
             not_supported='A demonstrated bed-removal tool action, our invented room geometry or a verified office outcome.')]
    assert sum(len(x['units']) for x in cases)==8
    dump(stage/'principle_source_review.json',dict(cases=cases,
        source_principles_checked=5,source_application_units=8,
        original_historical_designer_tool_trajectories_acquired=0,
        new_action_or_rationale_labels_added=0,
        scope='Current primary text review, not figure/plan measurement or whole-building reconstruction validation',
        access_failure_preserved='mobile-furniture overview timed out twice; the underlying Paper project interview was read directly instead. No failed source is treated as read.'))
    reading=[
        dict(title='Scene-Designer',URL='https://arxiv.org/html/2610.23121v1',
             scope='This turn: abstract, method3.1 and OTA constructionD.2 passages; not a new complete-paper or figure audit.',
             source_pin=external['source_pin'],
             source_read=['rewrite_cot_gpt.py:1-164 policy and review guidance;539-593 request construction;626-791 review/regeneration/apply functions',
                          'augment_sft_gpt.py:1-150 original thought prompt/API helpers, read only; API helper never run',
                          'sft_dataset.py:206-284 full original _build_messages',
                          'trajectory_compression/engine.py complete module',
                          'trajectory_compression/review.py:151-229 complete reviewer class; API/cache not invoked'],
             actual_execution='Original pure request/serializer function bodies on eight actual text-only floor turns, with24 fixed-input boundary checks. No model or full compression replay.',
             distinction='Annotation has the fixed target call as privilege; current/future results are excluded at this helper boundary. It is not a historical human designer thought log.',
             source_boundary='Current source pin unchanged from previously inspected release; no root license found. Newly fetched original files remain in RAM, not redistributed.',
             availability='README lists SFT checkpoint and held-out OTC benchmark. No OTA training samples acquired; no claim they cannot exist elsewhere.'),
        dict(title='SceneReVis',URL='https://arxiv.org/html/2602.09432v1',
             scope='Abstract/introduction, full method3.1-3.5, selected AlgorithmA.1 opening, appendixC.1-C.8 and promptsD.1-D.2. Not all reward appendix, experiments, figures or code reproduction.',
             distinction='Reverse construction and action-conditioned synthetic diagnostic thoughts already exist for scene generation; not original human chronological reasoning.',
             caution='Main experiment section states standard/generalization split300/100, while appendixC.1 states200/200. Discrepancy preserved; no effect magnitude inferred and no split copied into our plan.',
             no_reproduction='No weights, training, physical postprocessing or whole-room result reproduced.'),
        dict(title='SceneChain-12K public card',URL='https://huggingface.co/datasets/runder1/SceneChain-12K',
             scope='Public card and access terms only',
             status='Still gated with academic-use/no-redistribution/citation terms; no identity form submitted, no gated samples acquired.',
             distinction='Card describes reasoning/tool conversations; that description is not access to examples or a compatible native dataset.'),
        dict(title='Five sourced architecture principles',source='principle_source_review.json',
             scope='Original designer/project statements and original interviews reviewed at the cited passages, no full image/video or structural audit.',
             distinction='Supports design knowledge; local poses, IDs and operation demonstrations remain our reconstructions.'),
        dict(title='Book and earlier learning controls',source='../20261111_closed_loop_data/book_reading_ledger.json',
             scope='No new full-book claim in this turn. Prior explicit chapter8 reading and SDPO/OPSD source limits remain in force.',
             distinction='Field routing adds a control for the proposed information flow; no new training theorem claimed.')]
    dump(stage/'reading_ledger.json',reading)
    theory='''设计原则、操作计划与动作监督：可执行强对照及理论合同

1. 现在具体检验什么
任务保持为从文字生成完整、可编辑的室内三维场景。候选假设是：显式写出设计决定，再让动作读取它，能否改善原文要求完成或成本。
本轮把“设计决定”拆成C（当前条件）、R（来源原则）、P（操作意图和对象引用）、U（预期代价/未知）。不是给现有组件换一个新算法名。
新证据：8条结构化D都包含与动作参数完全相同的对象/资产引用；R原句、原则ID和该对象引用也都已在共同输入X中。
这些D没有typed numeric叶子，而对应动作有21个位置/朝向数值字段。故不能说D已复制整份坐标答案，也不能把提前选对象的收益当作原则推理收益。
现有24个单元仍全部保留，来自6个本地重建来源族。只有8条结构化来源原则应用，另2条为本地停止记录、14条缺D。
这不是24条建筑师原始工具轨迹，也不是新获得的完整场景正数据。

2. 本轮实际实现的三组
full_record：动作阶段读取整个自身生成的D。
without_principle_field：动作阶段读取C、P、U，移除D中的R字段。
plan_field_only：动作阶段只读取D中的P。
三个训练条件都保留完整D预测损失、相同动作标签和每个原始token系数。没有给其中一组增补专家文字。
来源知识K仍在公共X中；移除的是额外决策前缀里的字段，不是剥夺某一组的知识访问。
以后须报告真实生成D的schema覆盖率；若大多进入共同透传，三路并未形成有效信息干预，不能据此判定原则无用。所有方法共用格式提示，不给某一路免费修复或重试。
路由函数只收D字符串与mode，不接动作答案、场景ID或语料查询。非此schema、无D和无效JSON在三组中共同原样通过并记录，不补字段、不重试。
本轮已对全部24个单元完成每组34个组件的文本/token构造，总102组件，原标签和目标质量不因路由改变。
三组完整原监督权重质量均为1.8333333333333335；动作文本前缀token合计分别245416、244948、244474。
这不是严格等计算量：前缀长度确有差异，以后必须计输入编码、生成及工具全过程。没有测量显存、模型行为或学习收益。
两条其他schema的停止D三组共同保留；14条缺D不补成新理由。不能为了让某个路由好看而删除这些状态。

3. 普通目标与真正的区别
设g_m为三种路由，普通训练目标为
 L_m = E[-sum_k wD_k log pi(D_k|X,D_<k) - sum_j wA_j log pi(A_j|X,g_m(D),A_<j)]。
差异仅在动作的可读前缀，不是新的损失；全D训练质量、来源、原场景、可用工具和动作标签相同。
这比“有CoT vs 没CoT”更有判别力：若完整D和仅计划相当，现有收益可能无需显式原则字段；若无原则字段就退化，再检查是否为来源利用、格式、长度或注意力差异。
若完整D优于计划，仍不能由这一个对照直接证明因果忠实或新颖性，因为C/U也会变；without_principle_field用于缩小差异。
即便三个都不优于普通反馈或搜索，也不是三维智能体整体无效；应据实调整这个机制的投入和贡献主张。

4. 标注看到动作，和部署看到动作，是两件事
已执行原作者build_turn_request和SFT _build_messages纯函数，使用8条真实floor记录映射到对应样本字段。
标注器获得固定目标调用；策略用户侧只获得原文、动作前memory。24次标记注入检查中，当前结果参数、其他thought、未来turn均未改变固定前态下的请求。
原SFT with/without thinking保留相同目标调用；8例Task/Memory原文也与实际floor用户文本相符。
这没有运行标注模型、审核模型、完整memory重建或多模态标注；不以8个纯函数样例证明整套数据流水线正确。
Scene-Designer与SceneReVis都已有根据目标动作生成理由的构造。目标已知的标注本身是合法监督方法，不自动等于评测泄漏或有害数据。
但它近似q(D|X,A*)，不能据此宣称D是原设计师当时的真实心理过程。只有测试阶段把金标准D/A*输给策略，才会混淆部署可用信息。
不得用金标准D前缀下动作似然改善作为闭环生成改善。评测必须保存真正生成的D和之后真实工具轨迹。

5. 能证明与不能证明的信息论结论
把外部正确目标记作Y，模型自己的动作记作A_model，不能混为同一个随机变量。
若D_gen=f(X,U)，且随机数U在给定X后独立于外部Y，则I(Y;D_gen|X)=0。
所以生成解释并不凭空增加关于外部真值的信息；有限网络和有限计算仍可能因为显式中间计算而更好地使用X。
这不是说I(A_model;D_gen|X)=0：动作本来就条件于D，两者可以相关；也不是说解释一定无用。
训练时D可能是目标Y或A*的函数，上述独立假设不成立，因而teacher-forced训练条件更容易。
现有24个上下文几乎不重复，不能从这份小表实证估出条件互信息，亦不能把训练拟合当作“原则被用到了”。
此处均为普通条件概率/数据处理边界，不提出新的三维最优性或泛化保证。

6. 五个原始案例支持什么
POSITION自己的项目说明明确解释短期旅客的快速衣物使用；Richard Guilbault提供的项目文字说明整合收纳/技术需求。
Paper住户项目采访说明可移动餐桌与留宿转换；Waterloo原报道说明按客户灵活需求使用可分合桌。
Michelet设计师采访与原项目报道说明家庭变化后卧室转办公。这些来源支持原则，不支持我们选择的本地坐标、资产ID、动作次序或全屋功能通过。
所有5个原则及8条应用的逐项支持与不支持内容已分别保存。没有改旧标签，也没有把公开报道当作给我们重建场景的独立用户评分。

7. 与最近邻的明确边界
Scene-Designer已有OTA、动作理由、轨迹筛选、Critic和分支/进度奖励。公开工程的压缩引擎声明offline recorded-state/arithmetic验证，不能把读源码写成已运行Drake物理重放。
SceneReVis已有图像/图/历史输入、诊断CoT、反向构造轨迹及多阶段奖励，覆盖从空房和缺项修复。不能声称首次把理由、反馈或重做操作用于完整场景。
SceneChain公开卡仍有访问表单；没有取得样本。当前Scene-Designer源码pin未更新，README没有提供已取得的OTA训练文件；不能拿仅供评测的OTC-Bench补训练。
此前SDPO反馈蒸馏、LSS表示监督、RED推理组合和完整轨迹训练仍是已有强对照。新字段路由只帮助识别额外解释字段的价值，不承担创新主张。

8. 后续实际比较的算法步骤
先在相同模型、原始任务、当前真实多视图和知识K下生成一次D，保存原始输出及完整成本。
对同一父状态的full/noR/plan路径分别重建动作上下文，按同等累计限制生成动作，再由同一原生执行/检查器验证。
分支复制必须含已经验证的资产文件、注册表及原调用历史；不能只复制SceneJSON，也不把真实序列防重复历史清零。
这一冻结模型路由诊断只判断信息通路。随后各训练条件分别用自己生成的D走完整轨迹，不能共享金标准D当端到端成绩。
计划/原则均不保证执行正确。必需实例、语义身份、空间关系、保留要求、几何、未知项和总成本分别报告；超时、拒绝与中途失败留在分母。
8个旧原理应用与24个开发单元只能测试机制。泛化必须按新房间/资产和设计来源族隔离，不能同一项目换坐标算未见设计。
ID置换若用于消除记名捷径，还必须同步重做视图上的ID标签；只改文本而保留旧图像会制造矛盾输入，本轮不构造这种伪控制。

9. 下一步不能跳过的实际接入边界
旧PhasePolicy/NativeRoomPort当前是5视图家具阶段适配，单tool_call分类器；原生整场轨迹还含0图floor、其他阶段与批量调用。
本轮字段路由尚未把这一限制变成完整多阶段执行器，不宣称所有完整生成对照已经准备好。
下一步可在CPU用已保存的19条真实策略输出核对并接通多阶段/批量动作合同，不重新修289/404坐标，不伪造Agent提议，不启动新模型或工具生成。
模型闭环比较还需新的明确调用额度、实际前后向显存与工具角色隔离证据。旧关闭的生成额度不重开；本轮无模型、训练或计费资源启动。

10. 当前选择
继续保留“如何学习、使用条件化设计决定”这一待验证问题，并把普通操作计划作为必须击败的强对照。
当前不把它宣布为CVPR主创新，也不因缺少现成专家轨迹而要求整条研究停掉。来源原则、合成理由和实际执行质量分开记录，逐项补充能改变判断的证据。
'''
    (stage/'algorithm_and_theory.txt').write_text(theory)
    contract=dict(status='implemented_control_not_validated_research_method',
        question='Does explicit state-dependent explanatory conditioning improve complete indoor generation beyond the same learned plan and the same source knowledge?',
        fixed_source_units=24,families=6,structured_source_D=8,source_principle_families=5,
        controls=['full_record','without_principle_field','plan_field_only'],
        equal_supervision=['Full original D targets and weights','Exact original action targets and weights',
                           'Same pre-action messages/source knowledge and assets','No new expert text'],
        not_equal_compute='Action input prefix lengths differ; all actual generation/encoding/tool time must be charged',
        source_field_routing_only=True,principal_knowledge_remains_in_common_X=True,
        deployment='Only actual own-generated D; no action answer/corpus lookup in router; common pass-through status logged for nonmatching schemas',
        outcome_required='Free-running complete native rooms on isolated new source/room/asset families, with unknown/failure retained',
        unproven=['Learned policy gain','New algorithm/theory','Teacher action correctness','Full-room successful candidate rollout'],
        next_safe_work='CPU port of actual native multistage/batch output contracts; do not reopen closed model budget',
        protected_old_files=579,new_paid_model_API_calls=0,new_VLM_forwards=0,new_training_updates=0,
        new_complete_positive_scenes=0,goal_complete=False)
    dump(stage/'algorithm_contract.json',contract)
    rows=[('original_units_retained',24,'six reconstructed source families, not24full rooms'),
          ('structured_design_records',8,'all checked, five source principles'),
          ('unchanged_other_schema_D',2,'local stopping records, same across routes'),
          ('missing_D_units_retained',14,'not completed with invented reasons'),
          ('selected_ID_revealed_in_D',8,'does not reveal all spatial parameters'),
          ('principle_sentence_already_in_X',8,'common knowledge access retained'),
          ('action_pose_numeric_fields',21,'not copied as typed numeric D leaves'),
          ('same_supervised_components_per_arm',34,'complete original D and action targets'),
          ('prospective_components_total',102,'text material, not102inference trials'),
          ('source_boundary_floor_turns',8,'actual saved text-only observations mapped to original helper'),
          ('original_protected_files_unchanged',579,'actual SHA256 recomputation'),
          ('local_VLM_forwards',0,'tokenizer and pure-source helpers only'),
          ('new_training_updates',0,'no optimizer'),('new_complete_positive_scenes',0,'no generation run')]
    with (stage/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['metric','value','scope']);w.writerows(rows)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)
    sources=''.join('<li><a href="'+r['URL']+'">'+html.escape(r['decision_id'])+'</a></li>' for r in cases)
    links=['algorithm_and_theory.txt','algorithm_contract.json','reading_ledger.json','principle_source_review.json',
           'source_information_presence.json','summary.csv','field_controls_v1/decision_information.csv',
           'field_controls_v1/prospective_field_components.jsonl','annotation_boundary_v1/result.json',
           'SceneDesigner_read_sources.json','source_before_after.json']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>设计原则与操作计划的强对照</title>
<style>body{max-width:1100px;margin:35px auto;padding:20px;font:16px/1.65 system-ui}pre{white-space:pre-wrap;background:#f5f5f5;padding:20px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:7px}a{overflow-wrap:anywhere}</style>
<h1>原则、计划、动作：把需要证明的收益分开</h1>
<p>本轮实际核查五个原始项目来源，运行原作者标注请求/训练序列化纯函数，并为全部24个开发单元构造三组相同监督的字段对照。没有模型、训练、新完整场景或CVPR成功声明。</p>
<p>8条来源支持的结构化D都显式选择了动作对象，而且原则原句已在公共输入中。仅有“带理由优于动作”无法区分原则应用与先选对象的帮助。现在完整记录、去原则字段、仅计划三组都保留完整D监督与相同动作标签。</p>
<p>原作者标注器看到目标动作，当前/未来结果在所测请求构造边界被排除。这是监督构造，不是评测自动泄漏，也不是原设计师历史思维日志。训练与部署必须区分。</p>
<h2>实际交付范围</h2><table><tr><th>指标</th><th>值</th><th>范围</th></tr>'''+table+'''</table><h2>原始案例来源</h2><ul>'''+sources+'''</ul>
<h2>算法与理论</h2><pre>'''+html.escape(theory)+'''</pre><h2>近邻及交付文件</h2>
<p><a href="https://arxiv.org/html/2610.23121v1">Scene-Designer</a> · <a href="https://arxiv.org/html/2602.09432v1">SceneReVis</a> · <a href="https://huggingface.co/datasets/runder1/SceneChain-12K">SceneChain公开数据卡</a>。准确阅读、访问限制和未复现事项见账本。</p><ul>'''+''.join('<li><a href="'+p+'">'+p+'</a></li>' for p in links)+'''</ul>
<p>新读取的作者原始源码留在RAM，不随证据包分发；公开URL、revision与SHA256已保存。全程旧文件只读，无新模型服务API/训练/资源启动；公开GitHub元数据与网页确实进行了网络读取。</p></html>'''
    (stage/'stage_report.html').write_text(page)
    script_names=['decision_field_routes.py','audit_design_record_information.py',
                  'probe_original_annotation_boundary.py','report_decision_source_value.py']
    code=[]
    for name in script_names:
        p=repo/'scripts/autodl'/name;compile(p.read_text(),str(p),'exec')
        code.append(dict(path=str(p.relative_to(repo)),sha256=sha(p),bytes=p.stat().st_size))
    dump(stage/'code_manifest.json',dict(our_code=code,original_author_source_hashes_only=True,
                                       no_weights_or_original_gated_data_in_bundle=True))
    files=[p for p in stage.rglob('*') if p.is_file() and p.name not in {'manifest.json','evidence.zip','delivery_verification.json'}]
    with zipfile.ZipFile(stage/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(stage)))
        for r in code:z.write(repo/r['path'],r['path'])
    manifest=[dict(path=str(p.relative_to(stage)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files+[stage/'evidence.zip'])]
    dump(stage/'manifest.json',dict(files=manifest,protected_source_files=579,original_author_source_not_redistributed=True))
    print(json.dumps(dict(stage_report=str(stage/'stage_report.html'),summary=str(stage/'summary.csv'),
                          evidence=str(stage/'evidence.zip'),bundle_bytes=(stage/'evidence.zip').stat().st_size,
                          goal_complete=False),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True);a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
