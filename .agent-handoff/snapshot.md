# 当前研究状态

更新 2026-11-05T15:27:28.122674+00:00。总体 Goal active；OFFLOAD获批4更新已消耗，0新训练授权。上一轮SBUILD实质完成并推送3a85da931fda4ec38ef4044f893ae67a7d385394，远端精确核验；该轮分类progress。

## 活跃工作：必须继续同一进程

- P920-KBASE-v1_20261105T151743Z 正在运行两组冻结底座开发推理，执行session82694、batch PID 1323949、start_ticks 54981566；真实启动 2026-11-05T15:19:27.078519+00:00。最新已记录 strong 7模型/6工具；knowledge尚未开始生成。每次继续先查此handle或/proc身份和真实receipt，不能因观察超时重启。
- strong=完整需求/预算规划提示；knowledge=同提示+OFFLOAD三来源全部10段原理由。所有来源仍train，Casa是已暴露development。无当前项目参考坐标/布局/评分。完整固定知识库不是检索排序。
- 每组24模型/24工具、1024输出、五图65536pixels、GPU0、BF16/SDPA/greedy、0训练，首执行错误停止。输入上限共同16384（旧三组12288，背景比较不称严格同计算）。prepare均退出0，初始状态和5图相同，token9250/9998；已出现首个真实模型输出及实际回执。
- strong前5次已完成动作均重复get_current_scene_state带空键ignored，引擎接受但schema异常；未建家具。不能提前改提示/预算或改为成功。冻结driver将顺序完成两组及原审计、导出、九用途报告、schema和冷恢复。
- 当前来源计划runs/920/P920-KBASE-v1_20261105T151743Z/PLAN.json；持久运行../runtime/P920-KBASE-v1_20261105T151743Z/BATCH.json、batch_execution/receipt.json及各arm。最新视图/模型输出按各round目录读取。运行中脚本与plan不得改。

## 待本轮结束

1. 监看同一session82694，按实际终态核对两组及后处理。任何失败保留，不重启生成；后处理错误只能另标修复。
2. 各组终态5图审阅，归档小日志/每步证据及模型成本，和旧三组作有边界的开发比较。保留unknown/缺项/重复查询。
3. 已新增research/920/paper_v1的方法工作稿、6条主张证据和已完成pilot结果；KBASE未结束不填新结果。当前草稿不支持学习/泛化主张，不是论文成果验收完成。
4. 本轮完成关键节点再push/ls-remote。首输出发布以../preflight/publication_kbase_first_20261105/PUBLICATION.json为准；若不存在尚未发布。

## 已完成证据和后续主线

- SBUILD：6固定设施+9添加、50图、1复用来源理由，585目标token/组，0训练，15件末态。全部来源/动作/图/量测/失败和恢复已公开；代理较多，不证明整屋使用。
- OFFLOAD：两组各2更新80总反向、保存重载通过，训练周期loss略升。Casa三组24动作后家具7/6/6，各2接触且缺必要角色，未有学习收益证据。Casa事后参考12角色通过声明几何，不能当模型/专业示范。
- 先补强非训练对照与其他训练来源建设覆盖；任何新训练另审。封存测试、独立专业评审、充分项目数仍不足。保持来源主线，不扩Swiss或重复加载器检查。

## 工作区

runs/920；仅实验副本专用SSH，保护main/AutoDL，不force。未跟踪scripts/920/extract_swiss_development_input.py不碰。原生检查点/GLB/权重留服务器。
