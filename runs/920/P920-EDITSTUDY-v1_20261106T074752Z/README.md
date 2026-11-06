# 家具齐备房间中的专业决定学习试验

本轮检验：在同动作、同知识与相同工具预算下，增加有出处的条件化决定监督，能否改善完整房间的用途调整，并减少对其他用途的损害。用户在具体六 GPU 小时方案后明确要求继续；授权解释与边界见 AUTHORIZATION.json。旧四更新授权不复用。

当前快照 2026-11-06T08:17:11.462355+00:00：动作组实际完成 2/8 次完整批更新，训练继续。真实进程回执与微批进度在服务器 runtime 同名目录的 training 下；评测队列等待两组验收。此快照不是训练结束或设计收益报告。

## 数据与方法

仅用 Guilbault、AIR、Paper、Waterloo、Michelet、F-House 六家族。每组22个已有家具场景中的编辑步骤（6添加、15移动、1移除）；其中7个位置含决定监督，6条独特来源原则，15个位置只监督动作。来源原则与研究者编写的状态映射、操作及代价说明明确区分，均不声称历史软件轨迹或整屋专业金标准。

两组每次更新相同1245个动作目标token。联合组额外1094个决定token；按家族分别归一化动作CE和决定CE，决定系数1，额外监督与计算不声称等量。官方OTA思考/工具序列保留；推理时三组均可自行规划，不输入开发项目标准理由。动作组没有金标准理由前缀，联合组在7个位置学习自己产生决定再行动。该三组比较检验增加决定监督的收益，不能独自证明配对机制。

44个实际样本均通过官方加载器，五图/动作/共同输入核对通过。真实Trainer缩放CPU检查两组误差均<4e-8，无优化器或反向；真实训练使用固定8B底座、语言q/v LoRA r8 alpha16、3,833,856参数、BF16/SDPA、36层检查点输入CPU暂存、Adafactor2e-5、种子42。

## 固定预算与评价

两个adapter各最多8次完整数据更新、各最多7200秒，仅GPU0；已有代码每次更新保存完整恢复状态。首基础设施错误停止整批，不自动重试/扩训。最终第8次适配器用于评价，不根据开发结果挑检查点。大权重、张量、mesh、checkpoint和GLB只留服务器；Git记录路径和哈希。

Pine、Milan均为已暴露开发留出，已从本轮训练和知识池排除。三组：原底座+同知识+充分提示+反馈、动作LoRA、决定+动作LoRA。2任务×2种子(17/29)×3组=12终态；种子不是独立项目。共同原始需求、目录、五视图、工具、反馈；每份12模型/工具尝试，每次1024新token、输入18432、总12288新token。真实采样temperature0.7/top_p0.9/top_k50，逐轮seed=单元seed+轮号。

每份模型子进程累计最多600秒，包含加载/编码/生成/保存；十二份最多7200秒。CPU渲染/检查另记。语法/schema拒绝消耗原预算并返回错误，不偷偷重抽；基础设施错误停批。保留最后真实状态、动作、五图，导出带对象ID的GLB。

初态检查已显示：Pine缺第二同时工作位，既有参考重建通过静态代理；Milan原餐桌已能留材料并提供两个餐位，不能把初态成功算作模型收益。Milan用于检查保留可行用途与避免不必要修改，不能充作第二个需要改善的工作面缺口。初态已有的通路问题与新损失分开。候选表面允许共桌；有限网格、座位方形和通行均为研究代理，未找到组合不能证明所有替代方案不可用。真实舒适、客睡、设施、专业盲评仍未知。

## 运行与恢复

所有运行已实际启动，不要复制命令重复开作业。训练会话79563，顺序评测会话27562（会话号仅当前客户端有效）；应先核进程、receipt与checkpoint。

```bash
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/run_edit_training_batch.py --plan runs/920/P920-EDITSTUDY-v1_20261106T074752Z/TRAIN_PLAN.json --authorization runs/920/P920-EDITSTUDY-v1_20261106T074752Z/AUTHORIZATION.json
/data/wyh/conda/envs/c15_genrec/bin/python scripts/920/run_edit_evaluation_batch.py --study runs/920/P920-EDITSTUDY-v1_20261106T074752Z
```

恢复只从实际存在的记录判断：旧输出目录拒绝覆盖，断网不重启实验；训练中断保留已完成checkpoint，不自动恢复优化。评测具体命令/模板见 EVALUATION_COMMAND_MATRIX.json，代码/输入锁见 EVALUATION_EXECUTION_LOCK.json；模板仅在最终适配器验收后绑定其文件哈希，不改变任务或预算。

主要文件：DATASET.json、KNOWLEDGE_POOL.json、SPLIT.json、TASKS.json、ENCODING.json、TRAIN_PLAN.json、PREFLIGHT_RESULT.json、TRAINING_RESULT.json、EVALUATION_BATCH.json、STATUS.json。小型训练快照及阶段真实回执在 recorded/；快照之后进度以服务器实际记录为准。
