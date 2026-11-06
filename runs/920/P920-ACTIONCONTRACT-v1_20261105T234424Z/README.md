# 共同动作接口：CPU 验收，尚未接入模型回合

本批为后续公平学习对照补共同执行边界，不新增专业知识、决定样本或训练。原 OFFLOAD 四更新已完成，本批 **0 模型调用、0 场景编辑、0 优化更新**。所有作业已终态。

旧米兰会话只检查工具块闭合，允许 schema 外参数进入原生执行器。新 `shared_action_contract.py` 是独立、显式启用的组件；历史执行器、输入、输出和检查点均未改。它直接读取观察里五个真实工具的 schema，并把引用限制为当前家具 ID／目录资产 ID；不提供目标坐标、不剪裁坐标、不做布局优化，不依据参考答案筛选方案。

## 已核结果

| 检查 | 实际结果 |
|---|---|
| 参数、重复键、额外字段、非有限数、部分／多条输出等 | 22/22 符合预期 |
| 状态变化后使用旧 contract | 拒绝；必须由新观察重建 |
| 空目录／空家具 | 查询与显式停止仍可用 |
| Scene-Builder 真实 tokenizer 的强制前缀走查 | 7 个正例、8 个反例均符合预期 |
| 历史米兰输出的只读分类 | 第 0、5、6 次拒绝，第 1–4 次通过 |

历史第 1 次删除需保留的书架仍会通过，第 4 次造成几何问题的添加也会通过。这恰好表明 schema 有效与设计正确不同。这里没有重新执行历史动作，也不能推断启用新接口后的米兰轨迹会怎样。

依赖下载 13.517160 秒、参数检查 0.417003 秒、tokenizer 检查 12.861326 秒，均实际 exit 0。精确启动、首日志、命令与结束记录在 `execution/*/receipt.json`；摘要见 `SUMMARY.json`，完整分类见 `guard/GUARD_CHECKS.json`、`tokenizer/TOKENIZER_CHECKS.json`。

## 下一执行器的接入约定

1. 从本步实际观察取工具 schema、完整当前对象／资产表；创建新的 `ActionContract`。缓存 tokenizer trie，但不复用上一步的 parser。
2. 各组使用相同的新输出协议：单一 JSON `{"name": ..., "arguments": ...}`。停止为 `{"name":"__done__","arguments":{"unmet_or_unknown":"..."}}`，不能当工具执行或自动判成功。必须在提示中说明；不能直接沿用旧工具标签要求。
3. 把 `contract.prefix_function(tokenizer_data)` 传入 Transformers `generate(prefix_allowed_tokens_fn=...)`。保持原始需求、五视图和统一 token／工具预算。保存原始 token、输出和过滤成本；实际接入尚未运行。
4. 检查生成是否完整，按明确的 token ID 处理 EOS，不用文本猜测补全。不完整即保留并失败，不强行补括号。执行前调用 `contract.validate(raw_json, current_context)`；状态已变则拒绝，不改写动作。拒绝计入预算，无隐含重试。
5. 只有 `kind=tool` 才提交其原参数给原生工具。原生权限、不可变对象、朝向参数语义仍由实际工具检查；任务中的身份要求、几何、用途及专业品质独立计分。原生回执与后续观察照常保留。

接口支持同组内的工具名／参数关联和当前 ID 限制，不意味着任意 native 输入一定成功。原生 `direction` schema 仅声明 string，具体合法值仍由工具检查。JSON 语法约束也不能保证有限 token 上限内一定闭合，不能消除重复查询、坏布局或虚假自评。停止说明的真实性／充分性没有在此检验。

默认 tokenizer adapter 仅允许 EOS 151645；模型配置还列有 151643，后者属于被排除的 special token。此次 151669 词元 trie 和走查不加载模型权重，不验证 GPU `generate` 或真实场景闭环。因此状态是 **CPU 接口可用，尚非强基线整屋结果**。新旧输出协议不同，跨历史批次不作严格因果比较。后续各训练／冻结基线需共享它，收益不能记为配对学习贡献。

## 依赖与复现

使用 [LM Format Enforcer 官方 Transformers 接口](https://github.com/noamgat/lm-format-enforcer) 的固定 0.11.3 wheel，另有 jsonschema 4.23.0。依赖仅装入本次 runtime 的 `dependencies/`；原 conda/native 环境未安装或升级。14 包的准确版本和 wheel 哈希见 `requirements.lock.txt`、`DEPENDENCIES.json`；服务器保存完整 pip resolution report 与 381 文件哈希。可在同平台新目录执行 `pip install --only-binary=:all: --require-hashes --target NEW_DIR -r requirements.lock.txt`，再按回执命令把 PYTHONPATH 指向新目录。不得覆盖历史运行目录。

pip 保留了既有环境 CUDA 包元数据警告；本批 CPU 导入与检查成功，未据此修理旧环境，也未验证新 overlay 的 GPU 兼容性。基础模型及 tokenizer 哈希沿用米兰已冻结版本，本批仅读取 tokenizer/config。

本批基线为 `runs/920` 的 `9f75c1891fd795a4190e4174c590cfd59518dc64`。当前修改留本地，按每小时合并更新规则与下一实质批一起提交推送；不是已发布的新 commit。没有后台模型作业。
