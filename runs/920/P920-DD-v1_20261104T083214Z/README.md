# P920-DD-v1：920 同起点 G/U 对照结果

**本轮没有观察到一次用途反馈修复关键用途问题。** A-G、A-U、B-U 都请求了当前执行器不支持的 `check_physics`，场景保持原状。B-G 删除了餐桌、两把餐椅、床头柜和衣柜，相交对从 7 降到 1，但用餐配置受损，不能算设计质量提升。B-U 保留了家具，却未修复用途冲突；本轮不支持用途反馈或学习方法优越性的结论。

从可核验的两份历史未修订草稿开始，已完成 **4 次真实模型修订、9 次工具调用（5 成功、4 失败）、6 份场景评分**。没有重抽正式回合、训练或付费 API 调用。原输出、动作、日志、实际退出状态均保留。

## 主结果与交叉验收

G 为普通几何反馈；U 仅在相同 G 输入后追加当前场景实测的桌面用途与遮挡信息。A 允许饭前收起材料；B 要求材料保持原位展开。

| 场景 | 家具数 | 相交对 | 按 A 需求检查桌面 | 按 B 需求检查桌面 |
|---|---:|---:|---|---|
| A 草稿 | 9 | 7 | 无需收起材料的联合见证 | 联合见证 |
| A-G | 9 | 7 | 同上 | 同上 |
| A-U | 9 | 7 | 同上 | 同上 |
| B 草稿 | 10 | 7 | 分时台面有见证；收纳转换未知 | 有限搜索未找到联合见证 |
| B-G | 5 | 1 | 分时台面有见证；收纳转换未知 | 有限搜索未找到联合见证 |
| B-U | 10 | 7 | 分时台面有见证；收纳转换未知 | 有限搜索未找到联合见证 |

G、U 各自两行两列交叉表见 [RESULTS.json](RESULTS.json) 的 `cross_requirement_surface_matrices`。工作为 80×50 cm，用餐为 60×40 cm，允许同桌或两桌。按实际网格顶面及同高度遮挡，以 1 cm 网格、0°/90°、最多 100000 对枚举。正见证坐标另行对实际网格复核。`not_found` 不代表连续空间全局不可行。

**这些只是零厚度台面几何证据，六份场景的整屋 A/B 成功均为 unknown。** 坐姿、椅子拉出、到达路径、睡眠出入、门扇运动、窗景、收纳容量/状态切换、可用活动空间尚未验收。家具存在和工具成功不等于功能合格。相同 A 布局能满足两题的台面条件，因此如实保留。

B-G 消失的 6 对相交全部涉及删除物体；另一次删除不存在的 `bookcase_0` 被拒绝。本轮没有新的资产检索或替换。主要失败涉及不支持的工具、错误对象 ID 和损害功能的删物体策略，不能简单归因于几何不可行。

## 成本与产物

| 回合 | 输入 token | 输出 token | 生成秒数 | 工具成功/尝试 |
|---|---:|---:|---:|---:|
| A-G | 6857 | 249 | 10.32 | 0/1 |
| A-U | 7440 | 380 | 14.92 | 0/1 |
| B-G | 7443 | 291 | 12.09 | 5/6 |
| B-U | 7938 | 284 | 11.86 | 0/1 |

四次正式模型进程均退出 0，共 1204 输出 token、49.20 秒生成时间。单张已授权 RTX 3090 GPU0，FP16、batch 1、seed 0、原生 SDPA、KV cache，不量化。峰值已分配约 18.25 GiB，最大保留约 18.81 GiB。统一上限为输入 8192、输出 2048、每次 20 分钟、最多 12 个工具调用；实际输出未打满 2048，完整上限的余量依据实测加 KV 估算，未宣称穷尽测试。

工程尝试另计 3 次：首次输出保存因 ASCII 编码失败；同输入 UTF-8 重试成功；最长五视图 B-U 的 32-token 预检成功。工程输出未用于场景动作。两次用途测量因遗漏 NetworkX 失败，补齐后只重做确定性测量，未增加模型回合。

| 场景 | 俯视图 | 量测 |
|---|---|---|
| A 草稿 | [查看](drafts/A/topdown.png) | [JSON](drafts/A/use_measurement.json) |
| B 草稿 | [查看](drafts/B/topdown.png) | [JSON](drafts/B/use_measurement.json) |
| A-G | [查看](revisions/A_G/topdown.png) | [JSON](revisions/A_G/use_measurement.json) |
| A-U | [查看](revisions/A_U/topdown.png) | [JSON](revisions/A_U/use_measurement.json) |
| B-G | [查看](revisions/B_G/topdown.png) | [JSON](revisions/B_G/use_measurement.json) |
| B-U | [查看](revisions/B_U/topdown.png) | [JSON](revisions/B_U/use_measurement.json) |

GLB、checkpoint 的本机绝对路径、大小、哈希和恢复方式见 [ARTIFACTS.json](ARTIFACTS.json)，文件位于 `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-DD-v1_20261104T083214Z/`。权重、网格和完整 checkpoint 不在 Git。HSSD 来源与许可见仓库 `paris/NOTICE.txt` 与 `paris/licenses/`；俯视图用于本非商业研究记录。

## 基线和差异

基线为已合并 AutoDL 交付 `bdabc219772112ff2aee0b4a14ffd86809f48cb1`，见 [BASELINE.json](BASELINE.json)。原 `paris/` 历史文件保持未改动。草稿来自 `A_feedback_turn`、`B_feedback_turn`，不是已做网格反馈的 `*_mesh_feedback_sdpa_kv`。920 重下载原始网格并重构，核对对象姿态/边界和重新计算的 FCL，见 [DRAFT_REUSE_ACCEPTANCE.json](DRAFT_REUSE_ACCEPTANCE.json)。这是历史草稿本机重构，四份修订才是本轮新模型输出。

模型为 `XuanyuLiu/Scene-Builder-Qwen3-VL-8B-Instruct@63e235a0c0d6e5bc2f8f3485214df4457fd2e672`。12 个已批准文件共 17,541,458,531 字节，四分片完整哈希通过；原仓库 11 项权重锁全部匹配，额外文件仅为同 revision 的 385 字节视频处理配置。原需求卡 PNG 哈希为 `7dfc6789ce8c225b179020104ece86393d8e7942dc1821a37acbe4c9fdddcf0c`。

用户配置 HSSD 授权后，23 件选中网格及同 ID 坐标文件共 62,522,166 字节全部通过哈希验证。这是选中集合回放，未复现 436 候选排名。一个官方家具工具文件与 AutoDL 锁记录不一致，但两个独立 GitHub 读取结果一致；使用单独的 `upstream.920.lock.json`，原锁保留。差异见 [UPSTREAM_LOCK_CORRECTION.json](evidence/UPSTREAM_LOCK_CORRECTION.json)，不宣称历史执行器字节相同。

模型环境为现有 Python 3.10、Torch 2.8.0+cu128、Transformers 4.57.1。几何环境复用本机 Blender 附带的 Python 3.11.7 创建独立 venv，匹配原几何依赖并补充 NetworkX 3.4.2，关闭用户 site-packages。渲染复用本机 Blender 4.2.23 和原五视图代码，CPU Cycles、24 samples、默认 8 CPU threads。这不是完整 Scene-Designer 服务复现。

## 命令与恢复

完整实际命令、UTC 启止时间、首日志和退出码见 `execution/*/receipt.json`、`model_runs/*/launch.json`。环境变量为 `PYTHONNOUSERSITE=1`、`PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`；`P920_RUN_ROOT=/data/wyh/scene_cvpr_20261020/design_decisions_920`。已有结果不得覆盖；断线先检查回执，再续后续阶段。

权重恢复使用 `evidence/MODEL_DOWNLOAD_PLAN.json` 和 `scripts/920/download_fixed_model.py`，把脚本与计划放回机器实验根再执行；现有完整文件校验复用，partial 续传。官方小源码使用 `scripts/920/fetch_upstream.py` 与本目录的 920 锁。资产用 `paris/scripts/download.py assets --scope selected`，仅由本机获许可账户提供凭据。没有权重或资产通过 SSH 传输，token 不写入 Git。

几何依赖见 `scripts/920/requirements-geometry.txt`。运行时由 `paris/scripts/materialize.py` 创建；从两份原草稿重构的准确参数见 `execution/reconstruct_A/receipt.json` 和 B 对应回执。

本次对照入口依次为 `scripts/920/prepare_comparison.py`、`comparison_model.py --kind probe --label B_U`、`run_formal_models.py`、`process_revisions.py`、`score_comparison.py`，均带 `--root "$P920_RUN_ROOT"`。模型/编码入口用 `/data/wyh/conda/envs/c15_genrec/bin/python`；评分用 `$P920_RUN_ROOT/envs/geometry311/bin/python`；调度用系统 Python。准备时先宽松编码，再在任何模型调用前确认全部输入均在原 8192 上限内并冻结配置；没有截短或重新编码。完整原生提示从固定上游物化，Git 保存哈希、反馈和恢复代码，未重复分发提示全文。

恢复本次已生成场景时，直接对相同草稿重放已保存的 `parsed_calls.json`，无需重跑模型。最终验收见 [FINAL_ACCEPTANCE.json](FINAL_ACCEPTANCE.json)：新检出提交的原 167 个交付文件及 150 个运行证据文件通过校验，所有视图、场景、checkpoint 与模型回执相符。

本轮只有一间开发房、两个相关需求、一个种子和一次修订。下一步应先处理可执行工具约束与保留功能的动作选择，再另行设计实验；没有授权扩训，也没有跨项目泛化或真人偏好证据。
