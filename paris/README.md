# 巴黎开间 A/B：AutoDL 实际执行基线与 920 接手

这是固定开发壳体的家具阶段实验。真实模型生成、元数据检索、原生摆放内核、状态恢复、五视图渲染和一次网格反馈已运行；**不是完整 Scene-Designer 服务复现，不是完整房间成功，也没有证明偏好反转或新算法优势**。

主记录保留 8 次模型尝试：6 次完成、2 次基础设施失败。普通检索修正重放了 20 条实际资产请求。A 保留 9 件家具，相交对数 7→7；B 从 10 件变为 9 件，相交 7→5，两对消失相交都涉及被移除的椅子。不能把删物体直接计为质量提升。有限桌面活动见证另列，人的可达性、使用姿态、收纳等仍未知。

## 目录和版本

- `src/`：实际运行的本项目适配器快照，原目录名保留以解析依赖。
- `config/`：原 A/B brief、共同空壳、需求卡、冻结预算和模型/资产/官方源码清单。
- `scripts/`：可移植下载、工作区生成、阶段调度、场景重构和 CPU 验收。
- `runs/autodl/20261104_paris/`：只读机器结果，包含请求、响应、回执、状态与逐对象指标。
- `provenance/source_manifest.json`：原文件相对路径、原 SHA256、打包后 SHA256 与转换声明。
- `licenses/`、`NOTICE.txt`：上游及数据来源。权重、网格、第三方原图和模型生成的室内渲染不在 Git 中。

打包把 AutoDL 路径转为 `@WORK@` 等标记。官方原系统提示前缀改为 `@NATIVE_SYSTEM_A@/B@` 引用；`materialize.py` 用固定下载的模板恢复。机器结果中的原始 payload/request 哈希是**路径和提示引用转换之前**的哈希；当前可分享版本由 Git 和 `release_manifest.json` 校验。不能直接把转换后的 JSON 当作未修改的原字节。

## 从空缓存下载（920 本机执行）

从用户指定仓库检出交付 SHA。下面以当前仓库根为工作目录，`DATA`、`RUN` 和 Python 路径由操作者设置，不要求 `/root` 权限。

```bash
git clone https://github.com/sugusu/scenefunc.git
cd scenefunc
git switch --detach <AutoDL交付的完整SHA>
git switch -c experiments/920-paris-<日期>
export DATA=/your/data/scenefunc
export RUN="$PWD/paris/runs/920/<唯一运行ID>"
```

先执行无网络 dry-run：

```bash
python paris/scripts/download.py weights --data "$DATA"
python paris/scripts/download.py assets --scope full --data "$DATA"
python paris/scripts/download.py upstream --data "$DATA"
```

权重固定为 `XuanyuLiu/Scene-Builder-Qwen3-VL-8B-Instruct@63e235a0c0d6e5bc2f8f3485214df4457fd2e672`。四个 safetensors 分片和七个小文件的 **11 项完整白名单、大小和 SHA256** 见 `config/weights.lock.json`，总计 **17,541,458,146 B**。下载直接写目标目录，不额外保存一套 HF cache；`.part` 支持续传，服务器忽略 Range 时从头重写对应 partial，完整哈希通过才改名。失败保留 partial，不更换 revision。

```bash
python paris/scripts/download.py weights --data "$DATA" --execute
python paris/scripts/download.py upstream --data "$DATA" --execute
```

HSSD 为 CC BY-NC 4.0，并要求上游账户接受许可。使用自己的获许可账户在 920 配置 `HF_TOKEN`（不要写入仓库或命令历史），然后：

```bash
python paris/scripts/download.py assets --scope full --data "$DATA" --execute --accept-hssd
```

`--accept-hssd` 只是使用者确认，不绕过服务端授权。403/401 时先处理账户许可；不得把不同来源网格改名替代。`hssd/hssd-models@208ea6bc22d863bbc1cbaef1829b3e646600dfb6` 的实际 436 件候选合计 **903,107,916 B**；坐标配置来自 `hssd/hssd-hab@4369cb9876214c7fbebcf552eb532380e4d287e4`。每件 ID、类别、实际哈希、已选尺寸、朝向依据、源路径都在 `config/assets.lock.json`。

只想重构既有输出可用 `--scope selected`：23 件、**62,502,236 B**，加相应坐标配置。这是选中资产回放，**不能声称复现 436 候选的检索排名**。完整 436 候选池也不是全量 HSSD。

下载脚本为未下载数据加 5 GiB 保留线计算最低空间；空缓存权重至少 22,910,167,266 B 可用。建议为权重、约 0.9 GB 候选、Python 环境、导出和 5 GiB 余量合计预留 **35–45 GiB**。并行旧环境可能需要更多。不会自动删除已有数据。

## 环境与离线校验

原运行分成三个 Python 环境：模型 Python 3.12.3，几何 Python 3.11.17，渲染 Python 3.11。原卡 V100 32 GB，约 25 GiB 系统内存、6 核；所有模型推理 batch 1、FP16、seed 0、不采样。first/placement 使用 eager，网格反馈最终完成版本使用 SDPA efficient + repeat_kv；失败 eager 显存不足和旧 SDPA 内核不兼容均保留。新机器要记录架构与后端差异，不承诺跨 GPU 位级同输出。“920”是机器标签，本包没有验证任何非 CUDA 加速卡实现。

```bash
python3.12 -m venv /your/env/model
/your/env/model/bin/pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
/your/env/model/bin/pip install -r paris/requirements/model.txt
python3.11 -m venv /your/env/geometry
/your/env/geometry/bin/pip install -r paris/requirements/geometry.txt
python3.11 -m venv /your/env/render
/your/env/render/bin/pip install -r paris/requirements/renderer.txt
python paris/scripts/download.py weights --data "$DATA" --verify-only
/your/env/model/bin/python paris/scripts/verify_processor.py --weights "$DATA/weights"
```

`verify_processor.py` 只加载 processor、校验索引及小文件；不加载大模型。完整分片哈希由前一命令负责。环境锁是实际安装版本；若某轮子不支持目标平台，应明确报告安装差异，不默默换实现。

## 创建可运行路径映射

```bash
python paris/scripts/materialize.py --run "$RUN" --data "$DATA" \
  --model-python /your/env/model/bin/python \
  --geometry-python /your/env/geometry/bin/python \
  --render-python /your/env/render/bin/python
```

`RUN` 必须不存在。生成的 `run_config.json` 保存父 commit、路径映射和运行源哈希。共享代码不改；所有新输出放在 `runs/920/<run_id>/work/`。路径与新环境的源码哈希变化有记录。原 A 的 SDPA 重试比较只在既有失败请求存在时启用；新运行不要求人为制造失败。

## 实际生成、导出、验收与续跑

```bash
python paris/scripts/pipeline.py --run "$RUN" --arm A preflight
python paris/scripts/pipeline.py --run "$RUN" --arm A first
python paris/scripts/pipeline.py --run "$RUN" --arm A correct-retrieval
python paris/scripts/pipeline.py --run "$RUN" --arm A render-empty
python paris/scripts/pipeline.py --run "$RUN" --arm A placement
python paris/scripts/pipeline.py --run "$RUN" --arm A audit-placement
python paris/scripts/pipeline.py --run "$RUN" --arm A render-placement
python paris/scripts/pipeline.py --run "$RUN" --arm A feedback
python paris/scripts/pipeline.py --run "$RUN" --arm A audit-feedback
python paris/scripts/pipeline.py --run "$RUN" --arm A render-feedback
```

按同样顺序将 `--arm A` 换成 `--arm B`。两臂都完成后：

```bash
python paris/scripts/pipeline.py --run "$RUN" measure
```

这是已执行的**有限家具阶段流程**，不会自动跑任意多轮直到通过。`first` 生成资产请求；普通修正检索统一类别和尺寸接口；`placement` 接收真实资产和五视图；`feedback` 额外接收同一状态的真实网格检查。模型读写本地，网络禁用，不含外部付费 API。检查器仍不覆盖完整人体功能。失败与缺项计入结果，不换样本。

每个阶段完成后有独立 `execution/<arm>_<stage>.json`，场景和 `checkpoint.json` 保留。中断后先检查该阶段回执；已有回执不允许覆盖。继续后续阶段即可；失败阶段保留在原 run，在新 run 重试并记录 parent run，不修改历史。需要从任一现有状态恢复时，直接调用物化后的 `dispatch_corrected.py --resume <checkpoint> --calls <新的动作JSON> --output <新目录>`；几何和状态哈希先验证后执行。

## 零模型调用重构已有场景

```bash
/your/env/geometry/bin/python paris/scripts/rebuild_scene.py --run "$RUN" \
  --checkpoint paris/runs/autodl/20261104_paris/paris_local_run_results_20261104/B_mesh_feedback_sdpa_kv/tools/checkpoint.json \
  --out "$RUN/reconstructed_B"
/your/env/geometry/bin/python "$RUN/work/paris_local_run_results_20261104/audit_scene.py" \
  --checkpoint "$RUN/reconstructed_B/checkpoint.json" --out "$RUN/reconstructed_B_audit.json"
/your/env/render/bin/python "$RUN/work/paris_local_run_results_20261104/render_state.py" \
  --scene "$RUN/reconstructed_B/partial_scene.glb" --out "$RUN/reconstructed_B_views" --transient-blend
```

转换路径会重新计算 payload 哈希；对象 ID、姿态和源资产保留，原文件不覆盖。`reconstruction.json` 比较原与重建的单资产导出哈希。GLB 包装可能受库版本影响；必须重算对象与几何检查，不用文件哈希替代语义验收。视图采用公开记录的天花板切开观察，不删除导出场景几何。

## CPU 交付验收与 920 结果提交

```bash
python paris/scripts/verify_release.py
python paris/scripts/recompute_metrics.py --out /tmp/paris_recomputed.json
python paris/scripts/download.py weights --data /tmp/empty-cache-plan
```

验收要从最终 commit 的**新检出目录**执行。它重算 A/B 件数与相交对统计，不冒充重新跑 FCL。实际 CPU 重构/几何重算的收据另列；未做新机 GPU 全流程就标未验证。填写 `runs/920/<run_id>/machine.json`，记录父完整 SHA、硬件型号/显存/内存、依赖版本、attention 配置、资产清单哈希、stage exit 与耗时。新机器更改代码需留 diff，结果和代码在同一分支提交；不要覆盖 AutoDL 历史，不要 force-push。

该包的最近一次 Critic 8 次诊断未并入主线成绩；它不能替代生成。当前交付目标是让另一台机器按同一版本复现与继续验证，而非宣布方法成功。
