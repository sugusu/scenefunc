# 工作区与入口

- 实验根：/data/wyh/scene_cvpr_20261020/design_decisions_920；Git 副本为其 scenefunc/ 子目录，分支 runs/920。
- 基线 AutoDL：bdabc219772112ff2aee0b4a14ffd86809f48cb1。paris/ 原始交付不得修改，历史 v1 结果位于 runs/920/P920-DD-v1_20261104T083214Z/。
- 实际运行数据：../runtime/P920-DD-v1_20261104T083214Z/，含 comparison_inputs、model_runs、drafts、revisions；大型文件只留服务器。
- 固定模型：XuanyuLiu/Scene-Builder-Qwen3-VL-8B-Instruct@63e235a0c0d6e5bc2f8f3485214df4457fd2e672；路径由 ../MODEL_DOWNLOAD_PLAN.json 给出。23 个选中 HSSD 网格已获授权并验哈希。
- 模型 Python：/data/wyh/conda/envs/c15_genrec/bin/python。几何 Python：../envs/geometry311/bin/python。只读审计使用系统 python3。
- 授权 GPU0 UUID GPU-6df2660c-5a77-4e41-0b1a-c6208368fcd1；GPU1 UUID GPU-3be0bba6-ad57-928a-12ac-8f0a635b3187，实际均 RTX3090 24GiB。启动前重新核对卡、进程、锁和显存，不能凭本文件认定当前空闲。共享 GPU0 锁 /data/wyh/scene_cvpr_20261020/work/gpu0.lock。
- 编排脚本在 scripts/920/；run_formal_models.py、comparison_model.py、apply_revision.py、process_revisions.py、score_comparison.py 已支持 --run-id，默认仍是 v1，新实验必须显式传入正确 ID。prepare_comparison.py 等旧入口仍针对 v1，不得直接用于新实验。
- 整体目标 research/920/RESEARCH_GOAL.md；界面正文 GOAL_TEXT.txt 与实际同步 GOAL_SYNC.json 位于同目录。工具诊断运行目录 ../runtime/P920-TC-v1_20261104T104227Z/，生成结束、场景未执行，不重抽。
- 审计：python3 scripts/920/audit_tool_contract.py --root <实验根> --out <新的JSON路径>，输出路径必须不存在。无 GPU、模型或工具动作。
- 现有证据验收入口 scripts/920/verify_run.py；其覆盖 v1。不要调用旧 sync_status.py / publish_evidence.py 使已完成状态回退。
- Git SSH 专用配置只存在本副本 local config，不读取私钥内容；日志保存在 ../preflight/，不可提交凭据。
