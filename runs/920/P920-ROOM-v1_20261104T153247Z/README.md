# P920-ROOM-v1：完整重建及首次导出失败

保留初稿全部 21 件家具、6 墙和 5 门窗，构建并恢复实际原生 HouseScene。17 件家具使用 16 个唯一网格，4 件仍是明确代理；未读取终稿。新增 18 个固定资产／坐标文件共 17149470 字节。候选形状和朝向实际预览 21 图，其中一个阶段在第四图后失败，余下六图单独完成。

首次真实观察因 Drake 未注册 `package://scene` 而失败，0 图／工具／模型。全部 7 个阶段（含失败）86.03 秒。后续 v2 只迁移资源 URI，未重建或重跑模型。结果见 `RESULT.json`、`reconstruction/RESULT.json`、`observation/RESULT.json` 和 `execution/`。

实际运行源码在 `code/`，包括三版预览脚本；不要把当前脚本误当初次失败时的版本。每阶段原命令及精确时间在 `execution/<stage>/receipt.json`。下载用 `ASSET_PLAN.json` 的固定仓库、revision、路径、大小和 SHA256，使用已有标准 HF 授权，不把凭据写入命令或 Git。

完整生成初稿、网格、候选预览及原生包留在服务器，路径、字节和 SHA256 见 `SERVER_ARTIFACTS.json`。原输入路径和哈希在 `RECONSTRUCTION_PLAN.json`；官方公开参考的恢复来源见 `research/920/source_recovery_v1` 及 SOURCE run。原 23 件资产及其锁未覆盖。

恢复历史构建时使用 run 内冻结代码，`PYTHONPATH` 指向仓库 `scripts/920`，固定 Python 为 `/data/wyh/scene_cvpr_20261020/design_decisions_920/envs/scene_designer311_native/bin/python`；将 receipt 中 `--out` 改为不存在的新目录。原 v1 包的 URI 缺陷应保留为失败证据，正常后续工作使用 v2 的显式迁移包。

这些是从生成参考构造的开发输入，不是原住户家具、测绘房间或专业历史动作。质量未验收；失败不归因于模型推理。HSSD 来源 `hssd/hssd-models`，revision `208ea6bc22d863bbc1cbaef1829b3e646600dfb6`；坐标来自 `hssd/hssd-hab` revision `4369cb9876214c7fbebcf552eb532380e4d287e4`，按已有锁的 CC BY-NC 4.0／提供者访问条款使用。
