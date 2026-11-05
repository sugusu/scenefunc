# 初态恢复

- nominal: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-MOABIT-INPUT-v1_20261105T213853Z/nominal/build/bundle`, manifest SHA256 `f608b7961f41142b99c6e9e48250a1b7aabf8848a0a15375dc4c5f33dca89f95`。
- narrower: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-MOABIT-INPUT-v1_20261105T213853Z/narrower/build/bundle`, manifest SHA256 `f28b610c7bc3a15c6b122e578f185379656c8cb4e131aa39944fb31866a73c13`。
- wider: `/data/wyh/scene_cvpr_20261020/design_decisions_920/runtime/P920-MOABIT-INPUT-v1_20261105T213853Z/wider/build/bundle`, manifest SHA256 `2cc8d0445f1ae311a50eb0fd77eb275482444723c2b243572c8f4b5c26a26aa1`。

每个服务器bundle包含完整原生状态、房壳SDF和原目录资产依赖，已在构建后同进程恢复并核全部结构ID/位姿。Git只保存原生元数据、SDF、源资产清单哈希与名义五图。资产来源由catalog.json与manifest追溯到已有固定目录，没有新权重或大型网格上传。原生导入可复用native_project.import_project，不能把这些空壳恢复当成带家具功能验收。窄/宽变体未渲染，任何未来模型执行前须单独获取并核对各自观察。
