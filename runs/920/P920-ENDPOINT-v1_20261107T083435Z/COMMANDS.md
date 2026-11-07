# 实际执行命令与退出状态

## endpoint_receipt

```sh
../envs/scene_designer311_native/bin/python scripts/920/check_conditional_endpoints.py --plan runs/920/P920-ENDPOINT-v1_20261107T083435Z/PLAN.json --out ../runtime/P920-ENDPOINT-v1_20261107T083435Z/measurements
```

实际退出：1；7.943903秒。

## endpoint_v2_receipt

```sh
../envs/scene_designer311_native/bin/python scripts/920/check_conditional_endpoints.py --plan runs/920/P920-ENDPOINT-v1_20261107T083435Z/PLAN_V2.json --out ../runtime/P920-ENDPOINT-v1_20261107T083435Z/measurements_v2
```

实际退出：0；12.058086秒。

## endpoint_v3_receipt

```sh
../envs/scene_designer311_native/bin/python scripts/920/check_conditional_endpoints.py --plan runs/920/P920-ENDPOINT-v1_20261107T083435Z/PLAN_V3.json --out ../runtime/P920-ENDPOINT-v1_20261107T083435Z/measurements_v3
```

实际退出：0；12.007185秒。

## loop_receipt

```sh
../envs/scene_designer311_native/bin/python scripts/920/check_executed_live_loop.py --conditional-fixture --plan runs/920/P920-ENDPOINT-v1_20261107T083435Z/LOOP_PLAN.json --out ../runtime/P920-ENDPOINT-v1_20261107T083435Z/loop
```

实际退出：0；0.718750秒。

下一新终态的只读量测（须已冻结本文件所列代码和初态）：

```sh
../envs/scene_designer311_native/bin/python scripts/920/evaluate_conditional_room.py --plan <对应活动冻结计划> --bundle <新checkpoint> --out <不存在的输出文件>
```

不运行旧closed试验；权重/模型/网格留服务器，输入manifest和全部资源由validate_bundle验证。
