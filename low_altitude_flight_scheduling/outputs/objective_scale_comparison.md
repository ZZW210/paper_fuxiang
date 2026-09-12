# Objective scale comparison

Default remains raw_equation. No automatic selection.
Fitness values have different scales and must not be compared directly.
Identical initial inputs verified by SHA256; seed and all other configuration match.

Runtime is observed wall time, not a controlled performance benchmark.

```text
                          mode  stage1_Nc  final_Nc       Tdelay         Tair       ORISK  n_delay  changed_flights       fitness    runtime
                  raw_equation         98        18 33692.198031 64406.436987 1156.237925       22               70 844927.978901 361.918283
initial_reference_experimental         98        43 29427.392937 60277.639287 1136.450902        0               69      0.404161 292.671497
```

## Initial input fingerprints

- initial_plans.pkl: 16b5a1c08cd1ef9b7166f0d79c05aff97467ce9d49e1da9d78a246c29330f24e
- risk_map.npy: a87cd8aeabbaf1f802f20173231c7fccc594c442774321703806781374dbe462
- conflicts_uncertain.csv: 88de8ebd8bb3ea57467fc1046af74a02626790e986872bb802fa564cec190e6b
- key_flights.csv: 1effa8c49189adf4379bef7d409d973729d4b462feaec4d05b7bf5ebc81cd9ff

## raw_equation

固定CI关键计划涉及初始冲突点32/130；其余98点两端都不参与Stage1，故Stage1整体Nc不能低于98。本次Stage1 Nc=98；该覆盖诊断不改变CI或生成数据。

- 终代罚项=5200.000，占fitness 0.6%；changed架数及Stage2增量修改架数不直接入目标，不能保证少改动；不为拟合现象新增惩罚。
- Stage1冲突点130 -> 98，减少24.6%；整体点数不等于作者连续冲突私有数据。
- Stage1 schedule-only=1/10，组合flight=7；若仍只用schedule，如实列出。
- 实际改航flight：Stage1=1，Stage2相对Stage1=15。
- Stage2冲突点98 -> 18，继续下降=True；working base为Stage1，不代表必然找到更小改动。
- 最终changed=70，少于旧版61=False；Stage2增量修改60架，新增changed 60架。
- 提前12架，延后22架；平均/最大绝对偏移336.922/1800.000s。
- n_delay=0且Tdelay>20000观察标记=False；提前也计入绝对Tdelay，数学上可出现，此标记仅诊断，不触发限制或repair。

## initial_reference_experimental

固定CI关键计划涉及初始冲突点32/130；其余98点两端都不参与Stage1，故Stage1整体Nc不能低于98。本次Stage1 Nc=98；该覆盖诊断不改变CI或生成数据。

- 终代罚项=0.000，占fitness 0.0%；changed架数及Stage2增量修改架数不直接入目标，不能保证少改动；不为拟合现象新增惩罚。
- Stage1冲突点130 -> 98，减少24.6%；整体点数不等于作者连续冲突私有数据。
- Stage1 schedule-only=8/10，组合flight=0；若仍只用schedule，如实列出。
- 实际改航flight：Stage1=0，Stage2相对Stage1=2。
- Stage2冲突点98 -> 43，继续下降=True；working base为Stage1，不代表必然找到更小改动。
- 最终changed=69，少于旧版61=False；Stage2增量修改59架，新增changed 59架。
- 提前40架，延后0架；平均/最大绝对偏移294.274/1364.728s。
- n_delay=0且Tdelay>20000观察标记=True；提前也计入绝对Tdelay，数学上可出现，此标记仅诊断，不触发限制或repair。
