# Stage2 A/B Diagnostic Report

## 实验结论

| 方法 | seed 0 | seed 1 | seed 2 | 平均剩余冲突 ± 样本标准差 | 归零成功率 | Stage2 平均耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| A: current paper_strict Stage2 | 11 | 20 | 8 | 13.00 ± 6.24 | 0/3 | 35.53 s |
| B: legacy_targeted_stage2 | 0 | 0 | 0 | 0.00 ± 0.00 | 3/3 | 1.59 s |

每个 optimizer seed 仅运行一次 Stage1，分别得到 51、54、52 个冲突。A/B 使用完全相同的共享输入，环境 seed=2025，交通 seed=316，初始确定性/不确定性冲突=53/107，Top10、30 秒阈值及正式模型保持不变。

**判断：当前场景及三份 Stage1 输出并非无法消解；B 构造出了零冲突解，当前 Stage2 求解机制是剩余冲突的重要影响因素。** 这不证明 A 的编码内一定包含同样的解，也不能将优势单独归因于分段、针对性候选或全局回滚。两种方法的搜索空间、接受规则和计算预算形式不同，本实验比较的是完整求解机制，而非等评估次数实验。

A 三次最终冲突合计 39 个：原位置残留 22 个、同一航班对位置迁移 1 个、新航班对上的冲突点 16 个。对应新航班对数为 6、5、5；B 三次最终新航班对均为 0。因此 B 在最终结果上明显减少新冲突残留，但只有三个样本，不作统计显著性声明，也不声称从未产生瞬时新冲突。

代价：相对初始计划，A 风险增幅为 0.453%、1.274%、0.064%，B 为 4.598%、17.137%、18.802%。B 的风险平均增幅约 13.51%，高于 A 的约 0.60%。B 的旧 soft cost 和接受规则保留原样，不等同于论文目标最优化；final_fitness 则统一用当前 paper objective 在最终代数计算。不能仅凭归零将 B 判定为所有目标上的全面优越。

下一步仅建议 B0 完整机制、B1 去分段、B2 去针对性候选、B3 去全局回滚的机制拆解实验；本轮未运行拆解，也未修改正式 paper_strict。

旧逻辑来源：提交 `11e1a04` 的 metrics_summary.csv 记录初始 106、最终 0；与当前 optimization_model.py 的后续差异仅为提交 `6b39685` 增加 paper API 导入。实验仅适配旧局部 A* 调用的参数，使其使用当前 .8/.2、米制及无垂直偏好，不带回旧 A* 权重放大或距离单位。



Test: 20260913_113232_490610_stage2_ab



                          final_conflicts_mean  final_conflicts_std  zero_conflict_success_rate  runtime_mean  changed_flights_mean  delayed_flights_mean
method                                                                                                                                                   
A_current_stage2                          13.0             6.244998                         0.0     35.533349             45.666667             22.333333
B_legacy_targeted_stage2                   0.0             0.000000                         1.0      1.591379             44.000000              5.666667



Paired seeds 0,1,2; one Stage1 per seed; identical serialized Stage2 inputs. Initial conflicts: 53 deterministic / 107 uncertain.

Case B is legacy_targeted_stage2, NOT an exact paper/reference reproduction. Formal solvers unchanged.

Legacy local reroute calls are adapted only at the experiment boundary to the CURRENT A* weights .8/.2, meter units and no vertical preference; old weight inflation and grid units are not imported.

Runtime measures Stage2 solver only. Changed/delayed flights and risk compare against initial plans. Reduction ratio compares against shared Stage1.

Exact survivors match unordered flight pair, cell and aligned waypoint indices, ignoring arrival times. Moved same pair counts remaining nonexact events on original pairs. New pairs counts distinct final pairs absent from Stage1; new points counts their events. Resolved original events = original count minus exact survivors, including moved events.

New-pair creation counts final survivors, not all transient candidate conflicts. Rollback permits point increases when pair count decreases; it does not forbid replacement with new pairs.



 optimizer_seed                   method  final_conflicts  exact_survivors  moved_same_pair  new_conflict_points  new_conflict_pairs
              0         A_current_stage2               11                5                0                    6                   6
              0 B_legacy_targeted_stage2                0                0                0                    0                   0
              1         A_current_stage2               20               14                1                    5                   5
              1 B_legacy_targeted_stage2                0                0                0                    0                   0
              2         A_current_stage2                8                3                0                    5                   5
              2 B_legacy_targeted_stage2                0                0                0                    0                   0



A/B alone tests the combined grouping + targeted candidates + global rollback mechanism. Individual contributions require ablations. Three paired seeds support descriptive comparisons, not claims of statistical significance.

If B reaches zero, propose B0 full / B1 no grouping / B2 no targeted candidates / B3 no rollback next; do not run them automatically or alter paper_strict.

If B does not reach zero, scene structure, shared Stage1 output, 30-second threshold and local reroute feasibility remain hypotheses. Failure of these finite searches does NOT prove infeasibility.
