# A vs A+ Global Safeguard Report

## 配对实验结论

**当前 strict lexicographic global safeguard 暂不适合直接定型为正式 A+ 改进算法。** 它在这十个配对 seed 上仅带来很小的平均冲突改善，未提高归零率，且最终新增冲突对更多。保留为独立实验候选，不覆盖 paper_strict，不设为默认模式。

| 指标 | A: paper_strict 基线 | A+: 项目提出的全局非恶化保护 |
| --- | --- | --- |
| 平均剩余冲突 ± 样本标准差 | 11.2 ± 3.46 | 10.9 ± 4.70 |
| 剩余冲突中位数 | 10 | 11 |
| 零冲突成功率 | 0/10 | 0/10 |
| 平均最终新增冲突对 | 3.1 | 4.3 |
| 平均原位置 exact survivors | 7.6 | 6.5 |
| 平均风险增幅 | 0.412% | 0.755% |
| 平均延误航班 | 21.5 | 21.9 |
| 平均改动航班 | 46.5 | 47.3 |
| 平均原论文 fitness | 478060.88 | 470364.57 |
| 平均 Stage2 耗时 | 39.28 s | 40.52 s |

1. A 平均剩余冲突 11.2 个。
2. A+ 平均剩余冲突 10.9 个，仅减少 0.3 个，约 2.68%；中位数反而从 10 增至 11，不能据此宣称稳定或统计显著改善。
3. 两者归零率均为 0%，未达到“归零能力明显提高”的成功标准。
4. A+ 没有减少最终 new conflict pairs，均值从 3.1 增至 4.3，增加约 38.7%。
5. A+ 减少 exact survivors，均值从 7.6 降至 6.5，但新增冲突对增加，因此不能只凭旧位置冲突减少判定总体迁移受抑制。
6. 共 99,500 个 parent-to-trial 提案，接受 4,143 个，回滚 95,357 个，接受率约 4.16%。拒绝原因：不可行绕飞 57,770；冲突对增加 36,951；同对数下冲突点增加 565；同对/点数下 fitness 未严格改善 71（包括完全相同的 tie）。几何不可行是原有可行性约束，不能全归为新增保护机制的效果；在可行但被拒绝的 trial 中，约 98.31% 因冲突对增加而被拒绝。
7. A+ 平均风险增幅 0.755%，低于 2% 参考目标，但高于 A 的 0.412%；最高单次约 2.107%，没有超过 5%，不存在 legacy B 式平均 10% 以上的风险增幅。
8. 平均延误仅增加 0.4 架，改动增加 0.8 架，描述性差距较小，不声称统计显著。
9. 当前主要问题不是“改善明显而风险暴涨”，而是改善有限、归零率不变、最终新增冲突对更多。
10. 不建议直接正式化这一严格版本。后续需要人工确认后另行设计改进验证，本轮未加入 relaxed safeguard、epsilon、风险预算、warm start 或 legacy 候选。

## 机制边界

“为抑制局部策略调整引起的全局冲突迁移，在原 ADM-FATA 联合搜索框架上引入全局冲突非恶化保护机制。”这是本项目的独立改进，不是原论文明确公开的组成部分。

保护比较的是每个随机初始化个体自己的 parent 与 trial，**不是**相对共享 Stage1 禁止新冲突；pair 数下降时允许 point、风险增加，pair 数相同时也允许旧航班对被新航班对替换。因此，即使接受键严格改善，最终 provenance 的 new pairs 也可能更多。每代 best 的新增对均值从初始化的 20.4 降至最终的 4.3，但 9/10 条轨迹存在中途增加，且最终仍高于 A，不能把代内下降当作相对 A 的成功。

大量拒绝和较低接受率提示严格规则可能限制联合搜索的过渡探索，但未经额外对照不能作因果断言。此前 legacy B 的 leave-one-out 结果证明了回滚在其完整机制中的条件性重要性，不证明这一机制移植到不同候选生成和搜索空间后单独足够有效。

最终输出始终选择实际保留候选中按未修改 paper fitness 记录的 incumbent；best_conflict_lexicographic 只保存作诊断。父代 geometry、components、conflicts、context 缓存复用，fitness 则按当前代 delta 重计，避免代际比较失真。

## 验证

复用 seed 0–9 既有 Stage1，未重跑 Stage1、未重生成初始航路、未运行 B。A 的 seed 0–2 指标与上一轮逐项一致。十份共享输入哈希、20 个最终解的全局冲突/fitness/来源分区及 position-context 重解码均通过复核，十组 accepted mask 与 ADM 学习资格/回滚原因分区通过检查。完整测试：120 项通过。旧实验结果和 paper_strict 默认配置未改动。

Project enhancement: A_plus_global_safeguard. paper_strict remains the unmodified reproduction baseline.

```text
                         n_runs  final_conflicts_mean  final_conflicts_std  final_conflicts_median  zero_conflict_success_rate  new_conflict_pairs_mean  exact_survivors_mean  changed_flights_mean  delayed_flights_mean  risk_increase_mean  risk_increase_std   fitness_mean  runtime_mean
method                                                                                                                                                                                                                                                                                       
A_current_stage2             10                  11.2             3.457681                    10.0                         0.0                      3.1                   7.6                  46.5                  21.5            0.411743           0.496430  478060.884719     39.284623
A_plus_global_safeguard      10                  10.9             4.701064                    11.0                         0.0                      4.3                   6.5                  47.3                  21.9            0.754938           0.806826  470364.570181     40.522248
```

A mean conflicts=11.200; A+=10.900. Zero rates: 0.0% vs 0.0%.

New final pair means: 3.100 vs 4.300; exact survivors: 7.600 vs 6.500.

Rollback totals:
```text
rollback_due_to_pair_increase          36951
rollback_due_to_point_increase           565
rollback_due_to_fitness_worse             71
rollback_due_to_infeasible_route       57770
new_pair_trials_rejected               37582
rejected_lower_risk_equal_conflicts       28
```

Risk means: A=0.412%, A+=0.755%. Delayed flights: 21.500 vs 21.900. Changes: 46.500 vs 47.300.

Ideal criterion (better zero rate and mean risk <=2%): False.

Acceptance is strictly candidate-level (pairs, points, current-generation paper fitness). Pair decrease can accept point or risk increase; the rule does not prohibit all novel pairs. Equal metrics retain parent. Infeasible routes never pass a finite parent.

Parent position/context/components/conflicts are inherited on rollback. Parent and elites fitness are recomputed from cached components under current delta without decoding/detection. Trial evaluation uses existing globally equivalent incremental detector and caches. Final output is the original fitness incumbent among retained candidates, not the diagnostic lexicographic elite.

FATA good-point initialization, MLF/LPS formula, sequential proposal draws, ADM formula and learning rate unchanged. Accepted-mask filtering is the only ADM learning eligibility addition. Generation 1 initializes parents and does not count toward trial acceptance rates.

All 10 Stage1 inputs reused; seed 0..2 are stored in the original A/B directory and referenced by ablation provenance. No scene regeneration, Stage1 run, B rerun, legacy candidates or grouping. Environment arrays regenerated only to verify identical hashes.

Ten paired seeds are descriptive evidence, not proof of global optimality or statistical significance. No relaxed safeguard, altered objective/weights or default-mode changes. Previous legacy B average risk was 10.826% in the ten-seed ablation, cited only as diagnostic reference.
