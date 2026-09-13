# Stage2 Mechanism Ablation Report

## 核心结论

在这 10 个配对 seed、固定 60 轮预算的 leave-one-mechanism-out 实验中，**global rollback / non-worsening acceptance 的冲突消解贡献最大**。

| 变体 | 归零率 | 平均最终冲突 ± 样本标准差 | 平均新增冲突对 | 平均原位置残留 |
| --- | --- | --- | --- | --- |
| B0 完整机制 | 10/10 | 0 ± 0 | 0 | 0 |
| B1 去连续分段 | 10/10 | 0 ± 0 | 0 | 0 |
| B2 去时间差/到达顺序定向候选 | 10/10 | 0 ± 0 | 0 | 0 |
| B3 去全局回滚，保留目标 pair 局部改善 | 0/10 | 21.2 ± 8.05 | 2.5 | 18.4 |

1. 零冲突成功率下降最大：全局回滚，下降 **100 个百分点**。
2. 平均最终冲突增加最大：全局回滚，增加 **21.2 个**。Delta_grouping=0，Delta_targeted=0，Delta_rollback=21.2。
3. 最能抑制最终新增冲突对残留：全局回滚，去除后平均增加 **2.5 对**。这里比较最终来源指标，不声称完整机制从不产生瞬时新冲突。
4. 最能抑制原位置 exact survivors 残留：全局回滚，去除后平均增加 **18.4 个**。局部改善可能使此前已消除的冲突重新出现，因此不能把最终 exact survivor 全部解释为“从未改善”。
5. 相互作用：本设计无法单独验证或量化 interaction。B1/B2 未退化不意味着分段、定向候选在所有场景中无贡献；归零指标存在地板效应，且 leave-one-out 效果取决于另外两种机制仍然存在。也不能声称全局回滚单独足以归零。
6. 下一步 A+ 第一候选：研究 **soft feasibility safeguard / elite preservation**，防止已改善的全局冲突结构在搜索中反复退化。
7. 融合边界：全局非恶化约束是论文未明确公开的额外搜索约束，不能直接写入 paper_strict。后续应独立设计 A+ 对照实验，不复制整个 legacy solver，也不在本轮修改 A。

前三个 seed 的 B0 与上一轮 B 的最终计划及逐动作序列一致。原有 seed 0–2 没有重跑 Stage1；因 B1/B2 前三次均归零而自动扩展 seed 3–9，恰好新增 7 次 Stage1，每次供四个变体共享。全部 40 组的最终全局冲突与来源、10 份共享哈希、动作接受规则和检测次数均复核通过；相关测试 50 项通过。

Test: 20260913_115912_599871_ablation; paired seeds: 10; extended Stage1 runs: 7.

B0 reproduced the previous 0/0/0 and physically identical final plans before ablations.

## Results

```text
                                    n_runs  final_conflicts_mean  final_conflicts_std  final_conflicts_median  zero_conflict_success_rate  new_conflict_pairs_mean  exact_survivors_mean  runtime_mean  risk_increase_mean  changed_flights_mean  delayed_flights_mean
variant                                                                                                                                                                                                                                                               
B0_full                                 10                   0.0             0.000000                     0.0                         1.0                      0.0                   0.0      1.628833           10.826096                  44.1                   5.2
B1_no_segment_grouping                  10                   0.0             0.000000                     0.0                         1.0                      0.0                   0.0      1.519540           13.535520                  45.2                   5.1
B2_no_conflict_informed_candidates      10                   0.0             0.000000                     0.0                         1.0                      0.0                   0.0      1.842795           11.265893                  45.2                   5.2
B3_no_global_rollback                   10                  21.2             8.052605                    21.0                         0.0                      2.5                  18.4      1.421494            9.133740                  32.3                   5.2
```

## Contributions (lexicographic)

```text
                   mechanism                    removed_variant  zero_conflict_rate_drop  mean_final_conflict_increase  mean_new_pair_increase  mean_exact_survivor_increase  rank
             global_rollback              B3_no_global_rollback                      1.0                          21.2                     2.5                          18.4     1
            segment_grouping             B1_no_segment_grouping                      0.0                           0.0                     0.0                           0.0     2
conflict_informed_candidates B2_no_conflict_informed_candidates                      0.0                           0.0                     0.0                           0.0     2
```

Rate drop is a fraction; multiply by 100 for percentage points. No weighted score. Tied lexicographic values share ranks.

## Answers

Largest zero-rate drop: ['global_rollback'].

Largest mean conflict increase: ['global_rollback'].

Largest final new-pair increase (suppression evidence): ['global_rollback'].

Largest exact survivor increase: ['global_rollback'].

Leave-one-out effects are conditional on the other two mechanisms. Interactions are possible, but cannot be established or quantified without factorial experiments.

First A+ candidate: global_rollback.

soft feasibility safeguard or elite preservation; extra undisclosed constraint, never insert into paper_strict

## Definitions and Controls

B1 uses one raw Conflict per unit in detector order, never calls grouping. Classification stays unchanged. B2 fixed delay order=(absolute value, signed value), alternating a/b per value; speed follows configured factor order alternating a/b. Original eligibility filters and limits stay fixed. Reroute geometry and actor order remain unchanged and do not inspect arrival time.

B3 accepts strictly reduced total target-pair points, including pair disappearance; global detection does not veto. First accepted action behavior stays fixed. Round processing still stops on no accepted improvement, maximum 60.

Action trace new_pairs_created_by_action includes rejected hypothetical candidates. Accepted/rejected counts exclude no_candidate and unresolved notices. Reroute accepted includes altitude shifts. Detection counts include final validation; A* counts actual calls including unsuccessful searches.

Origins use prior A/B definitions: exact pair/cell/aligned indices ignoring arrival times; moved same pair counts nonexact events on original pairs; new pairs counts final distinct pairs absent from shared Stage1. Resolved original includes moved events. Runtime solver only; risk, changes and delay relative to initial scene. All current A* parameters retained through the same adapter as B0.

No A or formal paper_strict code modified. Finite paired sample results are descriptive, not statistical significance or standalone mechanism sufficiency claims.
