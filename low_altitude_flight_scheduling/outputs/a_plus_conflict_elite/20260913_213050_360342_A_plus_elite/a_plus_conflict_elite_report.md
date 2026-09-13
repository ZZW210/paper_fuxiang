# Global Conflict Elite Preservation Report

## 研究结论

GCEP 在本次十个配对 seed 中带来有限改善，但没有提高零冲突发现率或最终归零率。**当前证据更偏向 search discovery 瓶颈，而非零冲突解找到后丢失或最终 fitness 未选择它。** 不修改目标函数，不指定两个 A+ 输出中哪个作为正式算法输出，不自动进入下一轮改进。

| 输出 | 平均最终冲突 ± 样本标准差 | 最终归零率 | ever-zero 率 | 平均风险增幅 | 平均新增冲突对 |
| --- | --- | --- | --- | --- | --- |
| A_current_stage2 | 11.2 ± 3.46 | 0/10 | 0/10 | 0.412% | 3.1 |
| A_plus_paper_incumbent | 10.6 ± 3.57 | 0/10 | 0/10 | 0.421% | 1.9 |
| A_plus_conflict_elite | 10.6 ± 3.57 | 0/10 | 0/10 | 0.421% | 1.9 |

1. A 平均最终冲突 11.2 个。
2. GCEP paper incumbent 平均 10.6 个。
3. GCEP conflict elite 平均 10.6 个；本轮十个 seed 两个输出的最终指标一致，仍分别保存其计划、向量、context、冲突和指标。
4. 三者最终归零率都是 0/10。
5. A 与 GCEP 的每代正常评估种群中均从未产生零冲突个体。first_zero_generation/fitness/risk/changed/delayed 均为空，表示未发生，不是第 0 代。首个零冲突候选的保存路径已实现，但本轮没有触发。
6. 因从未发现零冲突个体，zero_found_then_lost_without_elite 在十个 A seed 均为 False；没有证据把本轮失败归因于零冲突解未保留或 paper fitness 未选择。不能据此证明目标尺度完全无问题，也不能证明编码空间不可行。
7. GCEP 平均重新注入 183.9 次，总计 1,839 次。每次最多替换一个成员，正常 trial 不因 pair/point 增加被回滚。
8. archive 副本的历史字典序最优记忆得到保护，但不等于整体搜索统计稳定性提高。最终冲突标准差从 3.46 略增至 3.57，中位数从 10 增至 10.5；没有稳定归零改善证据。
9. 平均最终 new conflict pairs 从 3.1 降至 1.9，减少约 38.7%。但 exact survivors 从 7.6 增至 8.2，原位置残留并未减少。字典序 pair 优先也不保证相对 Stage1 完全禁止新 pair。
10. 平均风险增幅 0.421%，接近 A 的 0.412%，低于 2% 参考目标；最高单次约 2.798%，未出现超过 5% 的风险增幅。平均改动航班从 46.5 增至 47.4，延误航班从 21.5 增至 22.3，均一并披露。
11. 相比历史 strict rollback，GCEP 平均冲突 10.6 略低于 10.9，风险 0.421% 低于 0.755%，新增冲突对也更少；但差距不能作统计显著性声明，且两者均 0/10 归零。GCEP 机制上保留自由 trial 探索，却尚未达到实质归零成功标准，不能宣布全面有效或直接正式定型。
12. 在固定预算、场景和当前编码下，search discovery 是本轮最直接的证据支持限制。精英保留本身不足以解决发现能力；后续 warm start、targeted initialization、dimension reduction 仅作为人工确认后的研究方向，本轮未实现。

## 历史严格回滚参考

| 方法 | 平均最终冲突 | 零冲突率 | 平均风险增幅 |
| --- | --- | --- | --- |
| A_current（本轮复现） | 11.2 | 0/10 | 0.412% |
| A_plus_strict_rollback（历史读取） | 10.9 | 0/10 | 0.755% |
| A_plus_conflict_elite（本轮） | 10.6 | 0/10 | 0.421% |

严格回滚未重跑，历史数据和实验文件未修改。GCEP 与 A 的平均 Stage2 诊断运行耗时分别约 36.31 s 和 35.96 s；两个 A+ 输出共享一次搜索耗时，不是两次独立搜索。

## 注入时序与学术边界

在原 ADM-FATA 联合优化框架基础上，引入面向全局冲突结构的精英记忆与保留机制，避免搜索过程中已发现的低冲突解因种群更新而完全丢失，同时保留非精英个体的自由探索能力。GCEP 是项目提出的额外 search-memory enhancement，不来自原论文。

本代 MLF/LPS 产生的种群完成评估、原 paper incumbent 选择和原 ADM 概率更新后，再在这份有完整一致评估的种群中检查精英 position/context 是否存在，并按当前代 (pairs, points, fitness) 替换最差成员。下一轮正常 MLF/LPS 可以改变该成员，正常策略 sampling 也保持原样；只有独立 archive 副本持续保留。没有冻结精英，没有 accepted_mask，也没有 parent-to-trial rollback。此时序避免用未评估 proposal 的旧分数挑选最差成员，以及同代重复计入 ADM 学习。

historical elite 用缓存 components 按当前 delta 重新计价；重新计价不 decode 或 detect。新 archive 候选才解码并保存完整 plans。population_generation_stats 是注入前正常候选统计，conflict_elite_history 同时记录注入前是否存在、注入动作与替换键。pair 优先允许 pair 下降时 point 上升，因此不能要求历史 elite_points 单调下降。

## 验证

Stage1 重跑次数为 0，初始确定性/不确定性冲突保持 53/107。十份共享哈希、30 个最终解的完整全局冲突/原 paper fitness/来源分区/position-context 重解码、20 份逐代种群与精英轨迹及全部 CSV run_id 均核验通过。十个 A 的 position、context、最终冲突和 fitness 与上一轮完全一致；默认 paper_strict 源码、配置及旧严格回滚结果无改动。完整测试 133 项通过。

Run ID: 20260913_213050_360342_A_plus_elite

Global Conflict Elite Preservation (GCEP) is an additional search-memory enhancement introduced on top of the reproduced ADM-FATA framework.

```text
                        n_runs  final_conflicts_mean  final_conflicts_std  final_conflicts_median  zero_conflict_success_rate  ever_zero_conflict_rate first_zero_generation_mean  new_conflict_pairs_mean  exact_survivors_mean  risk_increase_mean  risk_increase_std  changed_flights_mean  delayed_flights_mean   fitness_mean  runtime_mean  elite_reinjection_mean
method
A_current_stage2            10                  11.2             3.457681                    10.0                         0.0                      0.0                        NaN                      3.1                   7.6            0.411743           0.496430                  46.5                  21.5  478060.884719     35.959889                     0.0
A_plus_conflict_elite       10                  10.6             3.565265                    10.5                         0.0                      0.0                        NaN                      1.9                   8.2            0.420956           0.884152                  47.4                  22.3  453225.735088     36.313871                   183.9
A_plus_paper_incumbent      10                  10.6             3.565265                    10.5                         0.0                      0.0                        NaN                      1.9                   8.2            0.420956           0.884152                  47.4                  22.3  453225.735088     36.313871                   183.9
```

Historical strict rollback reference: mean conflicts=10.9, zero rate=0/10, risk=0.755%; read only, not rerun.

Candidate acceptance and ADM learning are unchanged. One archived elite is injected AFTER normal ADM learning into the coherent evaluated population, replacing lexicographic worst only if position AND context are absent. The next MLF/LPS proposal may mutate every row, including this elite; the next normal context sampling also remains unchanged. Archive copy is independent and repriced with current delta without historical decode/detection.

Population statistics and ever-zero checks use the normal PRE-injection evaluated candidates. Elites are not repeated in same-generation ADM learning. Means exclude geometrically infeasible candidates. Lexicographic preference is NOT an extra paper-fitness penalty and need not monotonically decrease points when pairs decrease.

Both paper-fitness incumbent and conflict archive are saved; no official output choice or objective change made.

Baseline diagnostics reproduced the prior A positions, contexts, final counts and fitness for all ten seeds. Stage1 runs=0. No strict rollback, targeted candidates, grouping, legacy solver, relaxed safeguard or parameter changes.

If no normal candidate reached zero, search discovery remains the evidence-supported limitation within this finite budget. If zero occurred but output is nonzero, final selection/preservation requires scrutiny; this does not prove an objective-scaling defect without further analysis.

Archive monotonicity is a structural memory guarantee, not proof of overall convergence stability or statistical significance. Runtime is shared by both A+ output rows; never sum them as independent runs. No next-round enhancement implemented.
