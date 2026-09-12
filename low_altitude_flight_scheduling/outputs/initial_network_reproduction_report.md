# Initial Network Reproduction Report

Run: 20260912_181030_632_paper_strict_random_initial_network_seed2025_f77083bb; seed: 2025; scene: paper_strict_random. No FATA, ADM, Stage1 or Stage2 was executed.

1. The heterogeneous scene used OD hotspots and mixed OD patterns, peaked takeoff times, variable speeds, forced cruise altitude and optional corridors. These introduce structure absent from the published random initial-scene description. Archived heterogeneous baseline, recounted at 30s: deterministic=99, uncertain=130, CI point coverage=0.2462.

2. paper_random restores random ground OD sampling, 100 plans, distance near 6km, uniform ETD in [0,1800] seconds, constant 10m/s, A* risk/distance weights 0.8/0.2 and t_conflict=30s, alpha=0.05. No forced corridors, waypoints, altitude preference, ground penalty or route bias is used. The A* heuristic and path distance use meters; the engineering distance division by 100 is disabled only at this initial generation call. The 1200m tolerance is an implementation assumption. Random OD positions are discrete ground cell centers, conditioned on feasibility and the fixed distance band; this discretization is an implementation assumption. Random city/population maps and sigma0=1.0, sigma_rate=0.01 remain implementation assumptions, unchanged in this round.

3. CI uses a binary simple graph with one node per plan and one edge per potentially conflicting pair. CI=(k_u-1)*sum(k_v-1) on the exact l-hop boundary. Edge count metadata does not enter CI. Top10 uses ordinary CI with flight-id tie breaking. This follows the published formula; weighted/adaptive key selection is not used for this scene.

4. l=2 is an implementation assumption, not a disclosed paper parameter. Radius sensitivity is diagnostic only; default selection remains l=2.

5. Raw events=123; unique pair-cell points=123; duplicate extra index events=0. The main Conflict definition is unchanged. A repeated pair-cell may correspond to multiple temporal events, so the paper counting convention needs manual review when these counts differ. Continuous conflict ratio=0.4228; continuity uses the existing adjacency definition and is also a diagnostic convention. Top1/5/10 contribution shares count the union of events incident to the selected flights, avoiding double counting.

6. seed2025 (or the explicitly requested seed 2025) CI l=2 Top10 point coverage=0.365854, pair coverage=0.380952. Paper point coverage=78/97=0.804124. Deterministic=74 vs 53; uncertain=123 vs 97.

7. Scene and path-generation assumptions change the underlying network. Counting and radius effects are quantified below; there is no evidence here that changing the ordinary CI implementation is warranted. Aggregate outputs alone cannot isolate the effects of undisclosed city maps, population maps, seed and sigma. Comparing only the total conflict count is insufficient to claim structural reproduction.

```json
[
  {
    "ci_l": 1,
    "top10_flight_ids": "[39, 45, 54, 94, 91, 92, 37, 18, 55, 36]",
    "top10_point_coverage": 0.35772357723577236,
    "top10_pair_coverage": 0.36904761904761907,
    "remaining_edges_after_top10_removal": 53,
    "lcc_ratio_after_top10_removal": 0.17
  },
  {
    "ci_l": 2,
    "top10_flight_ids": "[39, 94, 45, 92, 37, 91, 18, 54, 55, 70]",
    "top10_point_coverage": 0.36585365853658536,
    "top10_pair_coverage": 0.38095238095238093,
    "remaining_edges_after_top10_removal": 52,
    "lcc_ratio_after_top10_removal": 0.17
  },
  {
    "ci_l": 3,
    "top10_flight_ids": "[39, 94, 45, 91, 31, 68, 70, 53, 92, 9]",
    "top10_point_coverage": 0.3170731707317073,
    "top10_pair_coverage": 0.4166666666666667,
    "remaining_edges_after_top10_removal": 49,
    "lcc_ratio_after_top10_removal": 0.17
  }
]
```

8. Calibration is optional when a scene resembling published initial statistics is needed, and must be labelled calibrated reproduction scene. A small scan can reveal variability, but does not establish that the paper scene can be recovered. The unbiased seed=2025 scene remains the strict result. Calibration must never use final optimization results and must not change CI, key count, tolerance or sigma to fit the targets.

## 本轮验证结论与小规模扫描

验证：python -B -m pytest -q，76 passed。场景生成与扫描没有执行 FATA、ADM 或任何调度阶段。

| 指标 | 原工程场景（统一30秒重新统计） | paper_random seed2025 | 论文 |
| --- | --- | --- | --- |
| 确定性冲突事件 | 99 | 74 | 53 |
| 不确定性冲突事件 | 130 | 123 | 97 |
| CI Top10事件覆盖 | 32/130 = 24.62% | 45/123 = 36.59% | 78/97 = 80.41% |
| 唯一航班对-格点 | 130 | 123 | 未公开口径 |
| 冲突网络边数 | 85 | 84 | 未公开 |

paper_random的123个事件没有重复pair-cell，故本例的覆盖差异不能归因于重复索引计数。CI按二值邻接结构和精确l跳边界计算，未使用边的count属性，也未扩大关键计划数量。l=1/2/3的覆盖率都远低于论文，现有证据主要指向初始随机场景及未公开环境/风险与时间不确定性假设，而非修改CI算法的需要；不能仅凭这些统计进一步断定其中某个未公开因素是唯一原因。

按要求仅扫描2020至2039（含两端），8进程，共20个完整场景。扫描目录：

`outputs/scene_scans/20260912_181037_595_paper_strict_random_initial_network_seed2020_f77083bb/`

扫描墙钟时间为5.36秒；CI l=2 Top10点覆盖率范围26.36%至64.43%。最高覆盖率出现在seed2031，为96/149=64.43%，对应确定性冲突117、不确定性冲突149，仍未同时达到论文初始特征。

仅对已扫描结果计算规定的初始统计score，最小值仍来自seed2025，score=1.102538；这是诊断结果，没有启用校准选景或改变严格运行seed。当前20个候选内，启用paper_calibrated也不会得到比seed2025更低的score，因此本轮没有理由替换严格场景。将来若明确需要校准复现场景，可显式启用校准模式并保存选择报告；不应把最高coverage单独当成选景标准。

sigma0=1.0、sigma_rate=0.01保持未改。后续可独立研究sigma敏感性及风险地图/随机城市假设，严格结果仍应照实报告；本轮没有通过修改优化算法掩盖网络差异。
