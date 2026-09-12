# 基于复杂网络的城市低空飞行计划优化调度：算法级复现

本项目复现论文《基于复杂网络的城市低空飞行计划优化调度》的主要算法流程：三维栅格空域、城市低空第三方风险地图、改进 A* 四维飞行计划生成、考虑过点时间不确定性的冲突探测、冲突复杂网络关键飞行计划识别、多策略两阶段优化，以及原始/改进 FATA、PSO、GA 对比。

这是算法级复现，不是作者私有随机数据的逐点复刻。论文未公开完整仿真数据、随机种子、障碍物生成、人口密度细节和过点时间误差函数。严格初始场景使用论文条件下的随机 OD、均匀起飞时间和统一速度；交通脉冲仅属于工程模式。初始网络差异记录在 `outputs/initial_network_reproduction_report.md`。

## 安装与运行

```bash
pip install -r requirements.txt
python run_main.py --seed 2025
python run_main.py --quick --seed 2025
python run_sensitivity.py --quick --seed 2025
pytest
```

`--quick` uses a smaller FATA population and generation budget (`NP=20`,
50 generations per stage in strict mode). It is intended for smoke tests and fast diagnostics; it can
significantly reduce conflicts, but it does not guarantee that every conflict
will be eliminated. Use `python run_main.py --seed 2025` for the normal
two-stage run.

The optional `legacy_engineering` scheduler runs in safe mode: every accepted action is checked
against global conflict pairs, and actions that increase global conflict pairs
are rolled back. The second stage follows an ADM-style independent matching
engineering approximation inspired by reference [6]: each remaining continuous conflict unit
is classified and matched to scheduling, speed adjustment, or local rerouting
with an adaptive strategy probability matrix. Quick mode limits the number of
rounds and candidates, but it still keeps rerouting in the stage-2 strategy set.

Legacy calibration note: because the paper does not publish the original OD generator
or conflict-detection implementation, the synthetic reproduction reports
key-flight conflict coverage and uses a calibrated cell-level conflict window.
The stage-1 optimizer now uses continuous ATD and speed variables instead of
discrete delay/speed-factor indices, and key-flight ranking uses weighted CI so
continuous overlapping conflict segments are not undercounted.

## Paper-strict scheduling reproduction

`run_main.py`默认使用`optimization.scheduler_mode: paper_strict`。

```bash
python run_main.py --scheduler-mode paper_strict --quick --seed 2025
python run_main.py --scheduler-mode paper_strict --seed 2025
python run_main.py --scheduler-mode paper_strict --quick --seed 2025 --run-id test_001
# Optional, manually requested experiment:
python run_main.py --scheduler-mode paper_strict --paper-objective-scale-mode initial_reference_experimental --seed 2025
python compare_objective_scales.py --raw-run-id RAW_RUN_ID --norm-run-id NORM_RUN_ID
python check_paper_strict.py
python run_main.py --scheduler-mode legacy_engineering --seed 2025 --outputs outputs_legacy
```

目标论文3.2.1：对普通CI固定前10%的每个关键计划分配一个具体策略及其对应决策变量，同步求解整个关键集合。Stage1每架只有一个`strategy_gene∈[0,3]`，floor并clip至0/1/2分别对应schedule/speed/reroute；没有三个任意activation genes。Stage2重新检测后以每个conflict POINT为独立匹配单位建立`(m,3)`概率矩阵，同步独立采样原始策略标签，由FATA搜索`actor_gene∈[0,2]`选择本冲突主要调整的一端。最终多策略只来自不同冲突选择同一UAV却匹配不同策略，或来自两阶段累计；不投票，不全局强制单UAV单策略。

目标论文给出`ATD∈[1,3600]`、`|ETD-ATD|≤1800`、逐航段`v∈[5,20]`、电池时间1200秒，以及佳点集、动态delta和式(49)-(51)的适应度结构。严格模式使用连续ATD，允许提前起飞；完整逐航段速度用于Stage1，局部连续速度用于Stage2；FATA搜索局部via-cell的xyz，经A*连接为26邻域无障碍路径。冲突窗口使用目标论文表1的30秒，旧工程的20秒校准仅属于legacy。

normal默认`Parf=.2, NP=50`，两阶段各200代，权重`wc=.8, wd=.25, wr=.5, wt=.25, gamma=5`。quick仅改为`NP=20`、各50代，不改变策略、连续变量、ADM或目标函数。Stage1无剩余冲突时明确记录跳过Stage2。严格路径不使用离散候选、冲突对罚项、百万级冲突罚项、3%延误上限、changed ratio上限、工程repair、单步rollback、warm start、佳点集Gaussian噪声或Gaussian local search。默认8进程只并行fitness评价，位置更新、ADM、随机采样和best选择都按固定顺序在主进程进行；`--n-jobs 1`可用于串行对照。

参考[6]给出策略概率矩阵、逐冲突独立随机分配策略、优势种群、概率更新`P_new=(1-lrate)*P_old+lrate*dominant_frequency`及`lrate=.5`。其单UAV单策略限制不移植到目标论文：目标论文3.1.1和3.2.1允许组合。本实现用原始优势species逐冲突标签计算频率，采用目标论文指定的改进FATA而非参考[6]的ISFS。

### Implementation assumptions

旧约[0,1]目标直接加100/1000罚项时容易被罚项支配，且与论文公布的fitness量级不符。因此默认直算打印式(51)，initial-reference仅作实现假设对照；这不表示作者未公开的无量纲化已经复原。

1. 默认`optimization.paper_objective_scale_mode: raw_equation`直接计算打印式(51)：`(1-wc)*(wd*Tdelay+wt*Tair+wr*ORISK)+wc*delta*Nc*Tair+1000*n_battery+100*n_delay`。论文说明无量纲化但变换未公开，因此不能声称此尺度复原了作者实现。旧固定初始归一化仅作为`initial_reference_experimental`保留：`Tdelay/(N*1800)`、`Tair/initial_total_air_time`、`ORISK/initial_total_risk`、`Nc/max(1,initial_conflict_points)`，两阶段共用初始参考。不反向调系数拟合论文结果，也不自动选尺度。现有归一化风险地图未改变，ORISK仍对每次路径栅格访问求和。
2. 局部变长路径编码未公开；本实现用FATA优化局部via xyz，再由原基础A*连接。包围盒由局部段起点、冲突cell、终点确定，默认XY余量5格、Z余量1层，取空域交集并round。局部路径窗口半径4，相近索引默认`reroute_merge_window=5`合并，重叠窗口也合并以保证拼接；ADM冲突点不合并。via包围盒只限制via，不额外裁剪A*搜索路径。没有reroute二次开关，也没有特殊中点保留原路径旁路；策略为reroute且actor被选中即执行A*。替换段按归一化弧长投影继承当前速度，保留Stage1调整及跨冲突speed效果。这些均为编码假设。
3. learning_rate=0.5 来自参考[6]；dominant_fraction=.20仍是实现假设。初始均匀`[1/3,1/3,1/3]`遵循复现要求。更新原始优势species频率后，在normalize前`maximum(P,1e-12)`仅作数值稳定，不设0.05人工探索下限。
4. 目标论文将独立匹配机制与FATA组合的完整源代码未公开。本实现按每代独立采样species、评价、选优势种群、更新P、执行FATA的MLF/LPS。固定联合变量布局保留各策略的连续块，未选块被忽略；不声称逐行复刻参考[6]原ISFS的双层嵌套循环。
5. TRC第9页描述依据剩余航路、目的地选择改航对象的原则及示例，速度示例还会调整双方，但可获得文本没有完整公开通用conflict pair actor selection算法或源代码。本轮采用二元actor决策变量：<1选plan_a，否则选plan_b，由FATA决定，不按奇偶、ID或中心性指定；这是实现假设。固定维度FATA保留去重的共享槽：每UAV一个ATD、每实际segment一个speed、每独立区域一个via块。每个候选按sampled species和actor构造`flight_strategy_requirements`，只开放该actor实际对应的策略槽；不复制每个冲突的完整FlightPlan变量。
6. Stage2几何和速度变量严格基于Stage1计划及重新检测的冲突索引，不投影回原路径。未启用的变量保留Stage1值，允许跨阶段组合。ATD绝对边界、Tdelay、最终changed仍相对初始计划；Stage2增量修改数另相对Stage1。策略历史记录参与块（可有no-op），不等于实际物理变化；实际改航数单列。
7. FATA基础更新对照根目录`FATA.m`，包括scalar-rand reset、IP、p、Para1/Para2、两相折射及全内反射；不可行几何个体fitness为inf，有限哨兵仅供MLF积分的数值处理。电池超限按式(50)罚项计入，未增加硬拒绝。每代用当前delta重评估保存的best；省略末代未被评价的位置更新，它不影响FATA.m返回的解。
8. 初始城市/OD和时间不确定性模型沿用已有工程近似，不属于本次优化方法重构范围。原生成器可能产生ETD=0，严格运行将其提升至1秒作为合法初始计划，并重写initial_plans.pkl。

性能优化：静态pair/occupancy缓存的`IncrementalConflictEvaluator`与完整检测共用同一个数学内核，分别测试确定/不确定模型各50个候选的所有Conflict字段及顺序；目标贡献缓存按原计划和栅格求和顺序，50候选full/cached fitness差<1e-10；`CandidatePlanView`仅复制选中actor，其他保持base引用；每进程持久LRU32768，key含端点、via、障碍/风险地图hash及权重，逐项淘汰，不clear全缓存；每阶段持久pool最多8进程，BLAS线程各1；坐标向量化保留交错随机数及逐行更新，经3/40/180维和真实ADM测试与标量更新同seed逐位一致，可用`paper_performance.fata_vectorized_update`切回标量。`performance_profile.csv`为包含性计时：worker时间和父进程等待重叠，percentage可超过100%；序列化行仅是pickle探针估计，不假装精确拆分IPC。

每次`run_main.py`都生成唯一run_id，自动格式为`YYYYMMDD_HHMMSS_mmm_schedulerMode_objectiveMode_seedXXXX_gitSHORTSHA`，可显式`--run-id test_001`。输出只写入`outputs/runs/<run_id>/`（`--outputs`改归档根目录，不是直接文件目录）；已有目录默认FileExistsError，只有显式`--overwrite-run`才能覆盖。所有CSV第一列run_id；PNG底部标识run_id并放`figures/`；保存`config_snapshot.yaml`、`run_manifest.json`、`stdout.log`、`final_plans.pkl`及`logs/`。成功后append `outputs/run_index.csv`并更新`outputs/latest_run.txt`，失败不登记成功索引。

`python make_fata_report.py`默认重建latest_run的报告，`--run-id`可选择已完成归档，不再从旧根目录metrics生成当前strict报告。旧legacy报告需显式`--historical-output`。

科研文件含Stage1单具体策略、Stage2逐代generation_best和selected_best原始策略/actor/当时代采样概率、飞行策略历史（各策略所选actor冲突数及最终组合）、ADM第0代起概率历史、收敛、性能profile和方法报告。normal收敛为1..200与201..400，quick为1..50与51..100；若Stage1零冲突，明确跳过Stage2。策略参与不等于物理变化，实际改航另列。metrics还记录维度、fitness评价数、A*缓存、提前/延后及绝对ATD偏移；不以定性结果强制比例或repair。

两种尺度只能以不同child run_id保存；手动完成两组相同预算实验后，`compare_objective_scales.py --raw-run-id RAW_RUN_ID --norm-run-id NORM_RUN_ID`校验相同配置、种子和初始数据（CSV比较排除run_id），写入`outputs/comparisons/<comparison_id>/objective_scale_comparison.csv`及`comparison_manifest.json`。脚本不启动实验，不自动选更好模式。开发验证只需pytest和一次quick，不自动运行normal双尺度、重复实验、敏感性、PSO或GA。旧outputs根目录文件仅作历史结果保留，不代表新代码的运行。

如果环境暂时缺少 `pyyaml` 或 `tqdm`，代码会使用内置 fallback；`pytest` 仍建议安装后运行。

## 模块说明

- `src/grid.py`：60 x 60 x 4 三维空域栅格，柱状城市障碍物，26 邻域。
- `src/risk_map.py`：按论文式 (1)-(10) 的结构实现地面风险近似，输出归一化风险图。
- `src/astar_3d.py`：改进 A*，`f(n)=g(n)+w(n)h(n)`，其中 `g` 融合风险和长度。
- `src/flight_plan.py`：生成无人机起终点、A* 路径、ETD、ETAP、速度剖面和初始计划。
- `src/conflict_detection.py`：不考虑/考虑不确定性两种冲突探测，正态分位数由 `alpha` 控制。
- `src/conflict_network.py`：冲突网络、度/接近/介数/PageRank/CI 指标和顺序攻击实验。
- `src/optimization_model.py`：地面等待、速度调整、局部改航的统一解码、评估和贪心修复。
- `src/fata.py`：原始 FATA 与佳点集初始化的改进 FATA。
- `src/baselines.py`：PSO、GA、原始 FATA 与改进 FATA 的快速对比实验。
- `src/visualization.py`：PNG 与 Plotly 离线 HTML 可视化，HTML 风格对齐参考文件。

## 公式对应与近似

风险地图保留论文的综合风险结构 `P_total = P_bal + P_par`、撞击影响概率、严重程度概率、弹道下坠和降落伞漂移项。由于论文没有给出完整城市人口场，本项目用基础人口密度、热点和建筑影响生成空间人口密度，并进行 min-max normalization 供 A* 使用。

冲突探测中，不考虑不确定性使用同一栅格过点时间差 `|t_a - t_b| <= t_conflict`。考虑不确定性时，`t_ETA ~ Normal(mu_t, sigma_t^2)`，默认 `sigma_t = sigma0 + sigma_rate * elapsed_time`，并用正态置信区间扩展安全时间间隔。

工程模式采用地面起降与空域巡航分离机制：飞行计划的起点和终点固定在 `z=0` 地面层，巡航层按 `flight_generation.altitude.cruise_level_probs` 随机采样。严格初始场景由 A* 自行决定航迹层，不施加高度偏好。可视化中航迹线使用栅格中心高度，`z=0/1/2/3` 分别显示为空域格点中心约 15/45/75/105 m；起终点 marker 单独画在 0 m 地面高度。

严格模式默认按打印式(51)计算raw目标；初始尺度归一化仅用于明确标记的对照实验。旧legacy模式保留初始尺度归一化及`risk_increase_ratio`暴露风险指标。两种严格尺度都按打印式(50)加入电池超限和延后计划数罚项，不以实验现象增加硬约束。

## 输出

新主流程输出位于 `outputs/runs/<run_id>/`；旧 `outputs/` 根目录文件保留为历史结果。以下对比baseline表仅属于legacy或单独实验，strict不自动运行：

- `metrics_summary.csv`
- `conflicts_uncertain.csv`
- `conflicts_no_uncertain.csv`
- `unresolved_conflicts.csv`
- `final_conflicts_by_pair.csv`
- `final_conflict_segments.csv`
- `key_flights.csv`
- `table_two_stage_vs_one_stage.csv`
- `table_pso_ga_fata.csv`
- `table_original_vs_improved_fata.csv`
- 多张 PNG 图，包括风险、航迹、网络、攻击曲线、收敛曲线和敏感性图
- Plotly 离线 HTML：`strategy_overview.html`、`fata_3d_before.html`、`fata_3d_after.html`

结果不是硬编码论文表格，而是由当前随机种子、配置和算法运行计算得到。

## 非均匀初始飞行计划生成（仅 Legacy Engineering）

论文没有公开原始 OD 生成代码，因此本项目采用“非均匀随机热点 OD 模型”来模拟城市低空交通中的物流站点、服务热点和中心穿越流量。该模型不使用完全均匀随机采样、网格分层采样或 Latin Hypercube，也不人为保证各区域飞行计划数量均衡。

工程配置 `flight_generation.mode=heterogeneous_random` 会生成热点 OD、时间高峰、异质速度和中心走廊重叠航迹。这属于工程扩展，不能当作论文严格随机初始场景，也不能仅凭图形相似宣称更接近作者场景。

相关诊断输出包括 `outputs/od_points.png`、`outputs/takeoff_time_hist.png`、`outputs/route_density_heatmap.png`、`outputs/route_length_hist.png` 和 `outputs/flight_generation_report.csv`。若 `route_density_gini < 0.25`，程序会提示航迹仍过于均衡，可提高 `center_bias_strength` 或 `hotspot_std_cells`。

## Paper Random Initial Scenes

`paper_strict` resolves the scene settings from `paper_scene`, using
`flight_generation.mode=paper_random`. `legacy_engineering` resolves
`legacy_scene.flight_generation`, `.astar`, and `.conflict`; missing legacy
sections inherit the engineering sections of config.yaml. The convenience
`generate_initial_flight_plans()` helper continues to generate engineering
scenes; paper callers use `generate_flight_plans()` with strict config.

The paper scene has 100 random ground OD pairs, uniform ETD in [0,1800]s,
constant speed 10m/s, risk/distance A* weights 0.8/0.2, conflict threshold
30s and alpha=0.05. Initial A* uses meter distances in g and h without
altitude penalties, random biases, forced corridors or waypoints.

Implementation assumption: "about 6km" is interpreted as 6000 +/- 1200m.
OD cell centers are sampled uniformly then rejected if occupied, outside
the fixed distance band or disconnected. Tolerance is never relaxed to fit
conflict counts. z=0 is the ground terminal index; airspace cell centers
remain at (z+0.5)*30m. No new altitude preference is imposed.
City/population generation, sigma0=1 and sigma_rate=0.01 remain assumptions;
sigma is unchanged. CI radius l=2 is also an implementation assumption.

Only initial scenes and diagnostics are run by these commands:

```bash
python analyze_paper_scene.py --seed 2025
python scan_paper_scenes.py --seed-start 2020 --seed-end 2039 --n-jobs 8
# Explicit optional calibration, based only on published initial statistics:
python scan_paper_scenes.py --seed-start 2020 --seed-end 2039 --n-jobs 8 --scene-mode paper_calibrated
```

Seed ranges are inclusive. Each scan is isolated in
`outputs/scene_scans/<scan_id>/` using the existing timestamp/mode/seed/commit
run-id convention. The manifest records the full config and all candidates;
each seed has its own diagnostics, initial plans, obstacle/risk arrays and
conflict CSVs. FATA/ADM are never executed by these tools.

`paper_strict_random` reports the requested seed without selecting another.
`paper_calibrated` selects a complete candidate seed using
`abs(det-53)/53 + abs(unc-97)/97 + abs(point_coverage-78/97)` and writes
`scene_calibration_report.md`. Final optimization results are never used.
The selected candidate's `scene_config.json`, arrays and initial pickle
record the replayable scene; use its seed explicitly for later scheduling.
It is a calibrated reproduction scene, not the exact original scene.

CI uses the unweighted simple graph and the exact l-hop boundary formula,
with fixed Top10 and flight-id tie breaking. Raw point coverage counts each
event once if either endpoint is selected. Pair coverage is separate.
Diagnostics also report unique (unordered pair, cell) counts, extra repeated
index events, degree concentration, and l=1/2/3 sensitivity without changing
the default radius. Continuous segments use the existing adjacency predicate;
only segments with more than one event enter the continuous-point ratio.
Post-removal LCC ratios use the original number of plans as denominator.
Top1/5/10 conflict shares count event unions, rather than summing duplicated
endpoint contributions.

The one-scene report compares the existing archived engineering paths
recounted at the same 30s/sigma against paper_random. Its baseline source is
recorded; absent baseline entries and undisclosed paper metrics stay blank.
