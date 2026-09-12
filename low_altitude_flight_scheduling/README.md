# 基于复杂网络的城市低空飞行计划优化调度：算法级复现

本项目复现论文《基于复杂网络的城市低空飞行计划优化调度》的主要算法流程：三维栅格空域、城市低空第三方风险地图、改进 A* 四维飞行计划生成、考虑过点时间不确定性的冲突探测、冲突复杂网络关键飞行计划识别、多策略两阶段优化，以及原始/改进 FATA、PSO、GA 对比。

这是算法级复现，不是作者私有随机数据的逐点复刻。论文未公开完整仿真数据、随机种子、障碍物生成、人口密度细节和过点时间误差函数，因此本项目使用可配置随机城市与交通脉冲生成器，并在 `outputs/calibration_report.md` 中记录默认冲突数与论文现象的差异。

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
python run_main.py --scheduler-mode paper_strict --paper-objective-scale-mode initial_reference_experimental --seed 2025 --outputs outputs_initial_reference_experimental
python compare_objective_scales.py
python check_paper_strict.py
python run_main.py --scheduler-mode legacy_engineering --seed 2025 --outputs outputs_legacy
```

目标论文明确给出两阶段优化：普通CI排名固定前10%的关键飞行计划整体优化；重新探测剩余冲突点，再进行独立匹配和改进FATA求解。三策略是错峰起飞、速度调整、局部改航。Stage1每架关键UAV使用三个连续激活基因，以0.5为阈值，允许七种非空组合；全部低于阈值时启用最大基因对应策略，同分按策略ID。Stage2按每个剩余冲突点建立`(m,3)`概率矩阵，每个冲突独立采样一种策略并开放两端UAV对应变量；共享UAV可累积多种策略块，原始采样标签始终保留，不投票或重写。

目标论文给出`ATD∈[1,3600]`、`|ETD-ATD|≤1800`、逐航段`v∈[5,20]`、电池时间1200秒，以及佳点集、动态delta和式(49)-(51)的适应度结构。严格模式使用连续ATD，允许提前起飞；完整逐航段速度用于Stage1，局部连续速度用于Stage2；FATA搜索局部via-cell的xyz，经A*连接为26邻域无障碍路径。冲突窗口使用目标论文表1的30秒，旧工程的20秒校准仅属于legacy。

normal默认`Parf=.2, NP=50`，两阶段各200代，权重`wc=.8, wd=.25, wr=.5, wt=.25, gamma=5`。quick仅改为`NP=20`、各50代，不改变策略、连续变量、ADM或目标函数。Stage1无剩余冲突时明确记录跳过Stage2。严格路径不使用离散候选、冲突对罚项、百万级冲突罚项、3%延误上限、changed ratio上限、工程repair、单步rollback、warm start、佳点集Gaussian噪声或Gaussian local search。默认8进程只并行fitness评价，位置更新、ADM、随机采样和best选择都按固定顺序在主进程进行；`--n-jobs 1`可用于串行对照。

参考[6]给出策略概率矩阵、逐冲突独立随机分配策略、优势种群、概率更新`P_new=(1-lrate)*P_old+lrate*dominant_frequency`及`lrate=.5`。其单UAV单策略限制不移植到目标论文：目标论文3.1.1和3.2.1允许组合。本实现用原始优势species逐冲突标签计算频率，采用目标论文指定的改进FATA而非参考[6]的ISFS。

### Implementation assumptions

旧约[0,1]目标直接加100/1000罚项时容易被罚项支配，且与论文公布的fitness量级不符。因此默认直算打印式(51)，initial-reference仅作实现假设对照；这不表示作者未公开的无量纲化已经复原。

1. 默认`optimization.paper_objective_scale_mode: raw_equation`直接计算打印式(51)：`(1-wc)*(wd*Tdelay+wt*Tair+wr*ORISK)+wc*delta*Nc*Tair+1000*n_battery+100*n_delay`。论文说明无量纲化但变换未公开，因此不能声称此尺度复原了作者实现。旧固定初始归一化仅作为`initial_reference_experimental`保留：`Tdelay/(N*1800)`、`Tair/initial_total_air_time`、`ORISK/initial_total_risk`、`Nc/max(1,initial_conflict_points)`，两阶段共用初始参考。不反向调系数拟合论文结果，也不自动选尺度。现有归一化风险地图未改变，ORISK仍对每次路径栅格访问求和。
2. 局部变长路径编码未公开；本实现用FATA优化via-cell xyz，再由A*连接。xyz完整范围搜索后四舍五入，内部坐标0-based。局部窗口索引半径4，只合并重叠路径窗口，不合并ADM冲突点。via恰好等于原窗口中点时保留原路径；Stage2每个路径窗口另有一个FATA搜索的连续no-op激活基因（阈值0.5），让两端都可保持原路径。这些都是编码假设。替换段以归一化弧长投影继承当前速度，保留Stage1及组合内的逐段速度调整，不新增速度决策。
3. learning_rate=0.5 来自参考[6]；dominant_fraction 未在可获得论文文本中明确给出，0.20 为实现假设。初始均匀`[1/3,1/3,1/3]`遵循本次复现要求，Algorithm2本身未明确列出初始数值。
4. 目标论文将独立匹配机制与FATA组合的完整源代码未公开。本实现按每代独立采样species、评价、选优势种群、更新P、执行FATA的MLF/LPS。固定联合变量布局保留各策略的连续块，未选块被忽略；不声称逐行复刻参考[6]原ISFS的双层嵌套循环。
5. 同一UAV关联多个冲突时，各点的原始标签只决定开放哪些共享变量块，允许schedule+speed+reroute组合。速度只更新被采样为speed的冲突局部航段，改航只启用reroute标签关联且激活的窗口。两端开放不等于两端强制修改：ATD、逐段速度原值及不改航均合法，由FATA决定修改一端、两端或都不改；无固定owner或actor。
6. Stage2几何和速度变量严格基于Stage1计划及重新检测的冲突索引，不投影回原路径。未启用的变量保留Stage1值，允许跨阶段组合。ATD绝对边界、Tdelay、最终changed仍相对初始计划；Stage2增量修改数另相对Stage1。策略历史记录参与块（可有no-op），不等于实际物理变化；实际改航数单列。
7. FATA基础更新对照根目录`FATA.m`，包括scalar-rand reset、IP、p、Para1/Para2、两相折射及全内反射；不可行几何个体fitness为inf，有限哨兵仅供MLF积分的数值处理。电池超限按式(50)罚项计入，未增加硬拒绝。每代用当前delta重评估保存的best；省略末代未被评价的位置更新，它不影响FATA.m返回的解。
8. 初始城市/OD和时间不确定性模型沿用已有工程近似，不属于本次优化方法重构范围。原生成器可能产生ETD=0，严格运行将其提升至1秒作为合法初始计划，并重写initial_plans.pkl。

新增科研文件：`paper_stage1_strategy_assignment.csv`（组合以`+`连接）、`paper_stage2_conflict_strategy.csv`（最佳原始species）、`paper_flight_strategy_history.csv`、`adm_probability_history.csv`（含第0代）、`paper_convergence.csv/png`、`paper_run_config.json`和`paper_strict_implementation_report.md`。normal收敛代为1..200和201..400，阶段切换虚线为200；quick为50。策略CSV中的概率是终代P，selected_strategy未必等于终代P的argmax。metrics记录组合参与数、实际改航数、Stage2增量修改数、提前/延后架数及平均/最大绝对ATD偏移；`n_delay=0`且`Tdelay>20000`只做诊断，不触发限制。两组normal完成后运行`compare_objective_scales.py`，校验相同配置、种子和初始文件SHA256，并写`outputs/objective_scale_comparison.csv`；不同尺度fitness不可直接比较。旧比较/敏感性文件仅代表此前独立实验。

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

默认采用地面起降与空域巡航分离机制：飞行计划的起点和终点固定在 `z=0` 地面层，巡航层按 `flight_generation.altitude.cruise_level_probs` 随机采样。可视化中航迹线使用栅格中心高度，`z=0/1/2/3` 分别显示为空域格点中心约 15/45/75/105 m；起终点 marker 单独画在 0 m 地面高度。

严格模式默认按打印式(51)计算raw目标；初始尺度归一化仅用于明确标记的对照实验。旧legacy模式保留初始尺度归一化及`risk_increase_ratio`暴露风险指标。两种严格尺度都按打印式(50)加入电池超限和延后计划数罚项，不以实验现象增加硬约束。

## 输出

主流程输出位于 `outputs/`：

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

## 非均匀初始飞行计划生成

论文没有公开原始 OD 生成代码，因此本项目采用“非均匀随机热点 OD 模型”来模拟城市低空交通中的物流站点、服务热点和中心穿越流量。该模型不使用完全均匀随机采样、网格分层采样或 Latin Hypercube，也不人为保证各区域飞行计划数量均衡。

默认配置 `flight_generation.mode=heterogeneous_random` 会生成热点 OD、时间高峰、异质速度和中心走廊重叠航迹：约 55% 航班在热点之间飞行，约 25% 从边缘穿越中心区域，约 20% 保留背景随机流量；起飞时间由多个高斯峰和少量均匀噪声组成。该方法比均匀随机更接近论文仿真图中表现出的冲突不均匀、中心区域航迹密集和连续重叠航段现象。

相关诊断输出包括 `outputs/od_points.png`、`outputs/takeoff_time_hist.png`、`outputs/route_density_heatmap.png`、`outputs/route_length_hist.png` 和 `outputs/flight_generation_report.csv`。若 `route_density_gini < 0.25`，程序会提示航迹仍过于均衡，可提高 `center_bias_strength` 或 `hotspot_std_cells`。
