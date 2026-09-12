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
python check_paper_strict.py
python run_main.py --scheduler-mode legacy_engineering --seed 2025 --outputs outputs_legacy
```

目标论文明确给出两阶段优化：普通CI排名固定前10%的关键飞行计划整体优化；重新探测剩余冲突点，再进行独立匹配和改进FATA求解。三策略是错峰起飞、速度调整、局部改航。严格模式Stage1以混合策略基因同步求解，每条关键计划选一种策略；Stage2按每个剩余冲突点建立`(m,3)`概率矩阵，独立采样一整个种群的策略组合，同时解码并评价所有计划。

目标论文给出`ATD∈[1,3600]`、`|ETD-ATD|≤1800`、逐航段`v∈[5,20]`、电池时间1200秒，以及佳点集、动态delta和式(49)-(51)的适应度结构。严格模式使用连续ATD，允许提前起飞；完整逐航段速度用于Stage1，局部连续速度用于Stage2；FATA搜索局部via-cell的xyz，经A*连接为26邻域无障碍路径。冲突窗口使用目标论文表1的30秒，旧工程的20秒校准仅属于legacy。

normal默认`Parf=.2, NP=50`，两阶段各200代，权重`wc=.8, wd=.25, wr=.5, wt=.25, gamma=5`。quick仅改为`NP=20`、各50代，不改变策略、连续变量、ADM或目标函数。Stage1无剩余冲突时明确记录跳过Stage2。严格路径不使用离散候选、冲突对罚项、百万级冲突罚项、3%延误上限、changed ratio上限、工程repair、单步rollback、warm start、佳点集Gaussian噪声或Gaussian local search。默认8进程只并行fitness评价，位置更新、ADM、随机采样和best选择都按固定顺序在主进程进行；`--n-jobs 1`可用于串行对照。

参考[6]明确给出策略概率矩阵、逐冲突独立随机分配策略、优势种群、概率更新`P_new=(1-lrate)*P_old+lrate*dominant_frequency`、`lrate=.5`、同时处理全部冲突和单架UAV仅用一种策略。其原优化器为ISFS；本实现采用目标论文指定的改进FATA。

### Implementation assumptions

1. 论文明确说明进行了无量纲化，但未公开具体变换，本项目使用固定 initial-reference normalization。目标函数结构仍严格保持式(49)-(51)。`Tdelay/(N*1800)`、`Tair/initial_total_air_time`、`ORISK/initial_total_risk`、`Nc/max(1,initial_conflict_points)`；两个阶段共用相同参考尺度，零空中时间/风险尺度使用机器epsilon。现有风险地图为旧项目归一化风险近似，ORISK仍对每次路径栅格访问求和，并非复原了真实伤亡概率场。
2. 目标论文未公开局部改航的具体变长路径编码方式，因此严格模式使用 FATA 优化局部 via-cell + A* 保证可行性；这是实现层面的近似，而不是论文明确给出的编码。xyz完整范围搜索后四舍五入，内部坐标为0-based。局部窗口使用冲突路径索引半径4；只合并重叠路径窗口用于可行拼接，不合并ADM冲突点。改航插入段继承被替换段速度，避免引入第二个策略。
3. learning_rate=0.5 来自参考[6]；dominant_fraction 未在可获得论文文本中明确给出，0.20 为实现假设。初始均匀`[1/3,1/3,1/3]`遵循本次复现要求，Algorithm2本身未明确列出初始数值。
4. 目标论文将独立匹配机制与FATA组合的完整源代码未公开。本实现按每代独立采样species、评价、选优势种群、更新P、执行FATA的MLF/LPS。固定联合变量布局保留各策略的连续块，未选块被忽略；不声称逐行复刻参考[6]原ISFS的双层嵌套循环。
5. 多个冲突涉及同一UAV时的协调未公开。每个冲突按固定顺序轮换选择其两个端点之一为target flight，避免低ID端点始终承担调整。对同一target的采样标签以对应P加权投票，argmax确定单策略，同分使用策略ID；其所属冲突统一采用该策略。仅给被采样标签投票，使均匀初始化仍有探索能力；直接对P求argmax会忽略随机species并将全部UAV固定成scheduling。协调前采样独立，协调后标签受单UAV策略一致性约束。
6. 为避免跨阶段累加delay+speed+reroute，Stage2被操作UAV从原始计划重新解码成一种策略；未操作者保留Stage1计划。若Stage1改航导致剩余冲突栅格不在原路径，用最近栅格投影定位局部变量。此跨阶段一致性处理是实现假设。目标论文3.1.1及3.2.1允许策略组合，本次按用户要求和参考[6]实施单策略限制，不能称为目标论文的额外硬约束。
7. FATA基础更新对照根目录`FATA.m`，包括scalar-rand reset、IP、p、Para1/Para2、两相折射及全内反射；不可行几何个体fitness为inf，有限哨兵仅供MLF积分的数值处理。电池超限按式(50)罚项计入，未增加硬拒绝。每代用当前delta重评估保存的best；省略末代未被评价的位置更新，它不影响FATA.m返回的解。
8. 初始城市/OD和时间不确定性模型沿用已有工程近似，不属于本次优化方法重构范围。原生成器可能产生ETD=0，严格运行将其提升至1秒作为合法初始计划，并重写initial_plans.pkl。

新增科研文件：`paper_stage1_strategy_assignment.csv`、`paper_stage2_conflict_strategy.csv`、`adm_probability_history.csv`（含第0代）、`paper_convergence.csv/png`、`paper_run_config.json`和`paper_strict_implementation_report.md`。normal收敛代为1..200和201..400，在200代画阶段切换虚线；quick切换为50。策略CSV中的概率是终代P，selected_strategy为最终保存最佳species经一致性处理后的标签，未必等于终代P的argmax。`metrics_summary.csv`记录各阶段冲突点、策略数、FATA时间和最终原始目标分量。旧比较/敏感性文件若仍存在，仅代表之前独立实验；严格主流程不把旧baseline表伪装成本次结果。

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

适应度中的延误、空中时间和风险均按初始计划尺度归一化，冲突项使用剩余冲突数量和总空中时间归一化，避免风险和时间量纲压倒冲突项。`risk_increase_ratio` 使用归一化风险增量除以按 10 m 采样的总飞行距离暴露量，表示平均暴露风险增量。电池约束和延误计划数作为罚项加入。

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
