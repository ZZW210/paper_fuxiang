# Paper-strict implementation report

本轮恢复组合策略、Stage1增量基准、双端变量和原始ADM标签，默认raw_equation。初始生成、风险地图、冲突探测、复杂网络及CI实现均未修改。

| 原论文或用户要求 | 依据 | 对应代码函数 | 是否一致/近似 | 原因 |
|---|---|---|---|---|
| 两阶段、重新探测 | 目标论文3.2.1 | [paper_scheduler.optimize_paper_schedule](../src/paper_scheduler.py) | 一致 | 无 |
| 普通CI前10% | 式(29)、表4及4.3 | [conflict_network.select_paper_key_flights](../src/conflict_network.py) | 本轮未修改 | 同分ID排序；初始网络是既有数据近似 |
| 允许单独及组合策略 | 目标论文3.1.1、3.2.1优先于参考[6] | [paper_optimization._decode](../src/paper_optimization.py) | 恢复目标要求 | 参考[6]单UAV单策略仅为该参考自身假设，已停用 |
| Stage1七种非空组合 | 用户activation编码要求 | [paper_optimization.decode_activation_genes](../src/paper_optimization.py) | 编码假设 | 三gene [0,1]，>=.5激活；均未激活则取最大gene，同分ID，保证非空；不按冲突类型指定 |
| ATD连续[1,3600]，提前/延后<=1800 | 式(38)-(40)、表3 | [paper_optimization.paper_atd_bounds](../src/paper_optimization.py) | 范围一致 | Stage2范围仍相对original ETD；未匹配schedule则保留Stage1 ATD，匹配后优化新的绝对ATD |
| 连续逐航段速度[5,20] | v_ijk、式(41)、表3 | [paper_optimization._build_layout](../src/paper_optimization.py) | 范围一致；局部窗口假设 | Stage1完整profile；Stage2仅操作SPEED标签关联航段，其余Stage1速度不变；索引窗口半径4 |
| xyz改航、26邻域、端点、无障碍 | 式(35)、(37)、(42) | [paper_optimization._via_route](../src/paper_optimization.py) | via-cell+A*近似 | 变长路径编码未公开；全范围xyz取整，内部0-based；via等于原窗口中点时精确保留原路径no-op |
| 两端都开放，允许不改航 | 本轮用户要求 | [paper_optimization.PaperFlightBlock](../src/paper_optimization.py) | no-op编码假设 | Stage2每UAV局部窗口有[0,1]改航开关，由FATA决定；无owner或奇偶actor；no-op不等于实际改航 |
| 组合下保留已有速度 | 目标组合要求；映射细节未知 | [paper_optimization._via_route](../src/paper_optimization.py) | 弧长映射假设 | 相同路径精确保留profile；替换路径按归一化弧长投影当前速度，没有新速度决策 |
| Stage2在Stage1基础上增量优化 | 目标论文3.2.1 | [paper_optimization._decode](../src/paper_optimization.py) | 已修复working base | 布局使用Stage1路径及冲突索引；成本和changed仍比较initial，不回原计划重做 |
| 同flight累积多个冲突策略 | 目标组合要求，ADM本次扩展 | [paper_optimization.stage2_flight_strategies](../src/paper_optimization.py) | 共享变量编码假设 | ATD每flight共享；重叠速度段共享；重叠空间窗口共享via-cell；仅解码合并变量块，不合并或改写species |
| Tdelay绝对值、Tair、ORISK、Nc points | 式(31)-(33)、(51) | [paper_optimization.paper_objective_components](../src/paper_optimization.py) | 结构一致 | 初始生成、风险地图及冲突探测本轮未修改；ORISK是已有栅格风险而非真实概率场复原 |
| 默认raw Eq.(51) | 打印式(51)及本轮要求 | [paper_optimization.paper_fitness](../src/paper_optimization.py) | 直算打印公式，不能声称复原作者尺度 | 论文说明无量纲化但变换未知；旧约[0,1]尺度使100/1000罚项支配；不反向调系数 |
| initial-reference仅实验 | 无量纲化未知；本轮对照要求 | [paper_optimization.normalize_paper_objectives](../src/paper_optimization.py) | initial_reference_experimental假设 | 两阶段共用initial固定参考；不得跨尺度比较fitness大小或自动选更好模式 |
| 动态delta、gamma=5 | 式(49) | [paper_optimization.conflict_weight_delta](../src/paper_optimization.py) | 一致 | 每代用当前delta重评估保存best |
| Sfit打印罚项及固定权重 | 式(50)-(51)、表5 | [paper_optimization.paper_fitness](../src/paper_optimization.py) | 两种尺度均保持罚项 | wc=.8，wd=.25，wt=.25，wr=.5；nbattery=Tair>1200；ndelay=ATD>ETD+1e-6 |
| 无3%cap、changed cap、pair目标 | 模型约束与实验结果区分 | [paper_optimization._decode](../src/paper_optimization.py) | 一致 | 无百万冲突罚项、工程repair、单步rollback或结果强制比例 |
| 匹配每个剩余冲突点 | 目标3.2.1、参考[6]3.2.1 | [adm_matching.adm_fata_optimize](../src/adm_matching.py) | P形状(m,3) | 仅路径窗口共享，不合并概率矩阵或采样标签 |
| 均匀初始化、独立采样、并发评价 | 参考[6]Algorithm2及用户等概率要求 | [adm_matching.sample_strategy_species](../src/adm_matching.py) | 保留原始sampled species | 1/3为用户给定；参考可用Algorithm2未明确列出初始数值；无类型先验 |
| 原始优势种群频率更新，lrate=.5 | 参考[6]4.3及Algorithm2第18行 | [adm_matching.update_probability_matrix](../src/adm_matching.py) | 公式一致 | 无reconciliation或owner投票；dominantNo未知，用round(NP*.20)，无可行个体则不更新 |
| Stage2再次改进FATA | 目标3.2.1及3.2.2 | [adm_matching.adm_fata_optimize](../src/adm_matching.py) | 组合实现仍为近似 | 未公开替换ISFS后的源代码；每代采样、评价、更新P、MLF/LPS，不声称复刻双层ISFS |
| 佳点集和原始FATA，无Gaussian增补 | FATA.m、式(48) | [fata.fata_optimize_paper](../src/fata.py) | 本轮未改FATA | 保留scalar-rand reset、IP/p、两相折射和全反射，无噪声、warm start或Gaussian local search |
| 几何不可行解处理 | 路径约束，数值细节未知 | [paper_optimization.InfeasiblePaperRoute](../src/paper_optimization.py) | 数值假设 | 不可行fitness=inf，有限哨兵仅用于MLF积分；不回滚原路径掩盖不可行 |
| normal 50/200/200，quick 20/50/50 | 表5及用户预算 | [paper_scheduler.optimize_paper_schedule](../src/paper_scheduler.py) | 保持结构及预算 | 无剩余冲突时明确跳过Stage2；8进程只并行fitness，随机更新和ADM顺序固定 |
| 历史、提前/延后、实际改航诊断 | 本轮诊断要求 | [paper_scheduler.write_paper_diagnostics](../src/paper_scheduler.py) | 实现 | 策略计数允许重叠和no-op；实际改航另列；最终changed对initial，Stage2增量对Stage1 |
| 定性现象只做事后比较 | 本轮要求及论文实验现象 | [paper_consistency.qualitative_observations](../src/paper_consistency.py) | 不设硬约束 | 未改善项如实列出；大幅提前且ndelay=0仅标记，不限制或补repair |
| 两种尺度分别运行，不自动选择 | 本轮对照实验要求 | [compare_objective_scales.write_scale_comparison](../compare_objective_scales.py) | 独立实验目录 | 校验相同初始输入、seed和预算，CSV保留真实结果，raw仍默认 |

## 本次运行

真实参数与结果来自paper_run_config.json、paper_convergence.csv及metrics_summary.csv。

- `paper_objective_scale_mode`: raw_equation
- `seed`: 2025
- `NP`: 50
- `Ngen_max_stage1`: 200
- `Ngen_max_stage2`: 200
- `n_jobs`: 8
- `stage1_initial_conflict_points`: 130
- `stage1_final_conflict_points`: 98
- `stage2_final_conflict_points`: 18
- `changed_flight_count_two_stage`: 70
- `stage1_strategy_combination_count`: 7
- `stage2_strategy_combination_count`: 31
- `final_Tdelay`: 33692.19803057303
- `final_Tair`: 64406.43698749349
- `final_ORISK`: 1156.2379251571049
- `final_n_delay`: 22
- `final_n_battery`: 3
- `final_paper_fitness`: 844927.9789013348
- `stage1_fata_time`: 123.24280279999948
- `stage2_fata_time`: 229.5323176999991
- `total_runtime`: 361.9182828999983

## 定性现象观察

事后报告，不达预期也保留；策略参与数可重叠或no-op，实际改航另列。

- 终代罚项=5200.000，占fitness 0.6%；changed架数及Stage2增量修改架数不直接入目标，不能保证少改动；不为拟合现象新增惩罚。
- Stage1冲突点130 -> 98，减少24.6%；整体点数不等于作者连续冲突私有数据。
- Stage1 schedule-only=1/10，组合flight=7；若仍只用schedule，如实列出。
- 实际改航flight：Stage1=1，Stage2相对Stage1=15。
- Stage2冲突点98 -> 18，继续下降=True；working base为Stage1，不代表必然找到更小改动。
- 最终changed=70，少于旧版61=False；Stage2增量修改60架，新增changed 60架。
- 提前12架，延后22架；平均/最大绝对偏移336.922/1800.000s。
- n_delay=0且Tdelay>20000观察标记=False；提前也计入绝对Tdelay，数学上可出现，此标记仅诊断，不触发限制或repair。

## OLD / NEW

| 指标 | 旧版 | 本次运行 |
|---|---|---|
| 冲突点 | 130 -> 98 -> 32 | 130 -> 98 -> 18 |
| changed | 61 | 70 |
| Tdelay/s | 25188.77 | 33692.19803057303 |
| Stage1 schedule/speed/reroute | 10/0/0 | 8/9/1 |
| Stage2 schedule/speed/reroute | 23/27/1 | 26/41/40 |

## 验证范围

tests/test_paper_strict_optimization.py覆盖组合、增量ATD/profile/路径、双端no-op、原始species更新、两个fitness尺度、两次FATA、并行复现及历史输出。check_paper_strict.py任一失败返回非零；检查不证明作者未公开实现或实验数值已复原。


## Stage1固定关键计划覆盖

固定CI关键计划涉及初始冲突点32/130；其余98点两端都不参与Stage1，故Stage1整体Nc不能低于98。本次Stage1 Nc=98；该覆盖诊断不改变CI或生成数据。
