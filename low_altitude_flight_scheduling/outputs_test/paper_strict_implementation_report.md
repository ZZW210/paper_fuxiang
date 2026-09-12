# Paper-strict implementation report

依据：目标论文PDF第9-16页及paper_cn.txt；参考[6]PDF第10-11、18-20页及文字版；根目录FATA.m和FATA原始论文DOCX。

此报告区分论文明确内容、用户要求和实现假设。方法结构复现不等于作者私有数据的逐点复刻，也不保证零冲突。

| 原论文或用户要求 | 原文依据 | 对应代码文件和函数（src/） | 是否完全一致/是否近似 | 信息不足及原因 |
|---|---|---|---|---|
| 两阶段，阶段间重新探测 | 目标论文 3.2.1，第10页 | [paper_scheduler.optimize_paper_schedule](../src/paper_scheduler.py) | 一致 | 无 |
| 普通CI固定前10%，100架选10架 | 目标论文式(29)、表4及4.3 | [conflict_network.select_paper_key_flights](../src/conflict_network.py) | 一致；同分ID排序是实现假设 | 未公开同分排序 |
| Stage1同步求解策略与变量 | 目标论文3.2.1 | [paper_optimization.build_stage1_decision_layout / decode_stage1_solution](../src/paper_optimization.py) | 混合策略基因是必要编码近似 | 没有公开源代码或混合变量编码 |
| 每架UAV只能选一种策略 | 参考[6]3.2.1，第11页；用户明确要求 | [paper_optimization._decode / adm_matching.enforce_single_strategy_per_flight](../src/paper_optimization.py) | 遵循用户与参考[6]；目标论文允许组合，并非完全等同 | 目标论文3.1.1和3.2.1明确允许组合，此实现采用用户限定 |
| ATD连续且允许提前 | 目标论文3.1.1，式(38)-(40)、表3 | [paper_optimization.paper_atd_bounds / _decode](../src/paper_optimization.py) | 一致，[1,3600]，绝对偏移<=1800 | 无 |
| 逐航段连续速度[5,20] | 目标论文v_ijk，式(41)、表3 | [paper_optimization._build_layout / _recompute_paper_timing](../src/paper_optimization.py) | 一致；Stage2局部范围属实现假设 | 局部变量窗口宽度未公开，使用路径索引半径4 |
| 改航xyz，26邻域、无障碍、端点不变 | 目标论文式(35)、(37)、(42) | [paper_optimization._via_route / _valid_route](../src/paper_optimization.py) | via-cell+A*是必要近似 | 论文未公开变长路径编码；xyz全范围搜索后取整，内部0-based |
| 重新计算ETA、Tair和ORISK | 目标论文式(31)-(33) | [paper_optimization._recompute_paper_timing / paper_objective_components](../src/paper_optimization.py) | 一致于现有风险地图 | 现有风险地图本身是此前的归一化城市风险近似，不是真实伤亡概率场 |
| Tdelay使用绝对值 | 目标论文式(31) | [paper_optimization.paper_objective_components](../src/paper_optimization.py) | 一致 | 无 |
| Nc为conflict points，不是pairs | 目标论文3.1.2、式(51) | [paper_optimization._evaluate](../src/paper_optimization.py) | 一致于现有conflict point探测器 | 过点不确定性模型沿用现有实现，原文没有完整仿真数据 |
| 无量纲化 | 目标论文3.1.2说明但未给出精确变换 | [paper_optimization.PaperReference / normalize_paper_objectives](../src/paper_optimization.py) | 必要近似：固定initial-reference | 两阶段共用初始总空中时间/总风险/冲突数量；延误尺度为N*1800；零参考量使用epsilon |
| 动态delta，gamma=5 | 目标论文式(49) | [paper_optimization.conflict_weight_delta / fata.fata_optimize_paper](../src/paper_optimization.py) | 一致 | 每代以当前delta重评估保存的best，避免不同代fitness混比 |
| Sfit=fOBJ+1000*nbattery+100*ndelay | 目标论文式(50) | [paper_optimization.paper_fitness](../src/paper_optimization.py) | 一致 | 无 |
| fOBJ加权结构 | 目标论文式(51)、表5 | [paper_optimization.paper_fitness](../src/paper_optimization.py) | 结构一致，尺度属必要近似 | wc=.8，wd=.25，wt=.25，wr=.5；无额外conflict-pair目标或百万罚项 |
| nbattery：飞行时间>1200 | 目标论文表1、式(50) | [paper_optimization.paper_objective_components](../src/paper_optimization.py) | 一致按用户要求作为罚项 | 式(36)同时列为约束；实现使用式(50)软罚，不拒绝超电池个体 |
| ndelay：只计晚于ETD的计划 | 目标论文延误计划数及用户eps定义 | [paper_optimization.paper_objective_components](../src/paper_optimization.py) | 按用户给定eps=1e-6 | 原文未公开浮点容差；提前通过绝对值计成本但不算延误 |
| 不把3%实验结果设为约束 | 目标论文仿真结果与模型约束的区分 | [paper_optimization.paper_fitness / _decode](../src/paper_optimization.py) | 一致，无delay cap或changed ratio约束 | 无 |
| Stage2匹配每个剩余冲突点 | 目标论文3.2.1，参考[6]3.2.1 | [adm_matching.adm_fata_optimize](../src/adm_matching.py) | 一致，矩阵形状(m,3) | 路径窗口可以合并重叠以拼接；概率矩阵始终保留每个原始冲突点 |
| 初始策略均匀、无冲突类型先验 | 参考[6]Algorithm2概率初始化；用户要求等概率 | [adm_matching.initialize_probability_matrix](../src/adm_matching.py) | 遵循用户规定；不额外声称原文明确给出1/3 | 可获得Algorithm2未列出初始数值 |
| 独立随机采样，全部冲突同时评价 | 参考[6]3.2.1、Algorithm2 | [adm_matching.sample_strategy_species / adm_fata_optimize](../src/adm_matching.py) | 一致于独立采样及整体评价思想 | 采样后执行确定性UAV一致性协调，协调后的标签不再统计独立 |
| 优势种群更新P，lrate=.5 | 参考[6]4.3及Algorithm2第18行 | [adm_matching.update_probability_matrix](../src/adm_matching.py) | 更新公式及学习率一致；dominant比例属假设 | dominantNo未明确公开，使用round(NP*.20)，至少1；无可行种群时不更新 |
| ADM与改进FATA耦合 | 目标论文3.2.1引用[6]并再次求解，3.2.2指定FATA | [adm_matching.adm_fata_optimize / fata.fata_optimize_paper](../src/adm_matching.py) | 必要近似：每代一个ADM更新和一个FATA更新 | 参考[6]是双层ISFS；目标论文未公开替换后的完整源代码；不声称逐行复刻其嵌套循环 |
| 同一flight涉及多冲突的一致性 | 参考[6]单策略要求；合并规则未公开 | [adm_matching.enforce_single_strategy_per_flight](../src/adm_matching.py) | 必要近似：轮换端点owner、P加权采样标签投票、ID打破同分 | 对P直接argmax会在均匀初始化下把所有种群变成scheduling并忽略采样，所以只给采样标签投票 |
| 两阶段不累加不同策略 | 用户要求最终候选一架UAV一策略 | [paper_optimization.decode_stage2_solution / _decode](../src/paper_optimization.py) | 必要近似：被Stage2操作的flight从原计划解码 | 未公开跨阶段策略冲突处理；无owner的flight保留Stage1；原路径冲突局部索引用最近栅格投影 |
| MLF/LPS遵循FATA.m | 根目录FATA.m主循环 | [fata.fata_optimize_paper](../src/fata.py) | 尽量逐行对应，基础更新一致 | MATLAB reset的scalar rand原样保留；末代未评价更新省略；不保证跨语言RNG逐点相同 |
| 佳点集无噪声 | 目标论文式(48) | [fata.good_point_set / fata_optimize_paper](../src/fata.py) | 一致，最小满足条件素数 | 无warm start或附加Gaussian噪声 |
| 无Gaussian local search或额外repair | 目标论文仅有佳点集与delta改进；用户要求 | [fata.fata_optimize_paper / paper_scheduler.optimize_paper_schedule](../src/fata.py) | 一致 | 只保存群体global best，不要求候选冲突单调下降 |
| 几何不可行解处理 | 目标论文式(35)、(42)，数值细节未公开 | [paper_optimization.InfeasiblePaperRoute / fata.fata_optimize_paper](../src/paper_optimization.py) | 必要数值处理 | 不可行返回inf；有限哨兵仅用于MLF积分，不加入目标函数；不回滚到原路径掩盖不可行 |
| NP=50，两阶段各200，Parf=.2 | 目标论文表5及图15阶段切换 | [config.yaml / paper_scheduler.optimize_paper_schedule](../src/config.py) | 默认normal一致；quick=20/50/50 | Stage1无剩余冲突时明确跳过Stage2；quick仅缩减预算 |
| fitness并行且seed复现 | 用户CPU/并行要求；原文8核实验 | [fata.ProcessPoolExecutor / fata_optimize_paper](../src/fata.py) | 8-worker ordered map，随机更新单线程 | 源代码随机数只在主进程使用；测试验证n_jobs=1/8，包括真实ADM与路线解码 |
| 科研输出及阶段切换线 | 用户诊断要求，目标论文图15 | [paper_scheduler.write_paper_diagnostics](../src/paper_scheduler.py) | 实现 | normal全局代1..200/201..400；quick为1..50/51..100；策略概率为终代，标签是最佳个体而非终代argmax |
| 初始数据与冲突窗口 | 目标论文表1及旧项目生成器 | [paper_scheduler.run_paper_main](../src/paper_scheduler.py) | 优化窗口恢复30s；初始数据仍属旧项目近似 | 保持当前非均匀数据生成；将0秒ETD提升至1秒；未扩展本次任务至城市风险和OD的重新复现 |
| 旧工程模式保留 | 用户要求 | [run_main.main / config.yaml](../src/run_main.py) | 实现legacy_engineering | 旧FATA、weighted CI、adaptive coverage及repair接口保留；严格路径独立模块 |

## 本次运行

运行参数和真实结果来自paper_run_config.json、paper_convergence.csv及metrics_summary.csv，不以论文实验结果设置隐藏约束。

- `scheduler_mode`: paper_strict
- `seed`: 2025
- `NP`: 20
- `Ngen_max_stage1`: 50
- `Ngen_max_stage2`: 50
- `n_jobs`: 8
- `stage2_skipped`: False
- `stage1_initial_conflict_points`: 1
- `stage1_final_conflict_points`: 1
- `stage2_final_conflict_points`: 0
- `final_Tdelay`: 0.0
- `final_Tair`: 10304.398453018512
- `final_ORISK`: 188.85584151142808
- `final_n_delay`: 0
- `final_n_battery`: 0
- `final_paper_fitness`: 0.14716670784569638
- `stage1_fata_time`: 8.341819600000235
- `stage2_fata_time`: 7.893297200000234
- `total_runtime`: 19.93533920000118

## 验证范围

tests/test_paper_strict_optimization.py覆盖变量边界、绝对延误、逐段ETA、改航可行性、式(49)-(51)、ADM更新和单策略规则、Stage2再次调用FATA、无工程修复调用、无Gaussian改进、并行可复现。

check_paper_strict.py执行数值与代码路径一致性检查，任一失败返回非零。该检查验证实现约定，不证明未公开的作者实现或论文实验数值。
