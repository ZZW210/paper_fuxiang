# Paper-strict implementation report

Run ID: 20260912_174204_152_paper_strict_raw_equation_seed2025_01f04b03
Git commit: 01f04b03fa37cceb353a37ca2ace0ffe44ef9b86

## 修改文件

run_main.py、paper_optimization.py、adm_matching.py、fata.py、conflict_detection.py、config.yaml；paper_scheduler.py仅调度编排与诊断，新增run_archive.py及paper_performance.py；make_fata_report.py改为按run_id读取归档；测试、比较脚本、README与本报告生成器同步更新。optimization_model.py无需改动，strict不调用legacy repair。

## Stage1策略逻辑

对普通CI固定前10%的每个关键计划分配一个具体策略及其对应决策变量，同步求解。strategy_gene连续[0,3]取floor并限制到0/1/2；每个候选只启用一个主要策略，非三个activation genes。

## Stage2冲突级ADM

一个remaining conflict POINT对应P的一行，每代同时独立采样全部冲突的原始标签。优势种群频率不经过flight投票或标签合并，learning=.5；改进FATA再运行一次。

## 最终多策略与actor

策略组合来自多个冲突选择同一actor但匹配不同策略，或来自两阶段累计。actor_gene<1选plan_a，>=1选plan_b，由FATA搜索，不按奇偶、ID或中心性指定。对应已选策略变量可以取原值；没有额外reroute_enable。

## 继承Stage1

CandidatePlanView由Stage1不可变base引用和选中actor的副本构成；按schedule、speed、reroute固定顺序施加修改，未涉及航段/计划保留Stage1值。最终Tdelay和changed对照initial，Stage2增量对照Stage1。

## 性能与等价性

| 优化 | 等价性验证 |
|---|---|
| 增量冲突检测，静态pair/occupancy缓存 | 不确定及确定两种检测各50候选，比较所有Conflict字段及顺序 |
| 静态目标贡献缓存 | 50候选按原计划/栅格求和顺序，full/cached fitness差<1e-10 |
| base+actor overlays，无整组deepcopy | 未修改计划保持reference；测试原计划不被修改、跨阶段继承 |
| 每进程持久LRU32768，按地图/风险/权重/端点/via区分key | cached/uncached路径与fitness一致；测试LRU逐项淘汰及环境隔离 |
| 每阶段一个持久process pool，最多8进程，BLAS各1线程 | 同seed串行/并行ADM最佳向量、标签、P、收敛一致 |
| FATA坐标向量化，保留交错随机数和逐行更新 | 3/40/180维与标量更新逐位一致，另检验真实ADM开关一致 |
| 局部via范围、actor与区域合并 | 是本轮方法/编码变更，不宣称与上轮全空域双端编码等价 |

性能profile采用包含性计时，不是互斥分段；序列化行是pickle探针估计。不得以减NP、Ngen、冲突点或禁用改航提速。

## 实现假设

1. 目标论文3.2.1优先：Stage1对每个关键计划分配一个具体策略及对应变量并同步求解；最终多策略只来自不同Stage2冲突匹配和两阶段累计，不移植参考[6]的全局单UAV单策略限制。
2. 目标论文和参考[6]未完整公开conflict pair actor selection源代码。TRC第9页描述依据剩余航程、目的地选择改航对象的原则及例子，速度示例还会调整双方，但没有统一可执行的选择算法。按本轮要求采用二元actor_gene∈[0,2]，<1选plan_a，否则选plan_b，由FATA搜索；这是实现假设而非论文参数。
3. FATA使用固定维度去重槽：每架UAV一个ATD，每实际航段一个speed，每局部区域一个via块。每个候选先按原始sampled species和actor构造flight_strategy_requirements，仅对应槽生效；不因多个冲突复制整套FlightPlan变量，也不重写采样标签。
4. 局部改航编码未公开。本实现用路径索引窗口半径4，reroute_merge_window=5合并相近区域，重叠窗口也合并以保证拼接；ADM仍逐冲突点。根据区域端点和冲突cell建立局部via包围盒，默认XY余量5格、Z余量1层；基础A*保持原样，连接路径本身不额外裁剪在该via包围盒内。
5. 采样reroute且选中actor即执行via+A*，没有二次enable开关或特殊原窗口中点旁路。A*可产生与原路径相同的合法几何结果，不等于禁用策略。改航段速度按归一化弧长继承当前速度，保留Stage1及当前候选已施加的speed调整；应用顺序是schedule、speed、reroute，再重算ETA、风险和冲突。
6. dominant_fraction=.20未在可获得文本中明确给出，仍是实现假设。learning_rate=.5来自参考[6]；P初始化均匀，每次更新用原始优势species，normalize前仅以1e-12稳定epsilon防止数值塌缩，无0.05探索下限。
7. 作者无量纲化细节未知；raw_equation默认直算打印式(49)-(51)，initial_reference_experimental仍仅作实现假设对照，固定初始参考不变，绝不自动选优或调系数拟合。
8. 初始场景、OD生成、风险地图、基础A*、时间不确定性模型、复杂网络与普通CI不改。已有严格运行ETD=0提升到1秒的合法初始处理保留，normal保持50/200/200，quick仅20/50/50。
9. 性能计时是包含关系：worker计算时间求和可与父进程等待时间重叠，percentage可超过100%，不能加总为互斥耗时。multiprocessing_serialization是实际任务pickle探针估计，不是executor内部序列化或IPC时间的精确拆分。

## 本次quick或normal结果

quick=True；只有实际运行结果，不自动运行normal对照。

- run_id: 20260912_174204_152_paper_strict_raw_equation_seed2025_01f04b03
- seed: 2025
- NP: 20
- Ngen_max_stage1: 50
- Ngen_max_stage2: 50
- n_jobs: 8
- paper_objective_scale_mode: raw_equation
- stage1_dimension: 619
- stage2_dimension: 1413
- stage1_fitness_evaluations: 1049
- stage2_fitness_evaluations: 1049
- final_Tdelay: 37181.028317315606
- final_Tair: 59943.45300454895
- final_ORISK: 1141.88303954963
- final_n_delay: 27
- final_n_battery: 0
- final_paper_fitness: 611900.4186559017
- astar_calls: 16710.0
- astar_cache_hit_rate: 0.3825746378953591
- runtime_stage1: 10.503382999999303
- runtime_stage2: 13.61362260000169
- total_runtime: 31.05894279999848

- 冲突点：130 -> 102 -> 14。
- Stage1每架仅一个策略；最终组合UAV=14（Stage2自身组合，不含仅跨阶段新增组合）。
- 实际改航：Stage1=1，Stage2相对Stage1=16。
- changed=49，Stage2增量修改=41；这些数量不直接入目标，不强制少改动或零冲突。
- 提前/延后=2/27；平均/最大绝对ATD偏移=371.8102831731561/1550.4985675612752秒。
- n_delay=0且Tdelay>20000诊断=False；提前也累积到绝对Tdelay，不触发限制或repair。

固定关键计划涉及初始32/130点；未涉及的98点是Stage1整体Nc下限，不改CI扩大覆盖。

## 可追溯输出

每个CSV第一列run_id；PNG底部Run ID并放figures；run_manifest.json、config_snapshot.yaml和stdout.log记录实际参数与环境；成功后append run_index并更新latest_run。默认拒绝已存在run目录，只有显式--overwrite-run覆盖。比较必须使用两个child run_id和独立comparison_id。

## 尚无法确认

作者的局部变长路径编码、actor通用算法、dominant_fraction、无量纲化、独立匹配与FATA的完整源代码未公开，不能声称逐行或私有场景数值复原。

## 验证记录

本次执行pytest和一次独立quick；详细验证记录见logs/verification.json（由本次开发验证写入）。
