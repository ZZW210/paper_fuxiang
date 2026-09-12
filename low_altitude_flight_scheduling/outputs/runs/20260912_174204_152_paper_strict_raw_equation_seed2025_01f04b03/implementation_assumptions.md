# Implementation assumptions

目标论文3.2.1优先：Stage1对每个关键计划分配一个具体策略及对应变量并同步求解；最终多策略只来自不同Stage2冲突匹配和两阶段累计，不移植参考[6]的全局单UAV单策略限制。

目标论文和参考[6]未完整公开conflict pair actor selection源代码。TRC第9页描述依据剩余航程、目的地选择改航对象的原则及例子，速度示例还会调整双方，但没有统一可执行的选择算法。按本轮要求采用二元actor_gene∈[0,2]，<1选plan_a，否则选plan_b，由FATA搜索；这是实现假设而非论文参数。

FATA使用固定维度去重槽：每架UAV一个ATD，每实际航段一个speed，每局部区域一个via块。每个候选先按原始sampled species和actor构造flight_strategy_requirements，仅对应槽生效；不因多个冲突复制整套FlightPlan变量，也不重写采样标签。

局部改航编码未公开。本实现用路径索引窗口半径4，reroute_merge_window=5合并相近区域，重叠窗口也合并以保证拼接；ADM仍逐冲突点。根据区域端点和冲突cell建立局部via包围盒，默认XY余量5格、Z余量1层；基础A*保持原样，连接路径本身不额外裁剪在该via包围盒内。

采样reroute且选中actor即执行via+A*，没有二次enable开关或特殊原窗口中点旁路。A*可产生与原路径相同的合法几何结果，不等于禁用策略。改航段速度按归一化弧长继承当前速度，保留Stage1及当前候选已施加的speed调整；应用顺序是schedule、speed、reroute，再重算ETA、风险和冲突。

dominant_fraction=.20未在可获得文本中明确给出，仍是实现假设。learning_rate=.5来自参考[6]；P初始化均匀，每次更新用原始优势species，normalize前仅以1e-12稳定epsilon防止数值塌缩，无0.05探索下限。

作者无量纲化细节未知；raw_equation默认直算打印式(49)-(51)，initial_reference_experimental仍仅作实现假设对照，固定初始参考不变，绝不自动选优或调系数拟合。

初始场景、OD生成、风险地图、基础A*、时间不确定性模型、复杂网络与普通CI不改。已有严格运行ETD=0提升到1秒的合法初始处理保留，normal保持50/200/200，quick仅20/50/50。

性能计时是包含关系：worker计算时间求和可与父进程等待时间重叠，percentage可超过100%，不能加总为互斥耗时。multiprocessing_serialization是实际任务pickle探针估计，不是executor内部序列化或IPC时间的精确拆分。
