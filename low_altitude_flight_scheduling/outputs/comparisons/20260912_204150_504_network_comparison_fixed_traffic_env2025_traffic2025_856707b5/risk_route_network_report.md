# 人口风险—A*—冲突网络结构比较

Comparison ID: `20260912_204150_504_network_comparison_fixed_traffic_env2025_traffic2025_856707b5`

环境种子=2025；交通种子=2025；飞行计划数=100。
四组共享完全相同的建筑数组、OD、ETD 和初始速度；各组重新运行 A*，不重抽交通。
参数 beta=2.0，中心数设定=4，实际=4。
没有执行 Stage1、Stage2、FATA、ADM 或调度目标函数，没有按公开结果选择参数。

A: old_gaussian + meter（仅旧人口生成器对照；同一风险计算与归一化）。
B/C/D: reference28_gravity + meter/grid/kilometer。
默认尺度仍是 meter；全部使用 alpha_r=0.8、alpha_L=0.2。

## 论文公开值（仅 comparison）

deterministic conflicts=53；uncertain conflicts=97；CI Top10 point coverage=78/97≈0.804124。
来源：[目标论文](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2025.31479)。
这些数值不参与中心检测、beta 选择、路径规划、CI 排序或种子选择。

| group | det_Nc | unc_Nc | network_edges | max_degree | degree_gini | ci_top10_coverage | continuous_conflict_ratio | route_density_gini | route_cell_reuse_p90 | route_cell_reuse_max | routes_different_from_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 74 | 123 | 84 | 6 | 0.446786 | 0.365854 | 0.422764 | 0.583793 | 3.000000 | 9 | 0 |
| B | 75 | 137 | 85 | 6 | 0.450000 | 0.357664 | 0.481752 | 0.584082 | 3.000000 | 9 | 46 |
| C | 138 | 207 | 134 | 9 | 0.339851 | 0.357488 | 0.497585 | 0.605705 | 4.000000 | 11 | 86 |
| D | 246 | 352 | 159 | 8 | 0.302642 | 0.338068 | 0.707386 | 0.670531 | 5.000000 | 18 | 97 |


## 1. Gravity 是否使航迹更集中？

固定 meter 的 A→B，水平 route density Gini 增加；同时有
46/100 条路径与 A 不同。
该指标包括整张水平图的零使用网格。需结合下面的 3D 复用、连续冲突和节点集中度，
不能仅凭 Gini 或视觉热图宣称论文网络已复现。

## 2. 复用、连续冲突、度集中与 Top10 覆盖变化

- route_density_gini: A=0.583793, B=0.584082, B−A=+0.000288
- route_cell_reuse_p90: A=3.000000, B=3.000000, B−A=+0.000000
- route_cell_reuse_max: A=9.000000, B=9.000000, B−A=+0.000000
- continuous_conflict_ratio: A=0.422764, B=0.481752, B−A=+0.058988
- max_degree: A=6.000000, B=6.000000, B−A=+0.000000
- degree_gini: A=0.446786, B=0.450000, B−A=+0.003214
- ci_top10_coverage: A=0.365854, B=0.357664, B−A=-0.008189

3D reuse 是每个访问过的 3D 网格的不同飞行计划数（同一航迹多次访问只计一次）；
p90 只在已访问 3D 网格上计算。Gini 则是整张水平图上的不同航迹 footprint 计数，
两者不是同一个统计量。连续冲突沿用原有邻接定义，比例为多事件连续段内事件数/全部不确定冲突事件数。
CI 沿用原公式、l=2、固定 Top10 和 flight-id tie break；点覆盖统计 incident event union。

## 3. A* 距离尺度的影响

在同一 gravity 人口层中，B→C→D 的水平步长分别是 100m、1 grid unit、0.1km；
垂直和斜向移动按原有物理几何同比缩放，g 的长度项与 h 同时换单位，风险不换尺度。
原有 w(n)、26 邻域、关闭节点及终止条件不变。各模式与 A 的路径差异数见表。
coverage 范围=0.019596；uncertain conflicts 范围=
137..352。
runtime 包含各组诊断图/文件输出，不含共享城市和交通采样；不能作为纯 A* 性能计时。

## 4. Top10 coverage 差距主要来自哪里？

本样本中，A* 距离尺度 的 coverage 变化较大。人口模型 |B−A|=0.008189，gravity 三种尺度的 coverage 范围=0.019596。这是单座合成城市、单个交通样本的条件比较，不是总体因果结论，也不排除未公开城市布局、计数约定和风险近似的影响。
四组 coverage 范围仅 0.338068..
0.365854，全部远低于论文 0.804124。
人口模型对照和距离尺度的 coverage 变化幅度，都远小于这一缺口；
因此不能确认低 Top10 coverage 的主要成因已定位，也不能把更多冲突/更高航迹复用误判为更接近论文结构。
风险数值仍使用项目既有下坠/漂移近似与全图 min-max 归一化；归一化方法也是未公开实现假设，
本轮没有同时扫描它。当前实验不能单独排除这一上游因素。

## 5. 公开参数与实现假设

目标论文明确：3500 人/km²、仿真建筑地图、综合对地风险结构、100 条约 6km 初始航迹、
10m/s、风险/距离权重0.8/0.2；公开网络值仅用于以上对照。
实现假设：4 个中心、10×10 窗口、1km NMS 间距、beta=2、建筑密度归一化与平局规则、
静态 snapshot、既有风险近似与 min-max、A* 的距离单位、CI l=2 和连续段诊断口径。
参考 [28]：[作者仓库](https://hdl.handle.net/10356/170158)，
[DOI](https://doi.org/10.1109/DASC58513.2023.10311224)。源站 PDF 返回405，
分段扩散式按用户提供的规格实现；不复制新加坡人口密度，不使用 MRT 数据或随机森林。
完整假设见每个 run 的 `implementation_assumptions.md`。

## 独立运行目录

- A: [完整输出](../../runs/20260912_204150_591_network_old_gaussian_meter_env2025_traffic2025_856707b5/)，run_id=`20260912_204150_591_network_old_gaussian_meter_env2025_traffic2025_856707b5`
- B: [完整输出](../../runs/20260912_204153_058_network_reference28_gravity_meter_env2025_traffic2025_856707b5/)，run_id=`20260912_204153_058_network_reference28_gravity_meter_env2025_traffic2025_856707b5`
- C: [完整输出](../../runs/20260912_204155_492_network_reference28_gravity_grid_env2025_traffic2025_856707b5/)，run_id=`20260912_204155_492_network_reference28_gravity_grid_env2025_traffic2025_856707b5`
- D: [完整输出](../../runs/20260912_204157_716_network_reference28_gravity_kilometer_env2025_traffic2025_856707b5/)，run_id=`20260912_204157_716_network_reference28_gravity_kilometer_env2025_traffic2025_856707b5`
