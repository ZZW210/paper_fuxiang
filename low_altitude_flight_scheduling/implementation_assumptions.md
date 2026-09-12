# 人口、风险与初始网络实验的实现假设

## 原文依据与核验边界

[目标论文《基于复杂网络的城市低空飞行计划优化调度》](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2025.31479)
第 1.2.3 节给出 3500 人/km² 人口密度基准、城市建筑地图仿真数据和参考 [28] 生成人口环境；
第 1.2.2 节给出 `P_total=P_bal+P_par`、`P_r=P_failure*P_impact*P_severity`、`P_impact=A*rho`。
本地原文已读取核实这些内容。

参考 [28]：Pang et al., Population Density Estimation for Dynamic Ground Risk Assessment of Drone Operations,
[DOI](https://doi.org/10.1109/DASC58513.2023.10311224)，[作者仓库](https://hdl.handle.net/10356/170158)。
在线 PDF 请求返回 HTTP 405，随后已读取项目根目录的 `Manuscript_DASC_2023.pdf` 核验：
第 2 页 III.B 节公式 (1)–(2) 使用最近站点、1km 内 `sigma_center*exp(1-r_km^2)`、
1km 外静态人口。原文仅说明小于/大于 1km；恰好 1km 采用静态分支是本轮用户指定的边界约定。
只借鉴并核验人口扩散方法，不复制作者完整数据或训练实现；没有下载 MRT 数据、建立真实 MRT 站或训练随机森林。

## 人口层

- 默认 `reference28_gravity`、`static_snapshot`；所有地面网格先初始化 3500 人/km²。
- 障碍数组沿高度 `any(axis=2)` 得到 footprint，高度不会重复计权。
- 非重叠 10×10 窗口、窗口几何中心作为活动中心、8 邻域局部极大（含平局 plateau）、
  按密度降序再 x/y 排序、1000m 欧氏 NMS、默认选择至多 4 个中心，均是实现假设。
  当地图不能提供足够极大值时记录实际数目，不用冲突覆盖率补点、移点或调数量。
- `normalized_building_density=window_density/max_window_density`；空建筑图取 0、无中心且保持静态基准。
- `center_strength=3500*(1+beta*normalized_building_density)`，默认 beta=2，是实现假设。
  显式未来敏感性值 1/2/4，不自动选最接近论文的 beta。
- 地面网格取最近中心：`r_km<1` 时 `strength*exp(1-r_km²)`，否则 3500；
  恰好 1km 使用外侧基准。中心贡献不求和，不平滑边界，建筑网格不置零。
- 全部人口诊断数组/CSV 的单位为人/km²。风险内部在面积乘法前只除以一次一百万。
- Reference [28] is time-dependent, but the target paper does not disclose a time-of-day population scenario;
  therefore a static spatial snapshot using its gravity diffusion concept is used.

## 风险层

本轮保持既有弹道/降落伞严重程度与漂移近似，保留目标论文综合风险的乘积/求和结构；
没有把参考 [28] 的独立 ground-risk 表达式替换为最终风险。既有能量与漂移算法仍是近似，
此次只定位人口及 A* 尺度，不声称每一条下坠方程都已精确复原。

`risk_map_raw.npy` 保存上述未归一化概率；`risk_map.npy` 保留全图 min-max 归一化供 A* 使用。
归一化也是未公开的实现假设，四组共用它。严格模式不再用障碍占用覆盖风险数值；
不可通行性仍由 grid 判断。旧工程模式保留其原人口生成器和障碍风险标记。
实验 A 只恢复旧 Gaussian 人口生成器（含 footprint boost），与 B-D 共用相同风险计算，
不是直接复用旧历史风险结果。

`risk_map_stats.csv` 的 min/mean/median/max/std/p90/p95/p99 是归一化图统计。
现有四层中心高度 15/45/75/105m 分别命名 z1/z2/z3/z4；ground 图是最低层的对地投影，
不是 0m 处的下坠模拟，也不是第五层可通行空域。

## 路径与网络实验

环境种子和交通种子都默认 2025。前者控制城市、人口与风险，后者控制 OD/ETD。
四组先共享一份冻结的 OD/ETD 表，再规划路径；失败直接报告，不重抽任务。
OD 在自由地面网格、6000±1200m 欧氏距离带内条件采样，离散化和容差是已有假设。
每条 ETD 在 [0,1800] 均匀采样，速度固定 10m/s。

A* 保持现有 26 邻域、加权启发式 w、关闭节点和终止行为，仅通过单位除数同时缩放步长与 h。
meter/grid/kilometer 的水平步长为 100/1/0.1；风险/距离权重保持 0.8/0.2。
默认 meter 不变，不根据结果自动选择任何尺度。

route density Gini 统计全水平网格（含零值）上的不同航迹 footprint 数；
3D cell reuse 统计访问过的 3D 网格上的不同航迹数（每航迹每网格最多计一次），p90 仅在被访问网格上算。
连续冲突沿用既有邻接规则，仅多事件连续段计入比率。
不改变 CI 公式、l=2、固定 Top10、flight-id tie break、冲突定义、sigma 或任何调度算法。

53/97/78÷97 仅用于报告 comparison，不进入规划与参数选择。
四组的独立 run_id、双种子、共享输入 hash、代码 hash、git_dirty、实际阶段均写入 manifest；
旧目录不覆盖。单座合成城市的结果不足以作总体因果归因。
