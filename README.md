# 雷达三维可视化 / Radar 3D Visualization

基于 NEXRAD Level II 原始体扫数据的龙卷/超级单体分析与三维可视化。个例包括：

- **Moore, OK（2013-05-20）** EF-5级龙卷
- **Rolling Fork, MS（2023-03-25）** EF-4级龙卷

数据来自雷达站 KTLX / KDGX 的原始体扫（Level II Archive），用 [Py-ART](https://arm-doe.github.io/pyart/) 读取、网格化，
[Plotly](https://plotly.com/python/) 做交互式三维渲染，[pysteps](https://pysteps.readthedocs.io/) 做外推 nowcasting 演示。

## 目录结构

```
radar_3d_structure.py            CAPPI等高面、垂直剖面、经典等值面三维图（Moore + Rolling Fork）
radar_advanced_products.py       VIL、方位角切变(中气旋探测)、双偏振(ZDR/RHOHV，龙卷碎片特征TDS)
pysteps_nowcast_demo.py          雷达外推短临预报演示：CSI随预报时效衰减曲线 + 快照对比

make_interactive_3d.py           交互式3D：经典等值面版(35dBZ阈值，高度着色)
particle_3d.py                   交互式3D：粒子云版(密度/透明度随回波强度变化)
volume_3d.py                     交互式3D：体绘制版(连续渲染，NWS雷达回波色标)
animate_volume_3d.py             交互式3D：体绘制随时间演变动画版(7帧，播放/滑块控制)
flow_field_3d.py                 交互式3D：径向速度"流场"(基于真实退模糊速度数据，明确单雷达局限)
particle_flow_3d.py              交互式3D：艺术化粒子流动画(手工构造矢量场，纯视觉效果)

*_page.html                      对应可视化的页面模板(占位符待填入Plotly片段)
*_final.html                     填入数据后的完整自包含页面(可直接用浏览器打开)

check_ranges.py / check_ranges2.py   早期调试脚本，核对网格范围设置

moore_tornado_*.png                  Moore个例：CAPPI/剖面/等值面静态图
rollingfork_tornado_*.png            Rolling Fork个例：CAPPI/剖面/等值面静态图
rollingfork_VIL.png                  Rolling Fork：垂直积分液态水含量
rollingfork_azshear.png              Rolling Fork：方位角切变(中气旋探测)
rollingfork_dualpol.png              Rolling Fork：双偏振产品(龙卷碎片特征)
pysteps_demo_*.png                   pysteps外推演示结果
volume_3d_preview.png                体绘制版静态预览
flow_field_preview.png               流场版静态预览
particle_flow_preview*.png           粒子流版静态预览(初始帧+中间帧)
```

`.gitignore` 排除了 `plotly_fragment*.html`（构建过程的中间产物，内容已完整包含在对应的 `*_final.html` 里）
和 `_anim_volume_cache.npz`（体绘制网格化结果的本地缓存，可由脚本重新生成，不必版本化）。

## 三维可视化版本对比

Rolling Fork 个例做了四版交互式三维可视化，从静态到动态、从真实数据到纯视觉效果：

| 版本 | 脚本 | 特点 |
|---|---|---|
| 等值面 | `make_interactive_3d.py` | 单一35dBZ阈值的等值面网格，高度着色 |
| 粒子云 | `particle_3d.py` | 12万个粒子，强度决定保留概率(密度)和透明度 |
| 体绘制 | `volume_3d.py` | 连续体渲染(非离散抽样)，结构更连贯，经典NWS雷达回波色标 |
| 时间动画 | `animate_volume_3d.py` | 7个体扫拼成动画，覆盖龙卷整个生命史(00:54-02:12 UTC) |
| 径向速度流场 | `flow_field_3d.py` | 用退模糊后的真实径向速度构造3D箭头，蓝=朝向雷达/橙=远离雷达 |
| 艺术化粒子流 | `particle_flow_3d.py` | 手工构造的辐合+旋转+上升气流矢量场，纯视觉效果 |

**等值面 → 粒子云的迭代原因**：粒子云用随机抽样表现强度，容易把风暴连贯的结构（钩状回波、对流塔）打散，
看不出形状，因此进一步改成体绘制。

**径向速度流场的关键限制**：单部雷达（单多普勒）只能测到目标沿波束方向的速度分量，测不到切向/垂直分量，
无法重建真实完整的三维风场；图中蓝橙分离是风暴旋转的经典速度对特征，但不等同于反演出真实风场。
开发过程中还发现并修复了一个 Plotly 的坑——`go.Cone` 默认按矢量**模长**（恒为正）上色，而不是按
构造方向时用的带符号速度值，导致第一版无论朝向还是远离雷达的箭头全部同色；修复方式是按符号拆分成
两个独立的 Cone 图层。

**艺术化粒子流**：矢量场完全是手工构造的（参考超级单体教科书概念模型：低层入流辐合旋转、核心区上升），
不是从雷达数据反演得到，页面上有醒目标注避免误解为真实风场重建。

## 高级产品方法说明

- **VIL**：Greene & Clark (1972) 公式 `M=3.44e-6·Z^(4/7)`，沿垂直方向积分。
- **方位角切变**：在原生极坐标(方位角-距离)下计算 `d(Vr)/d(azimuth)/range`，而非网格化后的直角坐标，
  避免网格化插值抹平中气旋尺度的切变信号。
- **双偏振龙卷碎片特征(TDS)**：低 RHOHV(<0.8) 与速度对(couplet)共位。
- **NEXRAD split-cut扫描策略(VCP212)**：同一仰角有两次扫描——监视扫描(含双偏振，无速度)和多普勒扫描
  (含速度，无双偏振)，因此速度/切变类产品和双偏振类产品需要分别取自不同的sweep。

## pysteps 外推 nowcasting 演示

用 Rolling Fork 个例约7小时(58个体扫)的连续雷达资料，前10帧(~1小时)估计运动场(Lucas-Kanade光流)并做
拉格朗日外推，验证到约4.8小时预报时效，得到 CSI 随时效衰减、且衰减速率随降雨强度阈值提高而加快的
典型结果，并观察到风暴组织形态发生变化(转为飑线状)导致固定运动场外推失效的现象。

## 依赖

```
pip install arm_pyart plotly pysteps netCDF4 matplotlib numpy scikit-image kaleido
```
