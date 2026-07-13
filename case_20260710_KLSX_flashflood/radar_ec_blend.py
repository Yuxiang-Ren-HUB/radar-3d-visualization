"""
雷达 + ECMWF(IFS Open Data) 融合短临预报演示 (0-6小时)
个例: 2026-07-10 密苏里东南部(Iron/Reynolds County)特大暴雨山洪
      NWS圣路易斯办公室评定为"catastrophic flash flood", 累计雨量6-11英寸,
      灾情高峰约在 05:44 CDT (~10:44 UTC)。

T0 = 2026-07-10 06:00 UTC (灾情高峰落在 T0+4h44m, 处于0-6h验证窗口内)
雷达: KLSX(圣路易斯) NEXRAD Level II, 见 download_klsx_case.py
EC: ECMWF IFS Open Data, 2026-07-10 00Z起报, F06/F09/F12 (对应 T0 的 lead 0/3/6h)

思路:
  - 雷达外推 (pysteps LK光流 + 半拉格朗日外推, 降雨率dB空间)
  - EC只是全球模式(0.25度, 非对流可分辨), 没有模拟雷达反射率诊断量, 所以公共物理量
    改用"降雨率"(mm/h): 雷达走 Z-R 关系(Marshall-Palmer Z=200 R^1.6, 与pysteps_nowcast_demo.py一致),
    EC走累积降水量(tp)在相邻输出时次间做差分, 换算成3小时平均降雨率
  - 融合: 按时效做线性加权, 时效越短越依赖雷达, 时效越长越依赖EC
  - 检验: CSI, 阈值取 1/10/25 mm/h (与pysteps_nowcast_demo.py一致)

注意: EC Open Data 只有3小时一档, 所以检验点只有0/3/6h三个, 比逐小时粒度粗很多。
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from pysteps import motion, nowcasts
from pysteps.utils import transformation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # case_data.py lives one level up (shared across cases)
from case_data import load_radar_rainrate_stack, load_ec_cumulative_precip

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

RADAR_DIR = r"F:\WRF_backup\radar_demo\KLSX_20260710_flashflood"
OUT_PREFIX = "radar_ec_blend"
EC_INIT = "2026-07-10 00:00"
EC_FXX = [6, 9, 12]          # valid at 06Z/09Z/12Z UTC = T0 + 0h/3h/6h
T0 = datetime(2026, 7, 10, 6, 0, 0)
LEAD_HOURS = [0, 3, 6]

GRID_CENTER_XY_KM = (0, 0)   # centered on KLSX itself; storm sits ~110-135km south, well within extent
HALF_EXTENT_KM = 180
RES_KM = 2.0
LEVEL_KM = 1.5               # low-level slice for Z-R rain-rate estimation

N_INPUT = 10                 # radar frames used to estimate motion field
CSI_THRESHOLDS = [1.0, 10.0, 25.0]  # mm/h

# ================================================================
# 1) build radar rain-rate time series (low-level Z-R) on the local grid
# ================================================================
print("=== gridding radar volumes (low-level reflectivity -> rain rate) ===")
radar_times, rainrate_stack, lat2d, lon2d = load_radar_rainrate_stack(
    RADAR_DIR, GRID_CENTER_XY_KM, HALF_EXTENT_KM, RES_KM, LEVEL_KM)
print("radar rain-rate stack shape:", rainrate_stack.shape)

# ================================================================
# 2) download EC precip + regrid onto the radar's local lat/lon grid, convert to period rain rate
# ================================================================
print("\n=== downloading + regridding ECMWF IFS tp ===")
tp_on_radar_grid = load_ec_cumulative_precip(EC_INIT, EC_FXX, lat2d, lon2d)  # (3,ny,nx), cumulative mm since 00Z

ec_rate_0_3 = tp_on_radar_grid[1] / 3.0                                # mean mm/h over lead [0,3h]
ec_rate_3_6 = (tp_on_radar_grid[2] - tp_on_radar_grid[1]) / 3.0        # mean mm/h over lead [3,6h]
ec_ckpt = [np.zeros_like(ec_rate_0_3), ec_rate_0_3, ec_rate_3_6]       # no EC "rate" defined at lead 0h itself

# ================================================================
# 4) radar-only extrapolation nowcast (LK motion + semilagrangian, in dB(rain-rate) space)
# ================================================================
print("\n=== radar extrapolation nowcast ===")
t0_idx = int(np.argmin([abs((t - T0).total_seconds()) for t in radar_times]))
input_idx = list(range(max(0, t0_idx - N_INPUT + 1), t0_idx + 1))
dt_min = np.mean([(radar_times[i + 1] - radar_times[i]).total_seconds() / 60.0
                   for i in input_idx[:-1]])
print(f"T0 matched to radar frame {radar_times[t0_idx]}, dt~{dt_min:.1f}min, "
      f"{len(input_idx)} input frames")

R_input = rainrate_stack[input_idx]
R_db, metadata = transformation.dB_transform(R_input, threshold=0.1, zerovalue=-15.0)
R_db = np.nan_to_num(R_db, nan=-15.0, posinf=-15.0, neginf=-15.0)

oflow = motion.get_method("LK")
motion_field = oflow(R_db)
extrapolate = nowcasts.get_method("extrapolation")
n_leadtimes_native = int(round(6 * 60 / dt_min)) + 2
R_forecast_db = extrapolate(R_db[-1], motion_field, n_leadtimes_native)
R_forecast_native = transformation.dB_transform(R_forecast_db, inverse=True, threshold=-10, zerovalue=-15.0)[0]
R_forecast_native = np.nan_to_num(R_forecast_native, nan=0.0)
print("extrapolation forecast shape:", R_forecast_native.shape)

# ================================================================
# 5) sample radar-extrapolation + observations at 0/3/6h checkpoints (nearest actual radar frame)
# ================================================================
radar_extrap_ckpt = []
obs_ckpt = []
for h in LEAD_HOURS:
    target_time = T0 + timedelta(hours=h)
    idx = int(np.argmin([abs((t - target_time).total_seconds()) for t in radar_times]))
    obs_ckpt.append(rainrate_stack[idx])
    if h == 0:
        radar_extrap_ckpt.append(rainrate_stack[t0_idx])
    else:
        step = int(round(h * 60 / dt_min))
        step = min(step, R_forecast_native.shape[0] - 1)
        radar_extrap_ckpt.append(R_forecast_native[step])
radar_extrap_ckpt = np.array(radar_extrap_ckpt)
obs_ckpt = np.array(obs_ckpt)
ec_ckpt = np.array(ec_ckpt)

# ================================================================
# 6) blend: weight decays linearly from radar-heavy (t=0) to EC-heavy (t=6h)
# ================================================================
lead_hours_arr = np.array(LEAD_HOURS, dtype=float)
w_radar = np.clip(1 - lead_hours_arr / 6.0, 0, 1)
blended_ckpt = np.array([
    w_radar[i] * radar_extrap_ckpt[i] + (1 - w_radar[i]) * ec_ckpt[i]
    for i in range(len(LEAD_HOURS))
])

# ================================================================
# 7) verification: CSI vs lead time for radar-only / EC-only / blended
# ================================================================
print("\n=== verification ===")


def csi(forecast, obs, th):
    fc_bin = forecast > th
    obs_bin = obs > th
    hits = np.sum(fc_bin & obs_bin)
    misses = np.sum(~fc_bin & obs_bin)
    false_alarms = np.sum(fc_bin & ~obs_bin)
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else np.nan


csi_radar = {th: [] for th in CSI_THRESHOLDS}
csi_ec = {th: [] for th in CSI_THRESHOLDS}
csi_blend = {th: [] for th in CSI_THRESHOLDS}
for i, h in enumerate(LEAD_HOURS):
    for th in CSI_THRESHOLDS:
        csi_radar[th].append(csi(radar_extrap_ckpt[i], obs_ckpt[i], th))
        csi_ec[th].append(csi(ec_ckpt[i], obs_ckpt[i], th))
        csi_blend[th].append(csi(blended_ckpt[i], obs_ckpt[i], th))

fig, axes = plt.subplots(1, len(CSI_THRESHOLDS), figsize=(6 * len(CSI_THRESHOLDS), 5.5))
for ax, th in zip(axes, CSI_THRESHOLDS):
    ax.plot(LEAD_HOURS, csi_radar[th], '-o', label='纯雷达外推', color='#1e9bff')
    ax.plot(LEAD_HOURS, csi_ec[th], '-s', label='纯EC(IFS)预报', color='#ff8c3d')
    ax.plot(LEAD_HOURS, csi_blend[th], '-^', label='融合(雷达+EC)', color='#2ecc71', linewidth=2.5)
    ax.set_xlabel("预报时效 (小时)")
    ax.set_ylabel("CSI")
    ax.set_title(f"{th:g} mm/h 阈值")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("雷达外推 vs ECMWF(IFS)预报 vs 融合 —— CSI随预报时效变化\n(密苏里东南部特大暴雨个例, 2026-07-10)")
plt.savefig(f"{OUT_PREFIX}_CSI_comparison.png", dpi=180, bbox_inches='tight')
plt.close(fig)

print("\n--- CSI(10mm/h) 汇总 ---")
for i, h in enumerate(LEAD_HOURS):
    print(f"+{h}h: 雷达={csi_radar[10.0][i]:.3f}  EC={csi_ec[10.0][i]:.3f}  融合={csi_blend[10.0][i]:.3f}")

# ================================================================
# 8) snapshot comparison at the 3 checkpoints
# ================================================================
levels = [0.1, 1, 2, 5, 10, 20, 40, 80, 150]
fig2, axes2 = plt.subplots(4, len(LEAD_HOURS), figsize=(4.5 * len(LEAD_HOURS), 15))
for col, h in enumerate(LEAD_HOURS):
    panels = [
        (obs_ckpt[col], "实况观测"),
        (radar_extrap_ckpt[col], "纯雷达外推"),
        (ec_ckpt[col], "纯EC预报"),
        (blended_ckpt[col], "融合"),
    ]
    for row, (field, label) in enumerate(panels):
        ax = axes2[row, col]
        cf = ax.contourf(field, levels=levels, cmap='NWSRef', extend='max')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    axes2[0, col].set_title(f"实况观测 +{h}h")
for row, label in enumerate(["实况观测", "纯雷达外推", "纯EC预报", "融合"]):
    axes2[row, 0].set_ylabel(label, fontsize=13)
fig2.colorbar(cf, ax=axes2, label="降雨率 (mm/h)", shrink=0.6, location='right')
fig2.suptitle("雷达外推 / EC(IFS)预报 / 融合 三者对比\n(密苏里东南部特大暴雨个例, 2026-07-10)")
plt.savefig(f"{OUT_PREFIX}_snapshots.png", dpi=160, bbox_inches='tight')
plt.close(fig2)

print("\ndone:", f"{OUT_PREFIX}_CSI_comparison.png, {OUT_PREFIX}_snapshots.png")
