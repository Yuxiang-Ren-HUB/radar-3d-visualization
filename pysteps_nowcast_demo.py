"""
pysteps 雷达外推nowcasting演示 —— Rolling Fork超级单体(2023-03-25)
14个体扫(约96分钟, ~6分钟间隔): 前10个做"观测输入"估计运动场并外推，
留出最后4个做"held-out"验证，看外推预报跟实际观测差多少。

pip install pysteps arm_pyart netCDF4 matplotlib numpy
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pyart
from pysteps import motion, nowcasts
from pysteps.utils import transformation

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

RADAR_DIR = r"F:\WRF_backup\radar_demo\KDGX_20230325_extended4h"
OUT_PREFIX = "pysteps_demo_6h"

# fixed Cartesian grid covering the storm's whole track over ~7h (NOT re-centered per frame --
# we need the real translation preserved for motion estimation to work).
# Storm system moved from (-113,140) to (112,90) over this window -> wide east-west domain needed.
GRID_CENTER_XY_KM = (0, 110)
HALF_EXTENT_KM = 180
RES_KM = 2.0
LEVEL_KM = 1.5  # low-level CAPPI

N_INPUT = 10    # frames used to estimate motion + as nowcast starting point (~1h)
N_LEADTIMES = 48  # frames to predict forward: capped by data (58 total - 10 input = 48 -> ~4.8h verified)

files = sorted(glob.glob(os.path.join(RADAR_DIR, "*V06*")))
files = [f for f in files if not f.endswith("_MDM")]
print(f"found {len(files)} volume scans")

half_m = HALF_EXTENT_KM * 1000
res_m = RES_KM * 1000
nxy = int(2 * half_m / res_m) + 1
cx_m, cy_m = GRID_CENTER_XY_KM[0] * 1000, GRID_CENTER_XY_KM[1] * 1000

refl_stack = []
times = []
for f in files:
    radar = pyart.io.read_nexrad_archive(f)
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(3, nxy, nxy),  # just need a thin slab around LEVEL_KM
        grid_limits=(((LEVEL_KM - 0.5) * 1000, (LEVEL_KM + 0.5) * 1000),
                     (cy_m - half_m, cy_m + half_m), (cx_m - half_m, cx_m + half_m)),
        fields=['reflectivity'],
        weighting_function='Barnes2',
    )
    refl = np.ma.filled(grid.fields['reflectivity']['data'][1], -30)  # middle level
    refl_stack.append(refl)
    times.append(os.path.basename(f))
    print(f"gridded {os.path.basename(f)}")

refl_stack = np.array(refl_stack)  # (ntimes, ny, nx)
x_km = grid.x['data'] / 1000
y_km = grid.y['data'] / 1000

# ---------------- dBZ -> rain rate (Marshall-Palmer Z=200 R^1.6) -> pysteps dB-transform ----------------
Z_linear = 10 ** (refl_stack / 10.0)
rainrate = (Z_linear / 200.0) ** (1.0 / 1.6)  # mm/h
rainrate = np.clip(rainrate, 0, 300)

R_input = rainrate[:N_INPUT]
R_obs_future = rainrate[N_INPUT:N_INPUT + N_LEADTIMES]

R_db, metadata = transformation.dB_transform(R_input, threshold=0.1, zerovalue=-15.0)
R_db = np.nan_to_num(R_db, nan=-15.0, posinf=-15.0, neginf=-15.0)

# ---------------- motion estimation (Lucas-Kanade optical flow) ----------------
oflow = motion.get_method("LK")
motion_field = oflow(R_db)
print("motion field shape:", motion_field.shape, " mean speed (grid units/step):",
      np.nanmean(np.sqrt(motion_field[0] ** 2 + motion_field[1] ** 2)))

# ---------------- deterministic Lagrangian extrapolation ----------------
extrapolate = nowcasts.get_method("extrapolation")
R_forecast_db = extrapolate(R_db[-1], motion_field, N_LEADTIMES)
R_forecast = transformation.dB_transform(R_forecast_db, inverse=True, threshold=-10, zerovalue=-15.0)[0]
R_forecast = np.nan_to_num(R_forecast, nan=0.0)

# ================================================================
# 1) CSI-vs-leadtime curve (the main deliverable: skill decay out to ~4.8h)
# ================================================================
thresholds = [1.0, 10.0, 25.0]
csi_curves = {th: [] for th in thresholds}
leadtime_min = [(lt + 1) * 6 for lt in range(N_LEADTIMES)]

for lt in range(N_LEADTIMES):
    for th in thresholds:
        fc_bin = R_forecast[lt] > th
        obs_bin = R_obs_future[lt] > th
        hits = np.sum(fc_bin & obs_bin)
        misses = np.sum(~fc_bin & obs_bin)
        false_alarms = np.sum(fc_bin & ~obs_bin)
        denom = hits + misses + false_alarms
        csi = hits / denom if denom > 0 else np.nan
        csi_curves[th].append(csi)

fig0, ax0 = plt.subplots(figsize=(9, 6))
for th in thresholds:
    ax0.plot(np.array(leadtime_min) / 60, csi_curves[th], '-o', markersize=3, label=f'{th:g} mm/h阈值')
ax0.set_xlabel("预报时效 (小时)")
ax0.set_ylabel("CSI (临界成功指数)")
ax0.set_title("pysteps拉格朗日外推 —— 预报技巧随时效衰减曲线\n(Rolling Fork个例, 2023-03-25)")
ax0.legend()
ax0.grid(alpha=0.3)
ax0.set_ylim(0, 1)
plt.savefig(f"{OUT_PREFIX}_CSI_curve.png", dpi=180, bbox_inches='tight')
plt.close(fig0)

print("\n--- CSI (10mm/h阈值) 摘要 ---")
for lt in [0, 4, 9, 19, 29, 39, N_LEADTIMES - 1]:
    if lt < N_LEADTIMES:
        print(f"+{leadtime_min[lt]}min ({leadtime_min[lt] / 60:.1f}h): CSI={csi_curves[10.0][lt]:.3f}")

# ================================================================
# 2) snapshot comparisons at representative lead times
# ================================================================
snapshot_lts = [idx for idx in [4, 9, 19, 29, min(39, N_LEADTIMES - 1), N_LEADTIMES - 1] if idx < N_LEADTIMES]
snapshot_lts = sorted(set(snapshot_lts))
fig, axes = plt.subplots(2, len(snapshot_lts), figsize=(4.2 * len(snapshot_lts), 8.5))
levels = [0.1, 1, 2, 5, 10, 20, 40, 80, 150]
for col, lt in enumerate(snapshot_lts):
    ax_fc = axes[0, col]
    cf1 = ax_fc.contourf(x_km, y_km, R_forecast[lt], levels=levels, cmap='NWSRef', extend='max')
    ax_fc.set_title(f"外推预报 +{leadtime_min[lt] / 60:.1f}h")
    ax_fc.set_aspect('equal')

    ax_obs = axes[1, col]
    cf2 = ax_obs.contourf(x_km, y_km, R_obs_future[lt], levels=levels, cmap='NWSRef', extend='max')
    ax_obs.set_title(f"实际观测 +{leadtime_min[lt] / 60:.1f}h")
    ax_obs.set_aspect('equal')

fig.colorbar(cf1, ax=axes, label="降雨率 (mm/h)", shrink=0.6, location='right')
fig.suptitle("pysteps 拉格朗日外推 nowcasting vs 实际观测（Rolling Fork个例）")
plt.savefig(f"{OUT_PREFIX}_snapshots.png", dpi=180, bbox_inches='tight')
plt.close(fig)

print("\ndone:", f"{OUT_PREFIX}_CSI_curve.png, {OUT_PREFIX}_snapshots.png")
