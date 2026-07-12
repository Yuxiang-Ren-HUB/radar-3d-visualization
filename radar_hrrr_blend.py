"""
雷达 + HRRR(NWP) 融合短临预报演示 (0-6小时)
个例: Rolling Fork, MS 超级单体/龙卷 (2023-03-25)

思路:
  - 雷达外推 (pysteps LK光流 + 半拉格朗日外推): 短时效精度高, 但技巧随时效衰减快
  - HRRR预报 (2023-03-25 00Z起报, F00-F06 逐小时合成反射率REFC): 时效更长但起始时刻不如雷达精细
  - 融合: 按时效做线性加权, 时效越短越依赖雷达, 时效越长越依赖HRRR
  - 检验: 用后续雷达实况观测, 对比 纯雷达外推 / 纯HRRR / 融合 三者的CSI随时效衰减曲线

两者都换算成"合成反射率dBZ"这一个物理量直接融合, 不需要像纯雷达nowcast演示那样转换成降雨率。
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pyart
from pysteps import motion, nowcasts
from pysteps.utils import transformation
from scipy.interpolate import griddata
from herbie import Herbie

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

RADAR_DIR = r"F:\WRF_backup\radar_demo\KDGX_20230325_extended4h"
OUT_PREFIX = "radar_hrrr_blend"
HRRR_INIT = "2023-03-25 00:00"
HRRR_FXX = list(range(0, 7))  # F00..F06, hourly

GRID_CENTER_XY_KM = (0, 110)   # same domain as the earlier pysteps demo (storm-track-covering)
HALF_EXTENT_KM = 180
RES_KM = 2.0
Z_LEVELS_KM = [1.5, 3, 5, 7, 9, 12]  # coarse column-max sampling (compute budget vs. depth coverage)

N_INPUT = 10          # radar frames (~1h) used to estimate motion field
RADAR_DBZ_FLOOR = -20  # treat missing/clear-air as this floor value

CSI_THRESHOLDS = [20.0, 30.0, 40.0]

CACHE_NPZ = r"F:\研\科研\Python\radar_analysis\_radar_hrrr_blend_cache.npz"

if os.path.exists(CACHE_NPZ):
    print("=== loading cached radar+HRRR grids from", CACHE_NPZ, "===")
    _c = np.load(CACHE_NPZ, allow_pickle=True)
    refl_colmax_stack = _c["refl_colmax_stack"]
    radar_times = list(_c["radar_times"])
    lat2d = _c["lat2d"]
    lon2d = _c["lon2d"]
    hrrr_on_radar_grid = _c["hrrr_on_radar_grid"]
    hrrr_valid_hours = list(_c["hrrr_valid_hours"])
else:
    # ================================================================
    # 1) download + regrid HRRR REFC (composite reflectivity) forecast, F00-F06
    # ================================================================
    print("=== downloading HRRR REFC ===")
    hrrr_frames = []
    hrrr_valid_hours = []
    for fxx in HRRR_FXX:
        H = Herbie(HRRR_INIT, model='hrrr', product='sfc', fxx=fxx)
        ds = H.xarray(":REFC:")
        if isinstance(ds, list):
            ds = ds[0]
        refc = ds['refc'].values if 'refc' in ds else list(ds.data_vars.values())[0].values
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        hrrr_frames.append((lat, lon, refc))
        hrrr_valid_hours.append(fxx)
        print(f"F{fxx:02d} downloaded, shape={refc.shape}")

    # ================================================================
    # 2) build radar composite (column-max) reflectivity time series on the local grid
    # ================================================================
    files = sorted(glob.glob(os.path.join(RADAR_DIR, "*V06")))
    files = [f for f in files if "MDM" not in f]
    print(f"\n=== gridding {len(files)} radar volumes (column-max reflectivity) ===")

    half_m = HALF_EXTENT_KM * 1000
    res_m = RES_KM * 1000
    nxy = int(2 * half_m / res_m) + 1
    cx_m, cy_m = GRID_CENTER_XY_KM[0] * 1000, GRID_CENTER_XY_KM[1] * 1000
    z_levels_m = tuple((z * 1000, z * 1000) for z in Z_LEVELS_KM)

    refl_colmax_stack = []
    radar_times = []
    lat2d = lon2d = None
    for f in files:
        radar = pyart.io.read_nexrad_archive(f)
        col_max = None
        for z_lo, z_hi in z_levels_m:
            grid = pyart.map.grid_from_radars(
                (radar,), grid_shape=(1, nxy, nxy),
                grid_limits=((z_lo, z_hi), (cy_m - half_m, cy_m + half_m), (cx_m - half_m, cx_m + half_m)),
                fields=['reflectivity'], weighting_function='Barnes2',
            )
            lvl = np.ma.filled(grid.fields['reflectivity']['data'][0], RADAR_DBZ_FLOOR)
            col_max = lvl if col_max is None else np.maximum(col_max, lvl)
            if lat2d is None:
                lat2d = grid.point_latitude['data'][0]
                lon2d = grid.point_longitude['data'][0]
        refl_colmax_stack.append(col_max)
        radar_times.append(os.path.basename(f))
        print(f"gridded {os.path.basename(f)}")

    refl_colmax_stack = np.array(refl_colmax_stack)  # (ntimes, ny, nx)
    print("radar column-max stack shape:", refl_colmax_stack.shape)

    # ---------------- regrid HRRR REFC onto the radar's local lat/lon grid ----------------
    print("\n=== regridding HRRR onto radar grid ===")
    lat_min, lat_max = lat2d.min() - 0.5, lat2d.max() + 0.5
    lon_min, lon_max = lon2d.min() - 0.5, lon2d.max() + 0.5

    hrrr_on_radar_grid = []
    for (lat, lon, refc), fxx in zip(hrrr_frames, hrrr_valid_hours):
        lon_adj = np.where(lon > 180, lon - 360, lon)
        mask = (lat >= lat_min) & (lat <= lat_max) & (lon_adj >= lon_min) & (lon_adj <= lon_max)
        pts = np.column_stack([lon_adj[mask], lat[mask]])
        vals = refc[mask]
        interp = griddata(pts, vals, (lon2d, lat2d), method='linear')
        nn = griddata(pts, vals, (lon2d, lat2d), method='nearest')
        interp = np.where(np.isnan(interp), nn, interp)
        hrrr_on_radar_grid.append(interp)
        print(f"F{fxx:02d}: regridded, subset points={mask.sum()}")

    hrrr_on_radar_grid = np.array(hrrr_on_radar_grid)  # (7, ny, nx), hourly F00..F06

    np.savez_compressed(
        CACHE_NPZ, refl_colmax_stack=refl_colmax_stack.astype(np.float32),
        radar_times=np.array(radar_times), lat2d=lat2d, lon2d=lon2d,
        hrrr_on_radar_grid=hrrr_on_radar_grid.astype(np.float32),
        hrrr_valid_hours=np.array(hrrr_valid_hours),
    )
    print("cached ->", CACHE_NPZ)

# parse actual scan timestamps -- there's a ~63min data gap around 04:00-05:03 UTC in this dataset,
# so a single dataset-wide mean interval would be skewed; use real timestamps for verification matching,
# and use only the INPUT window's interval for the extrapolation step size (that's what pysteps actually steps by)
import re as _re
from datetime import datetime as _dt, timedelta as _td
_times = []
for fn in radar_times:
    m = _re.search(r"(\d{8})_(\d{6})_V\d\d", fn)
    _times.append(_dt.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
_dt_minutes_all = [(_times[i + 1] - _times[i]).total_seconds() / 60 for i in range(len(_times) - 1)]
print(f"scan interval range: min={min(_dt_minutes_all):.1f} max={max(_dt_minutes_all):.1f} "
      f"(largest gaps flag missing volumes)")
radar_dt_min_input = float(np.mean(_dt_minutes_all[:N_INPUT - 1]))  # interval actually used by the input window
print(f"input-window scan interval: {radar_dt_min_input:.2f} min "
      f"(span {_times[0]} .. {_times[N_INPUT - 1]})")

# ================================================================
# 4) radar-only extrapolation nowcast (LK motion + semilagrangian, directly in dBZ space)
# ================================================================
print("\n=== radar extrapolation nowcast ===")
n_leadtimes_6min = int(6 * 60 / radar_dt_min_input)  # ~6h worth of native-interval steps

R_input = refl_colmax_stack[:N_INPUT]
oflow = motion.get_method("LK")
motion_field = oflow(R_input)
extrapolate = nowcasts.get_method("extrapolation")
R_forecast_dbz = extrapolate(R_input[-1], motion_field, n_leadtimes_6min)
print("extrapolation forecast shape:", R_forecast_dbz.shape)

# ================================================================
# 5) sample radar-extrapolation + observations at hourly checkpoints (0..6h) to match HRRR cadence
# ================================================================
# extrapolation steps forward at a fixed increment of radar_dt_min_input per step (that's what the
# motion field / semi-Lagrangian advection implicitly assumes based on the input window spacing)
steps_per_hour = int(round(60 / radar_dt_min_input))
hourly_idx_in_forecast = [min(h * steps_per_hour, R_forecast_dbz.shape[0] - 1) for h in range(0, 7)]
radar_extrap_hourly = R_forecast_dbz[hourly_idx_in_forecast]  # (7, ny, nx), lead 0..6h

# match each target lead time to the ACTUAL nearest-in-real-time observation (handles the irregular
# gap correctly, unlike counting a fixed number of index steps with an assumed-uniform interval)
start_time = _times[N_INPUT - 1]  # t=0 for the forecast corresponds to this radar frame
obs_hourly = []
obs_time_offsets_min = []
for h in range(0, 7):
    target_time = start_time + _td(hours=h)
    diffs = [abs((t - target_time).total_seconds()) for t in _times]
    nearest_idx = int(np.argmin(diffs))
    obs_hourly.append(refl_colmax_stack[nearest_idx])
    offset_min = (_times[nearest_idx] - target_time).total_seconds() / 60
    obs_time_offsets_min.append(offset_min)
    print(f"+{h}h target={target_time}  nearest obs={_times[nearest_idx]} (offset {offset_min:+.1f} min)")
obs_hourly = np.array(obs_hourly)

# ================================================================
# 6) blend: weight decays linearly from radar-heavy (t=0) to NWP-heavy (t=6h)
# ================================================================
# semi-Lagrangian extrapolation over a FIXED domain leaves NaN where the storm has advected clean
# out of frame (this storm moves fast enough that ~100% of the domain is vacated by hour 5 -- verified
# directly). NaN must NOT propagate into the blend: fall back fully to HRRR wherever radar has no
# valid extrapolated coverage, rather than letting arithmetic turn "no radar data" into a NaN blend cell.
radar_valid_mask_hourly = ~np.isnan(radar_extrap_hourly)
print("\nradar valid-coverage fraction by lead hour:",
      [f"{m.mean():.2f}" for m in radar_valid_mask_hourly])
radar_extrap_filled = np.where(radar_valid_mask_hourly, radar_extrap_hourly, RADAR_DBZ_FLOOR)

# feather the radar/HRRR handoff instead of a hard binary cutover: taper radar's contribution to zero
# over a ~20km band approaching the coverage boundary, rather than an abrupt edge right at the mask
from scipy.ndimage import distance_transform_edt
FEATHER_KM = 20.0
feather_px = FEATHER_KM / RES_KM

lead_hours = np.arange(0, 7)
w_radar_base = np.clip(1 - lead_hours / 6.0, 0, 1)
blended_hourly = []
feather_factors = []
for h in range(7):
    if radar_valid_mask_hourly[h].any():
        dist_to_edge_px = distance_transform_edt(radar_valid_mask_hourly[h])
        feather = np.clip(dist_to_edge_px / feather_px, 0, 1)
    else:
        feather = np.zeros_like(radar_extrap_filled[h])
    effective_w = w_radar_base[h] * feather
    feather_factors.append(feather)
    blended_hourly.append(effective_w * radar_extrap_filled[h] + (1 - effective_w) * hrrr_on_radar_grid[h])
blended_hourly = np.array(blended_hourly)

# ================================================================
# 7) verification: CSI vs lead time for radar-only / HRRR-only / blended
# ================================================================
print("\n=== verification ===")
csi_radar = {th: [] for th in CSI_THRESHOLDS}
csi_hrrr = {th: [] for th in CSI_THRESHOLDS}
csi_blend = {th: [] for th in CSI_THRESHOLDS}


def csi(forecast, obs, th, mask=None):
    fc_bin = forecast > th
    obs_bin = obs > th
    if mask is not None:
        fc_bin, obs_bin = fc_bin[mask], obs_bin[mask]
    hits = np.sum(fc_bin & obs_bin)
    misses = np.sum(~fc_bin & obs_bin)
    false_alarms = np.sum(fc_bin & ~obs_bin)
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else np.nan


for h in range(7):
    # radar-only: score only over its still-valid extrapolated coverage, so it isn't unfairly
    # penalized for the region it geometrically could never forecast (fixed-domain advection artifact)
    mask_h = radar_valid_mask_hourly[h]
    for th in CSI_THRESHOLDS:
        csi_radar[th].append(csi(radar_extrap_filled[h], obs_hourly[h], th, mask=mask_h) if mask_h.any() else np.nan)
        csi_hrrr[th].append(csi(hrrr_on_radar_grid[h], obs_hourly[h], th))
        csi_blend[th].append(csi(blended_hourly[h], obs_hourly[h], th))

fig, axes = plt.subplots(1, len(CSI_THRESHOLDS), figsize=(6 * len(CSI_THRESHOLDS), 5.5))
for ax, th in zip(axes, CSI_THRESHOLDS):
    ax.plot(lead_hours, csi_radar[th], '-o', label='纯雷达外推', color='#1e9bff')
    ax.plot(lead_hours, csi_hrrr[th], '-s', label='纯HRRR预报', color='#ff8c3d')
    ax.plot(lead_hours, csi_blend[th], '-^', label='融合(雷达+HRRR)', color='#2ecc71', linewidth=2.5)
    ax.set_xlabel("预报时效 (小时)")
    ax.set_ylabel("CSI")
    ax.set_title(f"{th:g} dBZ 阈值")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("雷达外推 vs HRRR预报 vs 融合 —— CSI随预报时效变化 (Rolling Fork个例)")
plt.savefig(f"{OUT_PREFIX}_CSI_comparison.png", dpi=180, bbox_inches='tight')
plt.close(fig)

print("\n--- CSI(30dBZ) 汇总 ---")
for h in range(7):
    print(f"+{h}h: 雷达={csi_radar[30.0][h]:.3f}  HRRR={csi_hrrr[30.0][h]:.3f}  融合={csi_blend[30.0][h]:.3f}")

# ================================================================
# 8) snapshot comparison at representative lead times
# ================================================================
snapshot_hours = [0, 1, 2, 3, 4, 6]
levels = [-20, 0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60]
fig2, axes2 = plt.subplots(4, len(snapshot_hours), figsize=(4.0 * len(snapshot_hours), 15))
for col, h in enumerate(snapshot_hours):
    panels = [
        (obs_hourly[h], "实况观测", None),
        (radar_extrap_hourly[h], "纯雷达外推", radar_valid_mask_hourly[h]),
        (hrrr_on_radar_grid[h], "纯HRRR预报", None),
        (blended_hourly[h], "融合", None),
    ]
    for row, (field, label, cov_mask) in enumerate(panels):
        ax = axes2[row, col]
        ax.set_facecolor('#dddddd')  # grey background = "no forecast coverage here" (not "clear sky")
        cf = ax.contourf(field, levels=levels, cmap='NWSRef', extend='both')
        title = f"{label} +{h}h" if row == 0 else f"+{h}h"
        if cov_mask is not None:
            title += f"\n(覆盖率{cov_mask.mean():.0%})"
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    axes2[0, col].set_title(f"实况观测 +{h}h")
for row, label in enumerate(["实况观测", "纯雷达外推", "纯HRRR预报", "融合"]):
    axes2[row, 0].set_ylabel(label, fontsize=13)
fig2.colorbar(cf, ax=axes2, label="合成反射率 (dBZ)", shrink=0.6, location='right')
fig2.suptitle("雷达外推 / HRRR预报 / 融合 三者对比 (Rolling Fork个例, 2023-03-25)")
plt.savefig(f"{OUT_PREFIX}_snapshots.png", dpi=160, bbox_inches='tight')
plt.close(fig2)

print("\ndone:", f"{OUT_PREFIX}_CSI_comparison.png, {OUT_PREFIX}_snapshots.png")
