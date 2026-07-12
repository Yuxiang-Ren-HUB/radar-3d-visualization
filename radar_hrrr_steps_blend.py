"""
真正的 pysteps STEPS blending 集合融合系统 (对比我们自己写的线性加权版本)
复用 radar_hrrr_blend.py 已经缓存好的雷达/HRRR网格化数据 (_radar_hrrr_blend_cache.npz)，
避免重新下载/网格化。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as _dt, timedelta as _td
import re as _re
import pyart  # noqa: F401 -- side effect: registers the NWSRef colormap used below
from pysteps import motion, nowcasts
from pysteps.blending import steps as bsteps

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

CACHE_NPZ = r"F:\研\科研\Python\radar_analysis\_radar_hrrr_blend_cache.npz"
OUT_PREFIX = "radar_hrrr_stepsblend"

RES_KM = 2.0
N_INPUT = 10
N_ENS_MEMBERS = 8   # was 3 -- more members for a more reliable ensemble mean / spread estimate
N_CASCADE_LEVELS = 6
PRECIP_THR_DBZ = 15.0
CSI_THRESHOLDS = [20.0, 30.0, 40.0]

print("=== loading cached grids ===")
_c = np.load(CACHE_NPZ, allow_pickle=True)
refl_colmax_stack = _c["refl_colmax_stack"].astype(np.float64)
radar_times = list(_c["radar_times"])
hrrr_on_radar_grid = _c["hrrr_on_radar_grid"].astype(np.float64)

_times = []
for fn in radar_times:
    m = _re.search(r"(\d{8})_(\d{6})_V\d\d", str(fn))
    _times.append(_dt.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
radar_dt_min_input = float(np.mean([(_times[i + 1] - _times[i]).total_seconds() / 60
                                     for i in range(N_INPUT - 1)]))
print(f"input-window scan interval: {radar_dt_min_input:.2f} min")

# ================================================================
# build the pieces pysteps.blending.steps.forecast expects
# ================================================================
R_input_full = refl_colmax_stack[:N_INPUT]
oflow = motion.get_method("LK")
velocity = oflow(R_input_full)  # (2, ny, nx), represents one native radar step

precip = R_input_full[-3:]  # (ar_order+1=3, ny, nx): most recent frames for the AR(2) model

steps_per_hour = int(round(60 / radar_dt_min_input))
timesteps = [steps_per_hour * h for h in range(1, 7)]  # native-step indices for +1h..+6h
print("timesteps (native steps):", timesteps)
n_native_steps = timesteps[-1]  # pysteps requires precip_models at every native step, not just hourly

# NWP motion: derive one motion field per hour from optical flow (needed BEFORE interpolation,
# since interpolation itself is now motion-compensated rather than plain pixel blending)
print("=== estimating HRRR's own motion field between hourly frames ===")
extrapolate = nowcasts.get_method("extrapolation")
velocity_models_hourly = []
for i in range(hrrr_on_radar_grid.shape[0] - 1):
    pair = hrrr_on_radar_grid[i:i + 2]
    v = oflow(pair) / steps_per_hour  # per-hour motion -> per-native-step displacement
    velocity_models_hourly.append(v)

# HRRR's 7 hourly frames land exactly on native steps [0, steps_per_hour, 2*steps_per_hour, ..., 60]
# (by construction of steps_per_hour above). Instead of plain pixel-wise linear interpolation
# (which just cross-fades between anchors and blurs/ghosts a moving storm), advect the LO anchor
# forward and the HI anchor backward using the hour's own motion field, then blend the two
# motion-compensated estimates -- this keeps the storm's shape sharp between hourly frames.
print(f"=== motion-compensated interpolation of HRRR to {n_native_steps + 1} native-cadence frames ===")
ny, nx = hrrr_on_radar_grid.shape[1:]
precip_models_native = np.empty((n_native_steps + 1, ny, nx), dtype=np.float64)
for h in range(len(velocity_models_hourly)):
    lo_frame, hi_frame = hrrr_on_radar_grid[h], hrrr_on_radar_grid[h + 1]
    v_h = velocity_models_hourly[h]
    base_step = h * steps_per_hour
    precip_models_native[base_step] = lo_frame  # s=0 exactly matches the anchor
    forward_all = extrapolate(lo_frame, v_h, steps_per_hour)      # steps 1..steps_per_hour ahead of lo
    backward_all = extrapolate(hi_frame, -v_h, steps_per_hour)    # steps 1..steps_per_hour behind hi
    for s in range(1, steps_per_hour):
        frac = s / steps_per_hour
        fwd = forward_all[s - 1]
        bwd = backward_all[steps_per_hour - s - 1]
        blended = (1 - frac) * fwd + frac * bwd
        # extrapolation can leave NaN where its own domain got vacated -- fall back to whichever
        # motion-compensated estimate is valid, then to plain linear blend as a last resort
        nanmask = np.isnan(blended)
        if nanmask.any():
            linear = (1 - frac) * lo_frame + frac * hi_frame
            blended = np.where(np.isnan(fwd) & np.isnan(bwd), linear,
                                np.where(np.isnan(fwd), bwd, np.where(np.isnan(bwd), fwd, blended)))
        precip_models_native[base_step + s] = blended
precip_models_native[n_native_steps] = hrrr_on_radar_grid[-1]  # final anchor
precip_models = precip_models_native[np.newaxis, ...]  # (1, n_native_steps+1, ny, nx)
print("precip_models shape:", precip_models.shape)

# needs n_native_steps+1 entries (same off-by-one convention as precip_models -- confirmed by an
# IndexError at exactly index n_native_steps on an earlier run)
velocity_models_native = np.empty((n_native_steps + 1, 2, ny, nx), dtype=np.float64)
for t in range(n_native_steps + 1):
    hour_idx = min(t // steps_per_hour, len(velocity_models_hourly) - 1)
    velocity_models_native[t] = velocity_models_hourly[hour_idx]
velocity_models = velocity_models_native[np.newaxis, ...]  # (1, n_native_steps+1, 2, ny, nx)
print("velocity_models shape:", velocity_models.shape)

issuetime = _times[N_INPUT - 1]

# NOTE: passing `timesteps` as a sparse list (with gaps between requested outputs, e.g. [10,20,...,60])
# hits what looks like an internal pysteps bug in this version -- crashes with
# "TypeError: 'NoneType' object is not iterable" on the first skipped intermediate step, inside
# __blended_nowcast_main_loop's final_blended_forecast_all_members_one_timestep bookkeeping.
# Workaround: request every native step (plain int, no gaps) and subsample the hourly steps ourselves.
_all_cache = f"{OUT_PREFIX}_ensemble_all.npy"
if os.path.exists(_all_cache):
    print(f"=== loading cached raw ensemble from {_all_cache} ===")
    forecast_ens_all = np.load(_all_cache)
else:
    print("\n=== running pysteps.blending.steps.forecast (this is the slow part) ===")
    forecast_ens_all = bsteps.forecast(
        precip=precip,
        precip_models=precip_models,
        velocity=velocity,
        velocity_models=velocity_models,
        timesteps=n_native_steps,
        timestep=radar_dt_min_input,
        issuetime=issuetime,
        n_ens_members=N_ENS_MEMBERS,
        n_cascade_levels=N_CASCADE_LEVELS,
        precip_thr=PRECIP_THR_DBZ,
        kmperpixel=RES_KM,
        seed=42,
    )
    np.save(_all_cache, forecast_ens_all)
print("forecast_ens_all shape:", forecast_ens_all.shape)

# forecast_ens_all contains ONLY the 60 forecast steps (native step 1..60), NOT the t=0 analysis --
# so output array index i corresponds to native forecast step (i+1). Lead hour 0 (the analysis
# itself) isn't part of this array at all; use precip[-1] (last input radar frame) for that instead.
hourly_output_idx = [h * steps_per_hour - 1 for h in range(1, 7)]  # -> [9,19,29,39,49,59] for lead 1..6h
forecast_ens_future = forecast_ens_all[:, hourly_output_idx, :, :]  # (n_ens, 6, ny, nx), lead 1..6h
forecast_ens_hourly = np.concatenate([
    np.repeat(precip[-1][np.newaxis, np.newaxis, :, :], N_ENS_MEMBERS, axis=0),  # lead 0h = analysis
    forecast_ens_future,
], axis=1)  # (n_ens, 7, ny, nx), lead 0..6h
print("forecast_ens_hourly shape:", forecast_ens_hourly.shape)
np.save(f"{OUT_PREFIX}_ensemble.npy", forecast_ens_hourly)
print("saved ->", f"{OUT_PREFIX}_ensemble.npy")

steps_ensemble_mean_hourly = forecast_ens_hourly.mean(axis=0)  # (7, ny, nx)

# ================================================================
# verification: match each hourly checkpoint to the nearest real observation (same logic as
# radar_hrrr_blend.py), then compute CSI for the STEPS ensemble mean
# ================================================================
start_time = _times[N_INPUT - 1]
obs_hourly = []
for h in range(7):
    target_time = start_time + _td(hours=h)
    diffs = [abs((t - target_time).total_seconds()) for t in _times]
    nearest_idx = int(np.argmin(diffs))
    obs_hourly.append(refl_colmax_stack[nearest_idx])
obs_hourly = np.array(obs_hourly)


def csi(forecast, obs, th):
    fc_bin, obs_bin = forecast > th, obs > th
    hits = np.sum(fc_bin & obs_bin)
    misses = np.sum(~fc_bin & obs_bin)
    false_alarms = np.sum(fc_bin & ~obs_bin)
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else np.nan


csi_steps = {th: [csi(steps_ensemble_mean_hourly[h], obs_hourly[h], th) for h in range(7)]
             for th in CSI_THRESHOLDS}

print("\n--- STEPS集合融合 CSI(30dBZ) ---")
for h in range(7):
    print(f"+{h}h: STEPS融合={csi_steps[30.0][h]:.3f}")

lead_hours = np.arange(0, 7)

# ================================================================
# recompute the earlier simple radar-extrapolation / feathered-blend numbers here too (same logic
# as radar_hrrr_blend.py), so we can put all four methods on one honest side-by-side comparison
# ================================================================
print("\n=== recomputing simple radar extrapolation + feathered blend for a 4-way comparison ===")
from pysteps import nowcasts
from scipy.ndimage import distance_transform_edt

extrapolate = nowcasts.get_method("extrapolation")
n_leadtimes_native = int(6 * 60 / radar_dt_min_input)
R_forecast_dbz = extrapolate(R_input_full[-1], velocity, n_leadtimes_native)
hourly_idx_in_forecast = [min(h * steps_per_hour, R_forecast_dbz.shape[0] - 1) for h in range(7)]
radar_extrap_hourly = R_forecast_dbz[hourly_idx_in_forecast]
radar_valid_mask_hourly = ~np.isnan(radar_extrap_hourly)
radar_extrap_filled = np.where(radar_valid_mask_hourly, radar_extrap_hourly, -20.0)

w_radar_base = np.clip(1 - lead_hours / 6.0, 0, 1)
FEATHER_KM = 20.0
feather_px = FEATHER_KM / RES_KM
simple_blend_hourly = []
for h in range(7):
    if radar_valid_mask_hourly[h].any():
        dist_to_edge_px = distance_transform_edt(radar_valid_mask_hourly[h])
        feather = np.clip(dist_to_edge_px / feather_px, 0, 1)
    else:
        feather = np.zeros_like(radar_extrap_filled[h])
    effective_w = w_radar_base[h] * feather
    simple_blend_hourly.append(effective_w * radar_extrap_filled[h] + (1 - effective_w) * hrrr_on_radar_grid[h])
simple_blend_hourly = np.array(simple_blend_hourly)

csi_radar = {th: [] for th in CSI_THRESHOLDS}
csi_hrrr = {th: [] for th in CSI_THRESHOLDS}
csi_simple = {th: [] for th in CSI_THRESHOLDS}
for h in range(7):
    mask_h = radar_valid_mask_hourly[h]
    for th in CSI_THRESHOLDS:
        csi_radar[th].append(csi(radar_extrap_filled[h][mask_h], obs_hourly[h][mask_h], th) if mask_h.any() else np.nan)
        csi_hrrr[th].append(csi(hrrr_on_radar_grid[h], obs_hourly[h], th))
        csi_simple[th].append(csi(simple_blend_hourly[h], obs_hourly[h], th))

print("\n--- 30dBZ CSI 四方法对比 ---")
for h in range(7):
    r = csi_radar[30.0][h]
    print(f"+{h}h: 雷达={r if r!=r else f'{r:.3f}'}  HRRR={csi_hrrr[30.0][h]:.3f}  "
          f"简单融合={csi_simple[30.0][h]:.3f}  STEPS融合={csi_steps[30.0][h]:.3f}")

# ================================================================
# combined 4-way CSI comparison plot
# ================================================================
fig, axes = plt.subplots(1, len(CSI_THRESHOLDS), figsize=(6 * len(CSI_THRESHOLDS), 5.5))
for ax, th in zip(axes, CSI_THRESHOLDS):
    ax.plot(lead_hours, csi_radar[th], '-o', label='纯雷达外推', color='#1e9bff', alpha=0.7)
    ax.plot(lead_hours, csi_hrrr[th], '-s', label='纯HRRR预报', color='#ff8c3d', alpha=0.7)
    ax.plot(lead_hours, csi_simple[th], '-^', label='简单加权融合(自建)', color='#2ecc71', alpha=0.8)
    ax.plot(lead_hours, csi_steps[th], '-D', label='STEPS集合融合(真实pysteps)', color='#9b59b6', linewidth=2.5)
    ax.set_xlabel("预报时效 (小时)")
    ax.set_ylabel("CSI")
    ax.set_title(f"{th:g} dBZ 阈值")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle("雷达 / HRRR / 简单融合 / STEPS集合融合 —— 四方法CSI对比 (Rolling Fork个例)")
plt.savefig(f"{OUT_PREFIX}_CSI_4way.png", dpi=180, bbox_inches='tight')
plt.close(fig)
print("\nsaved:", f"{OUT_PREFIX}_CSI_4way.png")

# ================================================================
# combined 5-row snapshot: obs / radar / HRRR / simple blend / STEPS ensemble mean
# ================================================================
levels = [-20, 0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60]
snapshot_hours = [0, 1, 2, 3, 4, 6]
fig2, axes2 = plt.subplots(5, len(snapshot_hours), figsize=(4.0 * len(snapshot_hours), 18))
row_data = [
    (obs_hourly, "实况观测", None),
    (radar_extrap_filled, "纯雷达外推", radar_valid_mask_hourly),
    (hrrr_on_radar_grid, "纯HRRR预报", None),
    (simple_blend_hourly, "简单加权融合", None),
    (steps_ensemble_mean_hourly, "STEPS集合融合", None),
]
for row, (data, label, cov) in enumerate(row_data):
    for col, h in enumerate(snapshot_hours):
        ax = axes2[row, col]
        ax.set_facecolor('#dddddd')
        cf = ax.contourf(data[h], levels=levels, cmap='NWSRef', extend='both')
        title = f"+{h}h"
        if cov is not None:
            title += f" (覆盖率{cov[h].mean():.0%})"
        if row == 0:
            title = f"实况观测 {title}"
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    axes2[row, 0].set_ylabel(label, fontsize=13)
fig2.colorbar(cf, ax=axes2, label="合成反射率 (dBZ)", shrink=0.6, location='right')
fig2.suptitle("五方法快照对比 —— 实况/雷达外推/HRRR/简单融合/STEPS集合融合 (Rolling Fork个例)")
plt.savefig(f"{OUT_PREFIX}_snapshots_5way.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("saved:", f"{OUT_PREFIX}_snapshots_5way.png")

# ================================================================
# ensemble member spread: show individual members (not just the mean) to demonstrate the
# ensemble is genuinely stochastic/diverse, not just N copies of the same deterministic field
# ================================================================
spread_hours = [2, 4, 6]
fig3, axes3 = plt.subplots(N_ENS_MEMBERS + 1, len(spread_hours), figsize=(4.5 * len(spread_hours), 4 * (N_ENS_MEMBERS + 1)))
for col, h in enumerate(spread_hours):
    for m in range(N_ENS_MEMBERS):
        ax = axes3[m, col]
        cf = ax.contourf(forecast_ens_hourly[m, h], levels=levels, cmap='NWSRef', extend='both')
        ax.set_title(f"成员{m + 1} +{h}h", fontsize=10)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax = axes3[N_ENS_MEMBERS, col]
    ax.contourf(steps_ensemble_mean_hourly[h], levels=levels, cmap='NWSRef', extend='both')
    ax.set_title(f"集合均值 +{h}h", fontsize=10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for m in range(N_ENS_MEMBERS):
    axes3[m, 0].set_ylabel(f"成员{m + 1}", fontsize=12)
axes3[N_ENS_MEMBERS, 0].set_ylabel("集合均值", fontsize=12)
fig3.colorbar(cf, ax=axes3, label="合成反射率 (dBZ)", shrink=0.6, location='right')
fig3.suptitle(f"STEPS集合成员离散度 ({N_ENS_MEMBERS}个成员，证明确实是随机集合而非简单复制)")
plt.savefig(f"{OUT_PREFIX}_ensemble_spread.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print("saved:", f"{OUT_PREFIX}_ensemble_spread.png")

# ================================================================
# HRRR interpolation sanity check: 7 hourly anchors vs a few interpolated intermediate native steps
# ================================================================
fig4, axes4 = plt.subplots(1, 5, figsize=(20, 4.5))
check_steps = [0, 5, 10, 15, 20]  # spans F00 -> between F00/F01 -> F01 -> between -> F02
for i, t in enumerate(check_steps):
    ax = axes4[i]
    cf = ax.contourf(precip_models[0, t], levels=levels, cmap='NWSRef', extend='both')
    is_anchor = (t % steps_per_hour) == 0
    ax.set_title(f"native step {t}" + (" (HRRR原始帧)" if is_anchor else " (运动补偿插值)"), fontsize=10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
fig4.colorbar(cf, ax=axes4, label="合成反射率 (dBZ)", shrink=0.7)
fig4.suptitle("HRRR时间插值核查：从7个整点帧插值到雷达原生时间分辨率")
plt.savefig(f"{OUT_PREFIX}_hrrr_interp_check.png", dpi=150, bbox_inches='tight')
plt.close(fig4)
print("saved:", f"{OUT_PREFIX}_hrrr_interp_check.png")
