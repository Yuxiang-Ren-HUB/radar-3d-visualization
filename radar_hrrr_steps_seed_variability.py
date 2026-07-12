"""
量化 STEPS 集合融合本身的"跑一次到跑一次"随机噪声有多大 —— 固定8个集合成员、
运动补偿插值等配置不变，只换随机种子跑3次，看CSI波动范围跟之前"优化前后"的差异比哪个大。
如果种子间的波动 >= 优化前后的差异，说明上次看到的涨跌基本是噪声，不是真实效果。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as _dt, timedelta as _td
import re as _re
import pyart  # noqa: F401
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
N_ENS_MEMBERS = 8
N_CASCADE_LEVELS = 6
PRECIP_THR_DBZ = 15.0
CSI_THRESHOLDS = [20.0, 30.0, 40.0]
SEEDS = [42, 123, 2024]

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

# ================================================================
# deterministic setup (identical to radar_hrrr_steps_blend.py, independent of seed)
# ================================================================
R_input_full = refl_colmax_stack[:N_INPUT]
oflow = motion.get_method("LK")
velocity = oflow(R_input_full)
precip = R_input_full[-3:]

steps_per_hour = int(round(60 / radar_dt_min_input))
timesteps = [steps_per_hour * h for h in range(1, 7)]
n_native_steps = timesteps[-1]

extrapolate = nowcasts.get_method("extrapolation")
velocity_models_hourly = []
for i in range(hrrr_on_radar_grid.shape[0] - 1):
    pair = hrrr_on_radar_grid[i:i + 2]
    v = oflow(pair) / steps_per_hour
    velocity_models_hourly.append(v)

ny, nx = hrrr_on_radar_grid.shape[1:]
precip_models_native = np.empty((n_native_steps + 1, ny, nx), dtype=np.float64)
for h in range(len(velocity_models_hourly)):
    lo_frame, hi_frame = hrrr_on_radar_grid[h], hrrr_on_radar_grid[h + 1]
    v_h = velocity_models_hourly[h]
    base_step = h * steps_per_hour
    precip_models_native[base_step] = lo_frame
    forward_all = extrapolate(lo_frame, v_h, steps_per_hour)
    backward_all = extrapolate(hi_frame, -v_h, steps_per_hour)
    for s in range(1, steps_per_hour):
        frac = s / steps_per_hour
        fwd = forward_all[s - 1]
        bwd = backward_all[steps_per_hour - s - 1]
        blended = (1 - frac) * fwd + frac * bwd
        nanmask = np.isnan(blended)
        if nanmask.any():
            linear = (1 - frac) * lo_frame + frac * hi_frame
            blended = np.where(np.isnan(fwd) & np.isnan(bwd), linear,
                                np.where(np.isnan(fwd), bwd, np.where(np.isnan(bwd), fwd, blended)))
        precip_models_native[base_step + s] = blended
precip_models_native[n_native_steps] = hrrr_on_radar_grid[-1]
precip_models = precip_models_native[np.newaxis, ...]

velocity_models_native = np.empty((n_native_steps + 1, 2, ny, nx), dtype=np.float64)
for t in range(n_native_steps + 1):
    hour_idx = min(t // steps_per_hour, len(velocity_models_hourly) - 1)
    velocity_models_native[t] = velocity_models_hourly[hour_idx]
velocity_models = velocity_models_native[np.newaxis, ...]

issuetime = _times[N_INPUT - 1]

# ================================================================
# run (or load cached) ensemble for each seed
# ================================================================
hourly_output_idx = [h * steps_per_hour - 1 for h in range(1, 7)]
ensemble_means_by_seed = {}
for seed in SEEDS:
    cache_f = f"{OUT_PREFIX}_ensemble_all_seed{seed}.npy"
    if os.path.exists(cache_f):
        print(f"=== seed {seed}: loading cached {cache_f} ===")
        forecast_ens_all = np.load(cache_f)
    else:
        print(f"\n=== seed {seed}: running pysteps.blending.steps.forecast ===")
        forecast_ens_all = bsteps.forecast(
            precip=precip, precip_models=precip_models,
            velocity=velocity, velocity_models=velocity_models,
            timesteps=n_native_steps, timestep=radar_dt_min_input, issuetime=issuetime,
            n_ens_members=N_ENS_MEMBERS, n_cascade_levels=N_CASCADE_LEVELS,
            precip_thr=PRECIP_THR_DBZ, kmperpixel=RES_KM, seed=seed,
        )
        np.save(cache_f, forecast_ens_all)
    forecast_ens_future = forecast_ens_all[:, hourly_output_idx, :, :]
    forecast_ens_hourly = np.concatenate([
        np.repeat(precip[-1][np.newaxis, np.newaxis, :, :], N_ENS_MEMBERS, axis=0),
        forecast_ens_future,
    ], axis=1)
    ensemble_means_by_seed[seed] = forecast_ens_hourly.mean(axis=0)  # (7, ny, nx)
    print(f"seed {seed}: ensemble mean computed, shape={ensemble_means_by_seed[seed].shape}")

# ================================================================
# verification against real observations (same nearest-time matching as before)
# ================================================================
start_time = _times[N_INPUT - 1]
obs_hourly = []
for h in range(7):
    target_time = start_time + _td(hours=h)
    diffs = [abs((t - target_time).total_seconds()) for t in _times]
    obs_hourly.append(refl_colmax_stack[int(np.argmin(diffs))])
obs_hourly = np.array(obs_hourly)


def csi(forecast, obs, th):
    fc_bin, obs_bin = forecast > th, obs > th
    hits = np.sum(fc_bin & obs_bin)
    misses = np.sum(~fc_bin & obs_bin)
    false_alarms = np.sum(fc_bin & ~obs_bin)
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else np.nan


lead_hours = np.arange(0, 7)
csi_by_seed = {seed: {th: [csi(ensemble_means_by_seed[seed][h], obs_hourly[h], th) for h in range(7)]
                       for th in CSI_THRESHOLDS}
               for seed in SEEDS}

print("\n--- 30dBZ CSI, 三个随机种子对比 ---")
for h in range(7):
    vals = [csi_by_seed[s][30.0][h] for s in SEEDS]
    print(f"+{h}h: " + "  ".join(f"seed{s}={v:.3f}" for s, v in zip(SEEDS, vals)) +
          f"   std={np.std(vals):.3f}")

# ================================================================
# plot: individual seeds (thin) + mean (thick) + shaded min-max spread band
# ================================================================
fig, axes = plt.subplots(1, len(CSI_THRESHOLDS), figsize=(6 * len(CSI_THRESHOLDS), 5.5))
colors = ['#9b59b6', '#c39bd3', '#5b2c6f']
for ax, th in zip(axes, CSI_THRESHOLDS):
    seed_curves = np.array([csi_by_seed[s][th] for s in SEEDS])  # (3, 7)
    for s, color in zip(SEEDS, colors):
        ax.plot(lead_hours, csi_by_seed[s][th], '--o', color=color, alpha=0.6, markersize=4, label=f'种子{s}')
    mean_curve = np.nanmean(seed_curves, axis=0)
    ax.plot(lead_hours, mean_curve, '-D', color='#9b59b6', linewidth=2.5, markersize=7, label='3种子均值')
    ax.fill_between(lead_hours, np.nanmin(seed_curves, axis=0), np.nanmax(seed_curves, axis=0),
                     color='#9b59b6', alpha=0.15, label='种子间波动范围')
    ax.set_xlabel("预报时效 (小时)")
    ax.set_ylabel("CSI")
    ax.set_title(f"{th:g} dBZ 阈值")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("STEPS集合融合随机种子敏感性 —— 同配置跑3次的CSI波动范围 (Rolling Fork个例)")
plt.savefig(f"{OUT_PREFIX}_seed_variability.png", dpi=180, bbox_inches='tight')
plt.close(fig)
print("\nsaved:", f"{OUT_PREFIX}_seed_variability.png")

# explicit comparison against the earlier "optimization" delta (3mem/linear-interp -> 8mem/motion-comp)
optimization_delta_30dbz = {1: 0.243 - 0.232, 2: 0.191 - 0.174, 3: 0.129 - 0.151,
                             4: 0.201 - 0.165, 5: 0.176 - 0.170, 6: 0.223 - 0.211}
print("\n--- 种子间标准差 vs 优化前后差异 (30dBZ) ---")
for h in range(1, 7):
    vals = [csi_by_seed[s][30.0][h] for s in SEEDS]
    seed_std = np.std(vals)
    opt_delta = abs(optimization_delta_30dbz[h])
    verdict = "优化差异在噪声范围内" if opt_delta <= seed_std else "优化差异超出噪声范围"
    print(f"+{h}h: 种子间std={seed_std:.3f}  优化前后差={opt_delta:.3f}  -> {verdict}")
