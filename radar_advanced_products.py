"""
Rolling Fork (2023-03-25) 超级单体 —— 进阶雷达产品:
  1) VIL (垂直积分液态水含量) —— 从三维格点反射率用Greene-Clark公式积分
  2) 方位角剪切 (Azimuthal Shear) —— 原生极坐标下计算，探测中气旋/龙卷涡旋signature
  3) 双偏振速览 (ZDR + RHOHV 最低仰角) —— 检查有没有龙卷碎片特征(TDS)迹象

pip install arm_pyart scikit-image netCDF4 matplotlib numpy
"""
import numpy as np
import matplotlib.pyplot as plt
import pyart

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

RADAR_FILE = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado\KDGX20230325_013047_V06"
OUT_PREFIX = "rollingfork"

GRID_HALF_EXTENT_KM = 150
GRID_RES_KM = 1.0
Z_TOP_KM = 15
Z_RES_KM = 0.5
STORM_SEARCH_RADIUS_KM = 150
CROP_KM = 40  # crop around storm core for display

radar = pyart.io.read_nexrad_archive(RADAR_FILE)
print(f"loaded: {radar.nsweeps} sweeps")

# ---------------- dealias velocity ----------------
gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_masked('reflectivity')
dealias_vel = pyart.correct.dealias_region_based(
    radar, vel_field='velocity', keep_original=False, gatefilter=gatefilter
)
radar.add_field('corrected_velocity', dealias_vel, replace_existing=True)

# ================================================================
# locate storm core using lowest-sweep reflectivity (same approach as before)
# ================================================================
sweep0 = 0
refl0 = radar.get_field(sweep0, 'reflectivity')
sl0 = radar.get_slice(sweep0)
x0 = radar.gate_x['data'][sl0] / 1000
y0 = radar.gate_y['data'][sl0] / 1000
rng0 = np.sqrt(x0 ** 2 + y0 ** 2)
refl0_filled = np.ma.filled(refl0, -999)
valid = (rng0 > 8) & (rng0 < STORM_SEARCH_RADIUS_KM) & (refl0_filled > 45)
core_idx = np.unravel_index(np.argmax(np.where(valid, refl0_filled, -999)), refl0_filled.shape)
storm_x_km = x0[core_idx]
storm_y_km = y0[core_idx]
print(f"storm core: x={storm_x_km:.1f}km y={storm_y_km:.1f}km")

# ================================================================
# 1) VIL from gridded reflectivity (Greene & Clark 1972)
# ================================================================
half_m = GRID_HALF_EXTENT_KM * 1000
res_m = GRID_RES_KM * 1000
z_top_m = Z_TOP_KM * 1000
z_res_m = Z_RES_KM * 1000
nxy = int(2 * half_m / res_m) + 1
nz = int(z_top_m / z_res_m) + 1

grid = pyart.map.grid_from_radars(
    (radar,),
    grid_shape=(nz, nxy, nxy),
    grid_limits=((0, z_top_m), (-half_m, half_m), (-half_m, half_m)),
    fields=['reflectivity'],
    weighting_function='Barnes2',
)
refl_grid = np.ma.filled(grid.fields['reflectivity']['data'], -30)  # (nz, ny, nx) dBZ
z_km = grid.z['data'] / 1000
y_km = grid.y['data'] / 1000
x_km = grid.x['data'] / 1000

Z_linear = 10 ** (np.clip(refl_grid, -30, 75) / 10.0)  # mm^6/m^3
M = 3.44e-6 * Z_linear ** (4.0 / 7.0)  # g/m^3, liquid water content proxy
dz_m = Z_RES_KM * 1000
# Greene & Clark (1972) VIL: sum of (M_i+M_i+1)/2 * dh(m), result directly in kg/m^2
# (the empirical 3.44e-6 constant already folds in the necessary unit conversion --
#  an extra /1000 here would underestimate VIL by 3 orders of magnitude)
vil = np.sum((M[:-1] + M[1:]) / 2 * dz_m, axis=0)  # kg/m^2

fig1 = plt.figure(figsize=(8, 7))
xi0, xi1 = np.searchsorted(x_km, storm_x_km - CROP_KM), np.searchsorted(x_km, storm_x_km + CROP_KM)
yi0, yi1 = np.searchsorted(y_km, storm_y_km - CROP_KM), np.searchsorted(y_km, storm_y_km + CROP_KM)
cf = plt.contourf(x_km[xi0:xi1], y_km[yi0:yi1], vil[yi0:yi1, xi0:xi1],
                   levels=[0, 5, 10, 15, 20, 30, 40, 50, 65, 80], cmap='NWSRef', extend='max')
plt.plot(storm_x_km, storm_y_km, 'k+', markersize=14, markeredgewidth=2)
plt.colorbar(cf, label='VIL (kg/m$^2$ ≈ mm)')
plt.xlabel('东西方向距雷达 (km)')
plt.ylabel('南北方向距雷达 (km)')
plt.title('垂直积分液态水含量 VIL（Rolling Fork超级单体）')
plt.gca().set_aspect('equal')
plt.savefig(f"{OUT_PREFIX}_VIL.png", dpi=200, bbox_inches='tight')
plt.close(fig1)
print(f"VIL max: {vil.max():.1f} kg/m^2")

# ================================================================
# 2) Azimuthal shear at lowest tilt (native polar geometry) -- mesocyclone/TVS signature
# ================================================================
sweep_low = 1  # 0.3 deg Doppler (split-cut) sweep -- sweep 0 at this elevation is surveillance-only (no velocity)
vel_low = np.ma.filled(radar.get_field(sweep_low, 'corrected_velocity'), np.nan)  # (nrays, ngates)
azimuths = radar.get_azimuth(sweep_low)  # degrees, per ray
ranges = radar.range['data'] / 1000  # km, per gate

order = np.argsort(azimuths)
az_sorted = azimuths[order]
vel_sorted = vel_low[order]
valid_mask = ~np.isnan(vel_sorted)
vel_filled = np.where(valid_mask, vel_sorted, 0.0)  # np.gradient can't handle NaN -- it propagates
az_rad = np.deg2rad(az_sorted)
daz = np.gradient(np.unwrap(az_rad))  # radians between consecutive rays

dvel_daz = np.gradient(vel_filled, axis=0) / daz[:, None]
shear = dvel_daz / (ranges[None, :] * 1000)  # s^-1 (range in meters)
# re-mask anywhere the original data (or its immediate neighbors, since gradient uses them) was missing
neighbor_invalid = ~valid_mask | ~np.roll(valid_mask, 1, axis=0) | ~np.roll(valid_mask, -1, axis=0)
shear = np.where(neighbor_invalid, np.nan, shear)
shear = np.clip(shear, -0.02, 0.02)

lat0 = radar.gate_latitude['data']
lon0 = radar.gate_longitude['data']
sl_low = radar.get_slice(sweep_low)
x_low = radar.gate_x['data'][sl_low][order] / 1000
y_low = radar.gate_y['data'][sl_low][order] / 1000

fig2 = plt.figure(figsize=(8, 7))
sc = plt.pcolormesh(x_low, y_low, shear, cmap='BuDRd18', vmin=-0.015, vmax=0.015, shading='auto')
plt.plot(storm_x_km, storm_y_km, 'k+', markersize=14, markeredgewidth=2)
plt.xlim(storm_x_km - CROP_KM, storm_x_km + CROP_KM)
plt.ylim(storm_y_km - CROP_KM, storm_y_km + CROP_KM)
plt.colorbar(sc, label='方位角剪切 (s$^{-1}$)')
plt.xlabel('东西方向距雷达 (km)')
plt.ylabel('南北方向距雷达 (km)')
plt.title(f'低层(0.3°)方位角剪切 —— 找中气旋/龙卷涡旋看红蓝相邻的couplet')
plt.gca().set_aspect('equal')
plt.savefig(f"{OUT_PREFIX}_azshear.png", dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"max |shear| near core: {np.nanmax(np.abs(shear)):.4f} s^-1")

# ================================================================
# 3) Dual-pol quick look at lowest tilt: ZDR + RHOHV (look for TDS)
#    dual-pol moments live on the surveillance (sweep 0) cut, not the Doppler (sweep 1) cut
# ================================================================
sweep_dp = 0
az_dp = radar.get_azimuth(sweep_dp)
order_dp = np.argsort(az_dp)
sl_dp = radar.get_slice(sweep_dp)
x_dp = radar.gate_x['data'][sl_dp][order_dp] / 1000
y_dp = radar.gate_y['data'][sl_dp][order_dp] / 1000
zdr_low = np.ma.filled(radar.get_field(sweep_dp, 'differential_reflectivity'), np.nan)[order_dp]
rhohv_low = np.ma.filled(radar.get_field(sweep_dp, 'cross_correlation_ratio'), np.nan)[order_dp]

fig3, (axa, axb) = plt.subplots(1, 2, figsize=(15, 7))
c1 = axa.pcolormesh(x_dp, y_dp, zdr_low, cmap='NWS_SPW', vmin=-2, vmax=6, shading='auto')
axa.plot(storm_x_km, storm_y_km, 'k+', markersize=14, markeredgewidth=2)
axa.set_xlim(storm_x_km - CROP_KM, storm_x_km + CROP_KM)
axa.set_ylim(storm_y_km - CROP_KM, storm_y_km + CROP_KM)
axa.set_aspect('equal')
axa.set_title('差分反射率 ZDR (dB)')
plt.colorbar(c1, ax=axa, shrink=0.8)

c2 = axb.pcolormesh(x_dp, y_dp, rhohv_low, cmap='NWS_CC', vmin=0.5, vmax=1.05, shading='auto')
axb.plot(storm_x_km, storm_y_km, 'k+', markersize=14, markeredgewidth=2)
axb.set_xlim(storm_x_km - CROP_KM, storm_x_km + CROP_KM)
axb.set_ylim(storm_y_km - CROP_KM, storm_y_km + CROP_KM)
axb.set_aspect('equal')
axb.set_title('相关系数 RHOHV —— 低值(<0.8)可能是龙卷碎片(TDS)')
plt.colorbar(c2, ax=axb, shrink=0.8)

fig3.suptitle('双偏振速览（最低仰角0.3°，监视扫描）')
plt.savefig(f"{OUT_PREFIX}_dualpol.png", dpi=200, bbox_inches='tight')
plt.close(fig3)

# report min RHOHV near storm core
dist_dp = np.sqrt((x_dp - storm_x_km) ** 2 + (y_dp - storm_y_km) ** 2)
near_core = dist_dp < 10
rhohv_near_core = rhohv_low[near_core]
if np.any(~np.isnan(rhohv_near_core)):
    print(f"min RHOHV within 10km of core: {np.nanmin(rhohv_near_core):.3f}")
    print(f"points with RHOHV<0.8 within 10km of core: {np.nansum(rhohv_near_core < 0.8)}")
else:
    print("no valid RHOHV data within 10km of core")

print("done:", f"{OUT_PREFIX}_VIL.png, _azshear.png, _dualpol.png")
