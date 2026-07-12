"""
NEXRAD Level II volume scan -> 3D storm structure analysis.
Produces:
  1) CAPPI horizontal reflectivity slices at multiple heights
  2) vertical cross-section (RHI-style) through the storm core, reflectivity + dealiased velocity
  3) 3D reflectivity isosurface (marching cubes) showing overall storm shape

pip install arm_pyart scikit-image netCDF4 matplotlib numpy
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import pyart

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- config ----------------
RADAR_FILE = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado\KDGX20230325_013047_V06"
OUT_PREFIX = "rollingfork_tornado"

GRID_HALF_EXTENT_KM = 150     # grid covers +/- this range from radar in x and y (storm is ~110km out this time)
GRID_RES_KM = 0.5
Z_TOP_KM = 15
Z_RES_KM = 0.5
CAPPI_LEVELS_KM = [1, 2, 3, 4, 5, 6, 8, 10]   # heights to show as horizontal slices
REFL_ISOSURFACE_DBZ = 35                        # threshold for the 3D storm shape
STORM_SEARCH_RADIUS_KM = 150                    # ignore reflectivity beyond this range when finding storm core (avoids ground clutter near radar / distant unrelated echoes)

# ---------------- load ----------------
radar = pyart.io.read_nexrad_archive(RADAR_FILE)
print(f"loaded {RADAR_FILE}: {radar.nsweeps} sweeps, elevations = {np.round(radar.fixed_angle['data'], 1)}")

# ---------------- dealias velocity (needed for a physically meaningful field) ----------------
gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_masked('reflectivity')
dealias_vel = pyart.correct.dealias_region_based(
    radar, vel_field='velocity', keep_original=False, gatefilter=gatefilter
)
radar.add_field('corrected_velocity', dealias_vel, replace_existing=True)

# ---------------- grid to regular Cartesian 3D ----------------
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
    fields=['reflectivity', 'corrected_velocity'],
    weighting_function='Barnes2',
)

refl = grid.fields['reflectivity']['data']              # (nz, ny, nx), masked array, dBZ
vel = grid.fields['corrected_velocity']['data']         # (nz, ny, nx), masked array, m/s
z_km = grid.z['data'] / 1000
y_km = grid.y['data'] / 1000
x_km = grid.x['data'] / 1000

# ---------------- locate storm core (max low-level reflectivity within search radius) ----------------
xx, yy = np.meshgrid(x_km, y_km)
range_km = np.sqrt(xx ** 2 + yy ** 2)
low_level_idx = np.argmin(np.abs(z_km - 2))            # ~2km level, representative of storm's low-level core
refl_low = np.ma.filled(refl[low_level_idx], -999)
refl_low = np.where(range_km <= STORM_SEARCH_RADIUS_KM, refl_low, -999)
core_j, core_i = np.unravel_index(np.argmax(refl_low), refl_low.shape)
storm_x_km, storm_y_km = x_km[core_i], y_km[core_j]
print(f"storm core located at x={storm_x_km:.1f}km, y={storm_y_km:.1f}km from radar (range={range_km[core_j, core_i]:.1f}km)")

# ================================================================
# 1) CAPPI horizontal slices at multiple heights
# ================================================================
ncols = 4
nrows = int(np.ceil(len(CAPPI_LEVELS_KM) / ncols))
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)
for idx, h_km in enumerate(CAPPI_LEVELS_KM):
    ax = axes1[idx // ncols][idx % ncols]
    z_idx = np.argmin(np.abs(z_km - h_km))
    cf = ax.pcolormesh(x_km, y_km, refl[z_idx], cmap='NWSRef', vmin=-10, vmax=70)
    ax.plot(storm_x_km, storm_y_km, 'k+', markersize=12, markeredgewidth=2)
    ax.set_title(f"{z_km[z_idx]:.1f} km 高度反射率")
    ax.set_xlabel("距雷达东西方向 (km)")
    ax.set_ylabel("距雷达南北方向 (km)")
    ax.set_aspect('equal')
    ax.set_xlim(storm_x_km - 40, storm_x_km + 40)
    ax.set_ylim(storm_y_km - 40, storm_y_km + 40)
for idx in range(len(CAPPI_LEVELS_KM), nrows * ncols):
    axes1[idx // ncols][idx % ncols].axis('off')
fig1.colorbar(cf, ax=axes1, label="反射率 (dBZ)", shrink=0.6)
fig1.suptitle("超级单体不同高度水平切片 (CAPPI)")
fig1.savefig(f"{OUT_PREFIX}_cappi_levels.png", dpi=200, bbox_inches='tight')
plt.close(fig1)

# ================================================================
# 2) vertical cross-section through storm core (RHI-style)
#    one slice N-S through storm_x, one slice E-W through storm_y
# ================================================================
i_storm = np.argmin(np.abs(x_km - storm_x_km))
j_storm = np.argmin(np.abs(y_km - storm_y_km))

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

cf1 = axes2[0, 0].pcolormesh(y_km, z_km, refl[:, :, i_storm], cmap='NWSRef', vmin=-10, vmax=70)
axes2[0, 0].set_title(f"反射率垂直剖面 (南北向, x={storm_x_km:.1f}km)")
axes2[0, 0].set_xlabel("南北方向距离 (km)")
axes2[0, 0].set_ylabel("高度 (km)")
axes2[0, 0].set_xlim(storm_y_km - 30, storm_y_km + 30)
plt.colorbar(cf1, ax=axes2[0, 0], label="dBZ")

cf2 = axes2[0, 1].pcolormesh(x_km, z_km, refl[:, j_storm, :], cmap='NWSRef', vmin=-10, vmax=70)
axes2[0, 1].set_title(f"反射率垂直剖面 (东西向, y={storm_y_km:.1f}km)")
axes2[0, 1].set_xlabel("东西方向距离 (km)")
axes2[0, 1].set_ylabel("高度 (km)")
axes2[0, 1].set_xlim(storm_x_km - 30, storm_x_km + 30)
plt.colorbar(cf2, ax=axes2[0, 1], label="dBZ")

cf3 = axes2[1, 0].pcolormesh(y_km, z_km, vel[:, :, i_storm], cmap='BuDRd18', vmin=-40, vmax=40)
axes2[1, 0].set_title("径向速度垂直剖面 (南北向) -- 找中气旋/龙卷涡旋特征看这里")
axes2[1, 0].set_xlabel("南北方向距离 (km)")
axes2[1, 0].set_ylabel("高度 (km)")
axes2[1, 0].set_xlim(storm_y_km - 30, storm_y_km + 30)
plt.colorbar(cf3, ax=axes2[1, 0], label="m/s")

cf4 = axes2[1, 1].pcolormesh(x_km, z_km, vel[:, j_storm, :], cmap='BuDRd18', vmin=-40, vmax=40)
axes2[1, 1].set_title("径向速度垂直剖面 (东西向)")
axes2[1, 1].set_xlabel("东西方向距离 (km)")
axes2[1, 1].set_ylabel("高度 (km)")
axes2[1, 1].set_xlim(storm_x_km - 30, storm_x_km + 30)
plt.colorbar(cf4, ax=axes2[1, 1], label="m/s")

fig2.suptitle("过风暴核心的垂直剖面 (RHI风格)")
fig2.tight_layout()
fig2.savefig(f"{OUT_PREFIX}_vertical_cross_section.png", dpi=200, bbox_inches='tight')
plt.close(fig2)

# ================================================================
# 3) true 3D isosurface of reflectivity (marching cubes) -- overall storm shape
# ================================================================
refl_filled = np.ma.filled(refl, -999)
# crop to a box around the storm core to keep the isosurface computation reasonably sized
crop_km = 40
xi0, xi1 = np.searchsorted(x_km, storm_x_km - crop_km), np.searchsorted(x_km, storm_x_km + crop_km)
yi0, yi1 = np.searchsorted(y_km, storm_y_km - crop_km), np.searchsorted(y_km, storm_y_km + crop_km)
vol = refl_filled[:, yi0:yi1, xi0:xi1]

verts, faces, normals, values = measure.marching_cubes(vol, level=REFL_ISOSURFACE_DBZ)

# convert voxel indices back to physical coordinates (km)
verts_x = x_km[xi0] + verts[:, 2] * GRID_RES_KM
verts_y = y_km[yi0] + verts[:, 1] * GRID_RES_KM
verts_z = verts[:, 0] * Z_RES_KM

fig3 = plt.figure(figsize=(10, 9))
ax3 = fig3.add_subplot(111, projection='3d')
mesh = Poly3DCollection(np.stack([verts_x[faces], verts_y[faces], verts_z[faces]], axis=-1))
# color faces by height for a clear sense of vertical structure
face_heights = verts_z[faces].mean(axis=1)
mesh.set_array(face_heights)
mesh.set_cmap('viridis')
mesh.set_clim(0, Z_TOP_KM)
ax3.add_collection3d(mesh)
ax3.set_xlim(storm_x_km - crop_km, storm_x_km + crop_km)
ax3.set_ylim(storm_y_km - crop_km, storm_y_km + crop_km)
ax3.set_zlim(0, Z_TOP_KM)
ax3.set_xlabel("东西方向 (km)")
ax3.set_ylabel("南北方向 (km)")
ax3.set_zlabel("高度 (km)")
ax3.set_title(f"超级单体三维结构 ({REFL_ISOSURFACE_DBZ} dBZ 等值面, 颜色=高度)")
cb = fig3.colorbar(mesh, ax=ax3, shrink=0.6, pad=0.1)
cb.set_label("高度 (km)")
fig3.savefig(f"{OUT_PREFIX}_3d_isosurface.png", dpi=200, bbox_inches='tight')
plt.close(fig3)

print("done:")
print(f"  {OUT_PREFIX}_cappi_levels.png       -- 多高度水平切片")
print(f"  {OUT_PREFIX}_vertical_cross_section.png -- 过风暴核心垂直剖面(反射率+速度)")
print(f"  {OUT_PREFIX}_3d_isosurface.png      -- 三维反射率等值面结构图")
