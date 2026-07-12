"""
交互式3D超级单体结构 —— 粒子云版本 (Plotly Scatter3d)
与 make_interactive_3d.py 的等值面版不同：这里把体扫网格里每个格点渲染成一个"粒子"，
粒子的保留概率(密度)和透明度都随回波强度(dBZ)增大而增大——弱回波稀疏透明，
强回波核心稠密不透明，视觉上更接近云状/粒子状的风暴结构。
"""
import numpy as np
import plotly.graph_objects as go
import pyart

RADAR_FILE = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado\KDGX20230325_013047_V06"
OUT_FRAGMENT = r"F:\研\科研\Python\radar_analysis\plotly_fragment_particle.html"

GRID_HALF_EXTENT_KM = 150
GRID_RES_KM = 0.5
Z_TOP_KM = 15
Z_RES_KM = 0.5
STORM_SEARCH_RADIUS_KM = 150
CROP_KM = 40

REFL_MIN_DBZ = 10.0     # 低于此值视为噪声/无回波，直接丢弃
REFL_MAX_DBZ = 70.0     # 归一化上限（用于颜色/密度/透明度映射），对齐经典雷达回波色标最高档
MIN_KEEP_PROB = 0.12    # 弱回波格点仍以小概率保留，避免完全消失
MAX_POINTS = 120000     # 性能上限，超出则整体等比例再抽稀

radar = pyart.io.read_nexrad_archive(RADAR_FILE)

half_m = GRID_HALF_EXTENT_KM * 1000
res_m = GRID_RES_KM * 1000
z_top_m = Z_TOP_KM * 1000
z_res_m = Z_RES_KM * 1000
nxy = int(2 * half_m / res_m) + 1
nz = int(z_top_m / z_res_m) + 1

grid = pyart.map.grid_from_radars(
    (radar,), grid_shape=(nz, nxy, nxy),
    grid_limits=((0, z_top_m), (-half_m, half_m), (-half_m, half_m)),
    fields=['reflectivity'], weighting_function='Barnes2',
)
refl = np.ma.filled(grid.fields['reflectivity']['data'], -999)
z_km = grid.z['data'] / 1000
y_km = grid.y['data'] / 1000
x_km = grid.x['data'] / 1000

xx, yy = np.meshgrid(x_km, y_km)
range_km = np.sqrt(xx ** 2 + yy ** 2)
low_idx = np.argmin(np.abs(z_km - 2))
refl_low = np.where(range_km <= STORM_SEARCH_RADIUS_KM, refl[low_idx], -999)
core_j, core_i = np.unravel_index(np.argmax(refl_low), refl_low.shape)
storm_x, storm_y = x_km[core_i], y_km[core_j]

xi0, xi1 = np.searchsorted(x_km, storm_x - CROP_KM), np.searchsorted(x_km, storm_x + CROP_KM)
yi0, yi1 = np.searchsorted(y_km, storm_y - CROP_KM), np.searchsorted(y_km, storm_y + CROP_KM)
vol = refl[:, yi0:yi1, xi0:xi1]
vx = x_km[xi0:xi1]
vy = y_km[yi0:yi1]
vz = z_km

# ---------------- flatten volume to candidate particle points ----------------
ZZ, YY, XX = np.meshgrid(vz, vy, vx, indexing='ij')
vals = vol.ravel()
mask = vals >= REFL_MIN_DBZ
xs, ys, zs, dbz = XX.ravel()[mask], YY.ravel()[mask], ZZ.ravel()[mask], vals[mask]
print(f"candidate points above {REFL_MIN_DBZ}dBZ: {len(dbz)}")

# normalized intensity in [0,1]
norm = np.clip((dbz - REFL_MIN_DBZ) / (REFL_MAX_DBZ - REFL_MIN_DBZ), 0, 1)

# ---------------- importance-sample: keep-probability grows with intensity ----------------
keep_prob = MIN_KEEP_PROB + (1 - MIN_KEEP_PROB) * norm
rng = np.random.default_rng(42)
keep_mask = rng.random(len(norm)) < keep_prob
xs, ys, zs, dbz, norm = xs[keep_mask], ys[keep_mask], zs[keep_mask], dbz[keep_mask], norm[keep_mask]
print(f"kept after density thinning: {len(dbz)}")

if len(dbz) > MAX_POINTS:
    idx = rng.choice(len(dbz), size=MAX_POINTS, replace=False)
    xs, ys, zs, dbz, norm = xs[idx], ys[idx], zs[idx], dbz[idx], norm[idx]
    print(f"capped to {MAX_POINTS} points for performance")

# ---------------- color: custom radar-style ramp, interpolated per-point ----------------
stops = np.array([0.0, 0.35, 0.6, 0.8, 1.0])
colors_r = np.array([0, 0, 0, 255, 255])
colors_g = np.array([59, 163, 217, 184, 140])
colors_b = np.array([77, 196, 255, 77, 61])
r = np.interp(norm, stops, colors_r).astype(int)
g = np.interp(norm, stops, colors_g).astype(int)
b = np.interp(norm, stops, colors_b).astype(int)

# ---------------- alpha: transparency also grows with intensity ----------------
alpha = 0.08 + 0.82 * norm
rgba = [f"rgba({r[i]},{g[i]},{b[i]},{alpha[i]:.3f})" for i in range(len(norm))]

# size: subtle boost for stronger echoes, reinforcing the "dense core" look
size = 1.6 + 2.4 * norm

fig = go.Figure(data=[go.Scatter3d(
    x=xs, y=ys, z=zs,
    mode='markers',
    marker=dict(size=size, color=rgba, line=dict(width=0)),
    name='反射率粒子',
    hovertemplate='dBZ=%{customdata:.0f}<extra></extra>',
    customdata=dbz,
)])

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        xaxis=dict(title='东西方向 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        yaxis=dict(title='南北方向 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        zaxis=dict(title='高度 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.6),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=720,
    showlegend=False,
)

html_fragment = fig.to_html(full_html=False, include_plotlyjs='inline', div_id='plot3d_particle')
with open(OUT_FRAGMENT, 'w', encoding='utf-8') as f:
    f.write(html_fragment)

print(f"storm core at x={storm_x:.1f}km y={storm_y:.1f}km, final particle count={len(dbz)}")
print("fragment written:", OUT_FRAGMENT, "size:", len(html_fragment))
