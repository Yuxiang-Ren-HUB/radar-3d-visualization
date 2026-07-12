"""
"流场"风格可视化 —— 基于单雷达径向速度
重要说明：单部雷达(单多普勒)只能测到目标沿雷达波束方向的速度分量(径向速度)，
测不到切向/垂直分量，因此无法重建真正的三维风矢量场。这里把每个格点的径向速度
还原成"沿雷达->格点连线方向"的三维箭头(大小=径向速度，方向=径向)，
可以直观看到大范围的辐合/辐散、以及旋转对应的相邻"朝雷达/离雷达"箭头对(速度对)，
但不代表真实、完整的三维气流结构（缺少切向环流分量）。
"""
import numpy as np
import plotly.graph_objects as go
import pyart

RADAR_FILE = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado\KDGX20230325_013047_V06"
OUT_FRAGMENT = r"F:\研\科研\Python\radar_analysis\plotly_fragment_flow.html"
PREVIEW_PNG = r"F:\研\科研\Python\radar_analysis\flow_field_preview.png"

FULL_HALF_KM = 150
SEARCH_RES_KM = 2.0
SEARCH_Z_KM = 2.0

CROP_KM = 45
GRID_RES_KM = 2.0
Z_TOP_KM = 15
Z_RES_KM = 1.0

REFL_MIN_DBZ = 10.0
REFL_MAX_DBZ = 70.0
CONE_STRIDE = 2          # subsample every Nth grid point per axis for arrows (avoid overcrowding)
VEL_MIN_ABS = 8.0        # only draw arrows where |radial velocity| exceeds this (m/s), to declutter

radar = pyart.io.read_nexrad_archive(RADAR_FILE)

# ---------------- dealias radial velocity (unfold Nyquist aliasing) ----------------
gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_masked('velocity')
dealias_field = pyart.correct.dealias_region_based(radar, vel_field='velocity', keep_original=False, gatefilter=gatefilter)
radar.add_field('velocity_dealiased', dealias_field, replace_existing=True)

# ---------------- pass 1: cheap low-level full-domain search for storm core ----------------
half_full_m = FULL_HALF_KM * 1000
res_search_m = SEARCH_RES_KM * 1000
nxy_search = int(2 * half_full_m / res_search_m) + 1
z_lo_m = (SEARCH_Z_KM - 0.5) * 1000
z_hi_m = (SEARCH_Z_KM + 0.5) * 1000

grid_search = pyart.map.grid_from_radars(
    (radar,), grid_shape=(1, nxy_search, nxy_search),
    grid_limits=((z_lo_m, z_hi_m), (-half_full_m, half_full_m), (-half_full_m, half_full_m)),
    fields=['reflectivity'], weighting_function='Barnes2',
)
refl_low = np.ma.filled(grid_search.fields['reflectivity']['data'][0], -999)
y_search = grid_search.y['data']
x_search = grid_search.x['data']
j, i = np.unravel_index(np.argmax(refl_low), refl_low.shape)
storm_x_m, storm_y_m = x_search[i], y_search[j]
print(f"storm core: x={storm_x_m/1000:.1f}km y={storm_y_m/1000:.1f}km")

# ---------------- pass 2: fine crop grid, both reflectivity + dealiased velocity ----------------
half_crop_m = CROP_KM * 1000
res_m = GRID_RES_KM * 1000
z_top_m = Z_TOP_KM * 1000
z_res_m = Z_RES_KM * 1000
nxy = int(2 * half_crop_m / res_m) + 1
nz = int(z_top_m / z_res_m) + 1

grid = pyart.map.grid_from_radars(
    (radar,), grid_shape=(nz, nxy, nxy),
    grid_limits=((0, z_top_m),
                 (storm_y_m - half_crop_m, storm_y_m + half_crop_m),
                 (storm_x_m - half_crop_m, storm_x_m + half_crop_m)),
    fields=['reflectivity', 'velocity_dealiased'], weighting_function='Barnes2',
)
vol_refl = np.ma.filled(grid.fields['reflectivity']['data'], -999)
vol_vel = np.ma.filled(grid.fields['velocity_dealiased']['data'], np.nan)
z_km = grid.z['data'] / 1000
y_km = grid.y['data'] / 1000
x_km = grid.x['data'] / 1000
print(f"grid shape: {vol_refl.shape}")

ZZ, YY, XX = np.meshgrid(z_km, y_km, x_km, indexing='ij')

# ---------------- build radial "flow" vectors: direction = radar->point unit vector, magnitude = radial velocity ----------------
Xs = XX[::CONE_STRIDE, ::CONE_STRIDE, ::CONE_STRIDE]
Ys = YY[::CONE_STRIDE, ::CONE_STRIDE, ::CONE_STRIDE]
Zs = ZZ[::CONE_STRIDE, ::CONE_STRIDE, ::CONE_STRIDE]
Vs = vol_vel[::CONE_STRIDE, ::CONE_STRIDE, ::CONE_STRIDE]
Rs = vol_refl[::CONE_STRIDE, ::CONE_STRIDE, ::CONE_STRIDE]

valid = np.isfinite(Vs) & (np.abs(Vs) >= VEL_MIN_ABS) & (Rs >= REFL_MIN_DBZ)
xs, ys, zs, vel = Xs[valid], Ys[valid], Zs[valid], Vs[valid]
print(f"arrow points (|v|>={VEL_MIN_ABS}m/s, refl>={REFL_MIN_DBZ}dBZ): {len(vel)}")

range_km = np.sqrt(xs ** 2 + ys ** 2 + (zs) ** 2)
range_km = np.where(range_km < 1e-3, 1e-3, range_km)
ux, uy, uz = xs / range_km, ys / range_km, zs / range_km
# NEXRAD convention: positive velocity = receding (away from radar) -> vector points along +radial direction
u, v, w = ux * vel, uy * vel, uz * vel

# go.Cone colors by VECTOR MAGNITUDE (always >=0), never by the signed scalar used to build u/v/w --
# so a single Cone trace can never distinguish approaching vs receding by color, only by (hard to see)
# arrow orientation. Split into two traces with distinct hue palettes so direction is visible at a glance.
mask_approach = vel < 0
mask_recede = vel > 0

nws_colors = [
    (1, 159, 244), (3, 0, 244), (2, 253, 2), (1, 197, 1), (0, 142, 0),
    (253, 248, 2), (229, 188, 0), (253, 149, 0), (253, 0, 0),
    (212, 0, 0), (188, 0, 0), (248, 0, 253), (152, 84, 198),
]
stops_refl = np.linspace(0, 1, len(nws_colors))
colorscale_refl = [[float(s), f"rgb{c}"] for s, c in zip(stops_refl, nws_colors)]

fig = go.Figure()

# faint reflectivity volume for storm-structure context
fig.add_trace(go.Volume(
    x=XX.flatten(), y=YY.flatten(), z=ZZ.flatten(), value=vol_refl.flatten(),
    isomin=REFL_MIN_DBZ, isomax=REFL_MAX_DBZ,
    opacity=1.0,
    opacityscale=[[0, 0], [0.2, 0.02], [0.45, 0.08], [0.7, 0.2], [0.9, 0.35], [1, 0.45]],
    surface_count=16,
    colorscale=colorscale_refl,
    caps=dict(x_show=False, y_show=False, z_show=False),
    showscale=False,
    name='反射率(背景)',
))

# radial velocity vectors, split by sign so color can actually show approach vs recede:
# blue palette = approaching radar (vel<0), red/orange palette = receding (vel>0)
fig.add_trace(go.Cone(
    x=xs[mask_approach], y=ys[mask_approach], z=zs[mask_approach],
    u=u[mask_approach], v=v[mask_approach], w=w[mask_approach],
    colorscale=[[0, '#0a2a55'], [1, '#1e9bff']],
    cmin=0, cmax=35,
    sizemode='scaled', sizeref=1.6,
    anchor='tail', showscale=False,
    name='朝向雷达 (approaching)',
))
fig.add_trace(go.Cone(
    x=xs[mask_recede], y=ys[mask_recede], z=zs[mask_recede],
    u=u[mask_recede], v=v[mask_recede], w=w[mask_recede],
    colorscale=[[0, '#552008'], [1, '#ff5a1e']],
    cmin=0, cmax=35,
    sizemode='scaled', sizeref=1.6,
    anchor='tail', showscale=False,
    name='远离雷达 (receding)',
))

# radar location marker for reference (radial vectors only make sense relative to this point)
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode='markers+text',
    marker=dict(size=6, color='#ffffff', symbol='diamond'),
    text=['KDGX雷达'], textposition='top center', textfont=dict(color='#d8e0e8', size=11),
    name='雷达位置',
))

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

html_fragment = fig.to_html(full_html=False, include_plotlyjs='inline', div_id='plot3d_flow')
with open(OUT_FRAGMENT, 'w', encoding='utf-8') as f:
    f.write(html_fragment)
print("fragment written:", OUT_FRAGMENT, "size:", len(html_fragment))

try:
    fig.update_layout(width=1000, height=800)
    fig.write_image(PREVIEW_PNG, scale=2)
    print("preview written:", PREVIEW_PNG)
except Exception as e:
    print("preview render skipped:", e)
