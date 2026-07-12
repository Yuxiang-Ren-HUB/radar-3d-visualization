"""
交互式3D超级单体结构 —— 体绘制(Volume Rendering)版本
比粒子散点更能保留连续结构（钩状回波、主体塔状对流），配色改用经典NWS雷达回波色标
(蓝->绿->黄->橙->红->紫)，透明度仍随dBZ增大而增大。
"""
import numpy as np
import plotly.graph_objects as go
import pyart

RADAR_FILE = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado\KDGX20230325_013047_V06"
OUT_FRAGMENT = r"F:\研\科研\Python\radar_analysis\plotly_fragment_volume.html"
PREVIEW_PNG = r"F:\研\科研\Python\radar_analysis\volume_3d_preview.png"

FULL_HALF_KM = 150       # pass 1: 全域低层搜索找风暴核心
SEARCH_RES_KM = 2.0
SEARCH_Z_KM = 2.0        # 搜索用的低层高度

CROP_KM = 45             # pass 2: 以风暴核心为中心裁剪出的绘制体积半径
GRID_RES_KM = 1.5
Z_TOP_KM = 15
Z_RES_KM = 0.75

REFL_MIN_DBZ = 10.0
REFL_MAX_DBZ = 70.0

radar = pyart.io.read_nexrad_archive(RADAR_FILE)

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
print(f"storm core (pass1): x={storm_x_m/1000:.1f}km y={storm_y_m/1000:.1f}km")

# ---------------- pass 2: fine-ish volume centered on the storm ----------------
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
    fields=['reflectivity'], weighting_function='Barnes2',
)
vol = np.ma.filled(grid.fields['reflectivity']['data'], -999)  # (nz, ny, nx)
z_km = grid.z['data'] / 1000
y_km = grid.y['data'] / 1000
x_km = grid.x['data'] / 1000
print(f"render volume shape: {vol.shape}  ({vol.size} cells)")

ZZ, YY, XX = np.meshgrid(z_km, y_km, x_km, indexing='ij')

# ---------------- classic NWS reflectivity color scale (10-70 dBZ, every 5 dBZ) ----------------
nws_colors = [
    (1, 159, 244), (3, 0, 244), (2, 253, 2), (1, 197, 1), (0, 142, 0),
    (253, 248, 2), (229, 188, 0), (253, 149, 0), (253, 0, 0),
    (212, 0, 0), (188, 0, 0), (248, 0, 253), (152, 84, 198),
]
stops = np.linspace(0, 1, len(nws_colors))
colorscale = [[float(s), f"rgb{c}"] for s, c in zip(stops, nws_colors)]

fig = go.Figure(data=go.Volume(
    x=XX.flatten(), y=YY.flatten(), z=ZZ.flatten(), value=vol.flatten(),
    isomin=REFL_MIN_DBZ, isomax=REFL_MAX_DBZ,
    opacity=1.0,
    opacityscale=[[0, 0], [0.15, 0.02], [0.35, 0.08], [0.55, 0.28], [0.75, 0.55], [0.9, 0.8], [1, 0.95]],
    surface_count=26,
    colorscale=colorscale,
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorbar=dict(title=dict(text="dBZ", font=dict(color='#d8e0e8')), tickfont=dict(color='#d8e0e8'), len=0.7),
    name='反射率体绘制',
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

html_fragment = fig.to_html(full_html=False, include_plotlyjs='inline', div_id='plot3d_volume')
with open(OUT_FRAGMENT, 'w', encoding='utf-8') as f:
    f.write(html_fragment)
print("fragment written:", OUT_FRAGMENT, "size:", len(html_fragment))

try:
    fig.update_layout(width=1000, height=800)
    fig.write_image(PREVIEW_PNG, scale=2)
    print("preview written:", PREVIEW_PNG)
except Exception as e:
    print("preview render skipped (kaleido not available?):", e)
