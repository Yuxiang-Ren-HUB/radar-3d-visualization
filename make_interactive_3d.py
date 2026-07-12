"""
交互式3D超级单体结构 (Plotly, 可360度旋转+缩放) —— Rolling Fork个例
输出一段HTML片段(不含<html>/<head>/<body>)，供外层页面嵌入
"""
import numpy as np
from skimage import measure
import plotly.graph_objects as go
import pyart

RADAR_FILE = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado\KDGX20230325_013047_V06"
OUT_FRAGMENT = r"F:\研\科研\Python\radar_analysis\plotly_fragment.html"

GRID_HALF_EXTENT_KM = 150
GRID_RES_KM = 0.5
Z_TOP_KM = 15
Z_RES_KM = 0.5
STORM_SEARCH_RADIUS_KM = 150
REFL_ISOSURFACE_DBZ = 35
CROP_KM = 40

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

verts, faces, normals, values = measure.marching_cubes(vol, level=REFL_ISOSURFACE_DBZ)
verts_x = x_km[xi0] + verts[:, 2] * GRID_RES_KM
verts_y = y_km[yi0] + verts[:, 1] * GRID_RES_KM
verts_z = verts[:, 0] * Z_RES_KM

fig = go.Figure(data=[go.Mesh3d(
    x=verts_x, y=verts_y, z=verts_z,
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    intensity=verts_z,
    colorscale=[[0, '#003b4d'], [0.3, '#00a3c4'], [0.55, '#00d9ff'],
                [0.75, '#ffb84d'], [1.0, '#ff8c3d']],
    colorbar=dict(title=dict(text="高度 (km)", font=dict(color='#d8e0e8')),
                   tickfont=dict(color='#d8e0e8'), len=0.7),
    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25, roughness=0.6),
    lightposition=dict(x=100, y=200, z=300),
    flatshading=False,
    name='35 dBZ 等值面',
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
)

html_fragment = fig.to_html(full_html=False, include_plotlyjs='inline', div_id='plot3d')
with open(OUT_FRAGMENT, 'w', encoding='utf-8') as f:
    f.write(html_fragment)

print(f"storm core at x={storm_x:.1f}km y={storm_y:.1f}km, vertices={len(verts_x)}, faces={len(faces)}")
print("fragment written:", OUT_FRAGMENT, "size:", len(html_fragment))
