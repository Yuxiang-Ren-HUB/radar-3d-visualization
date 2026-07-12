"""
纯视觉风格：粒子沿"流线"流动的动画 —— 不是从雷达数据反演出的真实风场，
而是按照超级单体教科书概念模型手工构造的一个平滑矢量场(低层辐合入流 + 旋转 + 上升气流)，
纯粹用于视觉效果，让粒子看起来像是"被吸入并旋转抬升进风暴核心"。
背景叠加一层很淡的真实反射率体绘制，作为风暴结构参照。
"""
import numpy as np
import plotly.graph_objects as go
import pyart

RADAR_FILE = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado\KDGX20230325_013047_V06"
OUT_FRAGMENT = r"F:\研\科研\Python\radar_analysis\plotly_fragment_particleflow.html"
PREVIEW_PNG = r"F:\研\科研\Python\radar_analysis\particle_flow_preview.png"

CROP_KM = 45
GRID_RES_KM = 2.0
Z_TOP_KM = 15
Z_RES_KM = 1.0
REFL_MIN_DBZ = 10.0
REFL_MAX_DBZ = 70.0

N_PARTICLES = 700
N_FRAMES = 50
DT = 0.35           # integration step per frame (km-ish per unit field magnitude)
rng = np.random.default_rng(7)

# ---------------- get a single reflectivity volume snapshot for context (storm-centered coords) ----------------
radar = pyart.io.read_nexrad_archive(RADAR_FILE)

half_full_m, res_search_m = 150000, 2000
nxy_search = int(2 * half_full_m / res_search_m) + 1
grid_search = pyart.map.grid_from_radars(
    (radar,), grid_shape=(1, nxy_search, nxy_search),
    grid_limits=((1500, 2500), (-half_full_m, half_full_m), (-half_full_m, half_full_m)),
    fields=['reflectivity'], weighting_function='Barnes2',
)
refl_low = np.ma.filled(grid_search.fields['reflectivity']['data'][0], -999)
y_search, x_search = grid_search.y['data'], grid_search.x['data']
j, i = np.unravel_index(np.argmax(refl_low), refl_low.shape)
storm_x_m, storm_y_m = x_search[i], y_search[j]

half_crop_m, res_m = CROP_KM * 1000, GRID_RES_KM * 1000
z_top_m, z_res_m = Z_TOP_KM * 1000, Z_RES_KM * 1000
nxy = int(2 * half_crop_m / res_m) + 1
nz = int(z_top_m / z_res_m) + 1
grid = pyart.map.grid_from_radars(
    (radar,), grid_shape=(nz, nxy, nxy),
    grid_limits=((0, z_top_m), (storm_y_m - half_crop_m, storm_y_m + half_crop_m),
                 (storm_x_m - half_crop_m, storm_x_m + half_crop_m)),
    fields=['reflectivity'], weighting_function='Barnes2',
)
vol_refl = np.ma.filled(grid.fields['reflectivity']['data'], -999)
z_rel = grid.z['data'] / 1000
y_rel = (grid.y['data'] - storm_y_m) / 1000
x_rel = (grid.x['data'] - storm_x_m) / 1000
ZZ, YY, XX = np.meshgrid(z_rel, y_rel, x_rel, indexing='ij')
print("reflectivity backdrop ready, shape", vol_refl.shape)


# ---------------- synthetic supercell-inflow vector field (storm-centered coords, km) ----------------
def field(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.sqrt(x ** 2 + y ** 2) + 1e-6
    inflow_mag = 7.0 * np.exp(-r / 22.0) * np.exp(-np.clip(z - 1, 0, None) / 6.0)
    swirl_mag = inflow_mag * 0.85
    ux = -inflow_mag * (x / r) - swirl_mag * (-y / r)
    uy = -inflow_mag * (y / r) - swirl_mag * (x / r)
    updraft = 6.0 * np.exp(-r / 14.0) * (1 - np.exp(-z / 3.0)) * np.exp(-np.clip(z - 11, 0, None) / 3.0)
    uz = updraft
    return np.stack([ux, uy, uz], axis=1)


def respawn(n):
    """new particles enter at low levels near the outer edge, like fresh inflow air"""
    theta = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(CROP_KM * 0.55, CROP_KM * 0.95, n)
    z = rng.uniform(0.2, 2.5, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


particles = respawn(N_PARTICLES)
frames_pos = []
frames_speed = []
for t in range(N_FRAMES):
    v = field(particles)
    speed = np.linalg.norm(v, axis=1)
    frames_pos.append(particles.copy())
    frames_speed.append(speed.copy())
    particles = particles + v * DT
    out_of_bounds = (
        (np.sqrt(particles[:, 0] ** 2 + particles[:, 1] ** 2) > CROP_KM)
        | (particles[:, 2] > Z_TOP_KM - 0.5)
        | (particles[:, 2] < 0)
    )
    n_out = int(out_of_bounds.sum())
    if n_out > 0:
        particles[out_of_bounds] = respawn(n_out)
print(f"integrated {N_FRAMES} frames of {N_PARTICLES} particles")

# ---------------- build figure: faint reflectivity volume + animated particle scatter ----------------
nws_colors = [
    (1, 159, 244), (3, 0, 244), (2, 253, 2), (1, 197, 1), (0, 142, 0),
    (253, 248, 2), (229, 188, 0), (253, 149, 0), (253, 0, 0),
    (212, 0, 0), (188, 0, 0), (248, 0, 253), (152, 84, 198),
]
stops = np.linspace(0, 1, len(nws_colors))
colorscale_refl = [[float(s), f"rgb{c}"] for s, c in zip(stops, nws_colors)]

particle_colorscale = [[0, '#0a3d5c'], [0.4, '#00c2ff'], [0.75, '#a8ff5a'], [1, '#fff35a']]

static_particle_kwargs = dict(
    mode='markers',
    marker=dict(size=3.2, colorscale=particle_colorscale, cmin=0, cmax=8, showscale=False, opacity=0.9),
)

fig = go.Figure()
fig.add_trace(go.Volume(
    x=XX.flatten(), y=YY.flatten(), z=ZZ.flatten(), value=vol_refl.flatten(),
    isomin=REFL_MIN_DBZ, isomax=REFL_MAX_DBZ, opacity=1.0,
    opacityscale=[[0, 0], [0.2, 0.015], [0.45, 0.05], [0.7, 0.12], [0.9, 0.2], [1, 0.28]],
    surface_count=12, colorscale=colorscale_refl,
    caps=dict(x_show=False, y_show=False, z_show=False), showscale=False,
    name='反射率(背景)',
))
fig.add_trace(go.Scatter3d(
    x=frames_pos[0][:, 0], y=frames_pos[0][:, 1], z=frames_pos[0][:, 2],
    marker=dict(color=frames_speed[0], **static_particle_kwargs['marker']),
    mode='markers', name='气流粒子',
))

frames = [
    go.Frame(data=[go.Scatter3d(x=frames_pos[t][:, 0], y=frames_pos[t][:, 1], z=frames_pos[t][:, 2],
                                  marker=dict(color=frames_speed[t]))],
             traces=[1], name=str(t))
    for t in range(N_FRAMES)
]
fig.frames = frames

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        xaxis=dict(title='东西方向 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        yaxis=dict(title='南北方向 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        zaxis=dict(title='高度 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=760,
    showlegend=False,
    updatemenus=[dict(
        type='buttons', showactive=False, x=0.02, y=0.02, xanchor='left', yanchor='bottom',
        buttons=[
            dict(label='播放', method='animate',
                 args=[None, dict(frame=dict(duration=90, redraw=True), fromcurrent=False, transition=dict(duration=0))]),
            dict(label='暂停', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
        ],
    )],
    sliders=[dict(
        active=0, x=0.1, y=0.0, len=0.85,
        currentvalue=dict(prefix='帧: ', font=dict(color='#d8e0e8', size=12)),
        steps=[dict(method='animate', label=str(t),
                     args=[[str(t)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
               for t in range(N_FRAMES)],
    )],
)

html_fragment = fig.to_html(full_html=False, include_plotlyjs='inline', div_id='plot3d_particleflow')
with open(OUT_FRAGMENT, 'w', encoding='utf-8') as f:
    f.write(html_fragment)
print("fragment written:", OUT_FRAGMENT, "size:", len(html_fragment))

try:
    fig.update_layout(width=1000, height=800)
    fig.write_image(PREVIEW_PNG, scale=2)
    print("preview written:", PREVIEW_PNG)

    mid_t = N_FRAMES // 2
    fig.data[1].x = frames_pos[mid_t][:, 0]
    fig.data[1].y = frames_pos[mid_t][:, 1]
    fig.data[1].z = frames_pos[mid_t][:, 2]
    fig.data[1].marker.color = frames_speed[mid_t]
    fig.write_image(PREVIEW_PNG.replace(".png", "_midframe.png"), scale=2)
    print("mid-frame preview written")
except Exception as e:
    print("preview render skipped:", e)
