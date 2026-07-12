"""
超级单体三维结构·体绘制 —— 随时间演变动画版
用Rolling Fork个例的14个体扫(00:54-02:12 UTC，约78分钟，~6分钟间隔，覆盖整个龙卷生命史)
每个体扫都做一次"低分辨率找风暴核心 -> 以核心为中心裁剪精细体积"，
所有帧都用相对风暴核心的坐标(即镜头跟随风暴移动)，这样能直接看到结构随时间的变化
(强度、塔高、核心形态)，而不必关心风暴在地图上的平移量。
"""
import glob
import os
import re
import numpy as np
import plotly.graph_objects as go
import pyart

RADAR_DIR = r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado"
OUT_FRAGMENT = r"F:\研\科研\Python\radar_analysis\plotly_fragment_volume_anim.html"

FULL_HALF_KM = 150
SEARCH_RES_KM = 2.0
SEARCH_Z_KM = 2.0
TRACK_SEARCH_RADIUS_KM = 25.0  # constrain core search near previous frame to avoid jumping to a different cell

CROP_KM = 45
GRID_RES_KM = 2.0    # 动画版稍微降分辨率，控制每帧数据量
Z_TOP_KM = 15
Z_RES_KM = 1.0

REFL_MIN_DBZ = 10.0
REFL_MAX_DBZ = 70.0

CACHE_NPZ = r"F:\研\科研\Python\radar_analysis\_anim_volume_cache.npz"

files = sorted(glob.glob(os.path.join(RADAR_DIR, "*_V06")))
files = [f for f in files if "MDM" not in f]
print(f"found {len(files)} volume scans")

half_full_m = FULL_HALF_KM * 1000
res_search_m = SEARCH_RES_KM * 1000
nxy_search = int(2 * half_full_m / res_search_m) + 1
z_lo_m = (SEARCH_Z_KM - 0.5) * 1000
z_hi_m = (SEARCH_Z_KM + 0.5) * 1000

half_crop_m = CROP_KM * 1000
res_m = GRID_RES_KM * 1000
z_top_m = Z_TOP_KM * 1000
z_res_m = Z_RES_KM * 1000
nxy = int(2 * half_crop_m / res_m) + 1
nz = int(z_top_m / z_res_m) + 1
print(f"per-frame grid: {nz}x{nxy}x{nxy} = {nz*nxy*nxy} cells")

# fixed relative coordinate axes (identical for every frame, since crop is always centered on that frame's storm core)
z_rel = np.linspace(0, Z_TOP_KM, nz)
y_rel = np.linspace(-CROP_KM, CROP_KM, nxy)
x_rel = np.linspace(-CROP_KM, CROP_KM, nxy)
ZZ, YY, XX = np.meshgrid(z_rel, y_rel, x_rel, indexing='ij')
X_flat, Y_flat, Z_flat = XX.flatten(), YY.flatten(), ZZ.flatten()

frame_vols = []
frame_labels = []
core_positions = []
prev_core_m = None  # (x_m, y_m) of previous frame's storm core, for constrained tracking

if os.path.exists(CACHE_NPZ):
    print("loading cached gridded volumes from", CACHE_NPZ)
    cache = np.load(CACHE_NPZ, allow_pickle=True)
    frame_vols = list(cache["frame_vols"])  # 2D float array -> list of 1D float arrays, one per frame
    frame_labels = list(cache["frame_labels"])
    core_positions = [tuple(p) for p in cache["core_positions"]]
    files = []  # skip the processing loop below

for fpath in files:
    m = re.search(r"_(\d{6})_V06", os.path.basename(fpath))
    hhmmss = m.group(1)
    label = f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]} UTC"

    radar = pyart.io.read_nexrad_archive(fpath)

    grid_search = pyart.map.grid_from_radars(
        (radar,), grid_shape=(1, nxy_search, nxy_search),
        grid_limits=((z_lo_m, z_hi_m), (-half_full_m, half_full_m), (-half_full_m, half_full_m)),
        fields=['reflectivity'], weighting_function='Barnes2',
    )
    refl_low = np.ma.filled(grid_search.fields['reflectivity']['data'][0], -999)
    y_search = grid_search.y['data']
    x_search = grid_search.x['data']

    if prev_core_m is not None:
        xs_grid, ys_grid = np.meshgrid(x_search, y_search)
        dist_m = np.sqrt((xs_grid - prev_core_m[0]) ** 2 + (ys_grid - prev_core_m[1]) ** 2)
        near_mask = dist_m <= TRACK_SEARCH_RADIUS_KM * 1000
        refl_constrained = np.where(near_mask, refl_low, -999)
        if np.nanmax(refl_constrained) > -999:
            j, i = np.unravel_index(np.argmax(refl_constrained), refl_constrained.shape)
        else:
            j, i = np.unravel_index(np.argmax(refl_low), refl_low.shape)  # fallback: lost track, re-search globally
    else:
        j, i = np.unravel_index(np.argmax(refl_low), refl_low.shape)  # first frame: global search

    storm_x_m, storm_y_m = x_search[i], y_search[j]
    prev_core_m = (storm_x_m, storm_y_m)
    core_positions.append((storm_x_m / 1000, storm_y_m / 1000))

    grid = pyart.map.grid_from_radars(
        (radar,), grid_shape=(nz, nxy, nxy),
        grid_limits=((0, z_top_m),
                     (storm_y_m - half_crop_m, storm_y_m + half_crop_m),
                     (storm_x_m - half_crop_m, storm_x_m + half_crop_m)),
        fields=['reflectivity'], weighting_function='Barnes2',
    )
    vol = np.ma.filled(grid.fields['reflectivity']['data'], -999)
    frame_vols.append(vol.flatten())
    frame_labels.append(label)
    print(f"processed {os.path.basename(fpath)} -> core=({storm_x_m/1000:.1f},{storm_y_m/1000:.1f})km label={label}")

if not os.path.exists(CACHE_NPZ):
    np.savez_compressed(CACHE_NPZ, frame_vols=np.stack(frame_vols).astype(np.float32),
                        frame_labels=np.array(frame_labels), core_positions=np.array(core_positions))
    print("cached gridded volumes to", CACHE_NPZ)

# the Artifact viewer timed out loading the 14-frame/surface_count=20 version (too much client-side
# WebGL geometry to build at once) -- thin frames and simplify surfaces to cut render cost
FRAME_STRIDE = 2
frame_vols = frame_vols[::FRAME_STRIDE]
frame_labels = frame_labels[::FRAME_STRIDE]
core_positions = core_positions[::FRAME_STRIDE]
print(f"thinned to {len(frame_vols)} frames (stride={FRAME_STRIDE})")

nws_colors = [
    (1, 159, 244), (3, 0, 244), (2, 253, 2), (1, 197, 1), (0, 142, 0),
    (253, 248, 2), (229, 188, 0), (253, 149, 0), (253, 0, 0),
    (212, 0, 0), (188, 0, 0), (248, 0, 253), (152, 84, 198),
]
stops = np.linspace(0, 1, len(nws_colors))
colorscale = [[float(s), f"rgb{c}"] for s, c in zip(stops, nws_colors)]

# round to integer dBZ: shrinks embedded JSON substantially with no meaningful visual loss
frame_vols = [np.round(v).astype(np.int16) for v in frame_vols]

static_kwargs = dict(
    x=X_flat, y=Y_flat, z=Z_flat,
    isomin=REFL_MIN_DBZ, isomax=REFL_MAX_DBZ,
    opacity=1.0,
    opacityscale=[[0, 0], [0.15, 0.02], [0.35, 0.08], [0.55, 0.28], [0.75, 0.55], [0.9, 0.8], [1, 0.95]],
    surface_count=10,
    colorscale=colorscale,
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorbar=dict(title=dict(text="dBZ", font=dict(color='#d8e0e8')), tickfont=dict(color='#d8e0e8'), len=0.7),
)

# frames only carry the changing 'value' array -- x/y/z/colorscale/etc. are inherited from the
# initial trace by Plotly.js, so repeating them per frame would bloat the embedded JSON ~14x for nothing
frames = [
    go.Frame(data=[go.Volume(value=frame_vols[t])], name=str(t))
    for t in range(len(frame_vols))
]

fig = go.Figure(
    data=[go.Volume(value=frame_vols[0], **static_kwargs)],
    frames=frames,
)

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        xaxis=dict(title='相对东西方向 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        yaxis=dict(title='相对南北方向 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        zaxis=dict(title='高度 (km)', backgroundcolor='#0a0e14', gridcolor='#232b35', color='#8a94a3'),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.6),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    height=760,
    showlegend=False,
    title=dict(text=frame_labels[0], font=dict(color='#d8e0e8', size=14), x=0.02),
    updatemenus=[dict(
        type='buttons', showactive=False, x=0.02, y=0.02, xanchor='left', yanchor='bottom',
        buttons=[
            dict(label='播放', method='animate',
                 args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label='暂停', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
        ],
    )],
    sliders=[dict(
        active=0, x=0.1, y=0.0, len=0.85,
        currentvalue=dict(prefix='体扫时刻: ', font=dict(color='#d8e0e8', size=12)),
        steps=[dict(method='animate', label=frame_labels[t],
                     args=[[str(t)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
               for t in range(len(frame_vols))],
    )],
)

html_fragment = fig.to_html(full_html=False, include_plotlyjs='inline', div_id='plot3d_volume_anim')
with open(OUT_FRAGMENT, 'w', encoding='utf-8') as f:
    f.write(html_fragment)

print("\ncore track (km, relative to radar):")
for lbl, (cx, cy) in zip(frame_labels, core_positions):
    print(f"  {lbl}: x={cx:.1f} y={cy:.1f}")
print("\nfragment written:", OUT_FRAGMENT, "size:", len(html_fragment))
