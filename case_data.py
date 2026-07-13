"""
共享的雷达/EC数据预处理逻辑, 从 radar_ec_blend.py 中抽出来, 供传统融合方案和
深度学习方案共用, 避免重复实现雷达体扫处理 / EC降水场重网格这两块逻辑。
"""
import glob
import os
import re
from datetime import datetime

import numpy as np
import pyart
import xarray as xr
from scipy.interpolate import griddata
from herbie import Herbie

RADAR_DBZ_FLOOR = -30


def load_radar_rainrate_stack(radar_dir, grid_center_xy_km=(0, 0), half_extent_km=180,
                               res_km=2.0, level_km=1.5):
    """读取目录下所有 .ar2v 体扫, 输出低层(level_km附近)反射率转换的降雨率时间序列。

    返回: radar_times (datetime列表), rainrate_stack (ntimes,ny,nx) mm/h, lat2d, lon2d
    """
    files = sorted(glob.glob(os.path.join(radar_dir, "*.ar2v")))
    name_re = re.compile(r"Level2_\w+_(\d{8})_(\d{4})\.ar2v")
    radar_times = []
    for f in files:
        m = name_re.search(os.path.basename(f))
        radar_times.append(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))

    half_m = half_extent_km * 1000
    res_m = res_km * 1000
    nxy = int(2 * half_m / res_m) + 1
    cx_m, cy_m = grid_center_xy_km[0] * 1000, grid_center_xy_km[1] * 1000

    rainrate_stack = []
    lat2d = lon2d = None
    for f in files:
        radar = pyart.io.read_nexrad_archive(f)
        grid = pyart.map.grid_from_radars(
            (radar,), grid_shape=(3, nxy, nxy),
            grid_limits=(((level_km - 0.5) * 1000, (level_km + 0.5) * 1000),
                         (cy_m - half_m, cy_m + half_m), (cx_m - half_m, cx_m + half_m)),
            fields=['reflectivity'], weighting_function='Barnes2',
        )
        refl = np.ma.filled(grid.fields['reflectivity']['data'][1], RADAR_DBZ_FLOOR)
        if lat2d is None:
            lat2d = grid.point_latitude['data'][1]
            lon2d = grid.point_longitude['data'][1]
        Z_linear = 10 ** (refl / 10.0)
        rr = (Z_linear / 200.0) ** (1.0 / 1.6)  # Marshall-Palmer Z=200 R^1.6, mm/h
        rainrate_stack.append(np.clip(rr, 0, 300))

    return radar_times, np.array(rainrate_stack), lat2d, lon2d


def load_ec_cumulative_precip(ec_init, ec_fxx, lat2d, lon2d):
    """下载指定起报时次的 ECMWF IFS Open Data 累积降水(tp), 重网格到雷达网格上。

    返回: tp_on_radar_grid, shape (len(ec_fxx), ny, nx), 单位mm, 从起报时刻累积。
    """
    tp_on_radar_grid = []
    lat_min, lat_max = lat2d.min() - 0.5, lat2d.max() + 0.5
    lon_min, lon_max = lon2d.min() - 0.5, lon2d.max() + 0.5

    for fxx in ec_fxx:
        H = Herbie(ec_init, model='ifs', product='oper', fxx=fxx)
        path = H.download(":tp:")
        ds = xr.open_dataset(path, engine='cfgrib')
        tp_mm = ds['tp'].values * 1000.0  # m -> mm, accumulated since forecast start
        lon1d = ds['longitude'].values
        lat1d = ds['latitude'].values
        lon, lat = np.meshgrid(lon1d, lat1d)
        lon_adj = np.where(lon > 180, lon - 360, lon)

        mask = (lat >= lat_min) & (lat <= lat_max) & (lon_adj >= lon_min) & (lon_adj <= lon_max)
        pts = np.column_stack([lon_adj[mask], lat[mask]])
        vals = tp_mm[mask]
        interp = griddata(pts, vals, (lon2d, lat2d), method='linear')
        nn = griddata(pts, vals, (lon2d, lat2d), method='nearest')
        interp = np.where(np.isnan(interp), nn, interp)
        tp_on_radar_grid.append(interp)

    return np.array(tp_on_radar_grid)
