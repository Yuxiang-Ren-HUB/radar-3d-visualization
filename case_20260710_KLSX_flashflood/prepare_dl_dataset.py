"""
把雷达体扫+EC预报预处理成深度学习训练要用的数组, 缓存成一个 .npz 文件。

单独拆出这一步(不在 dl_nowcast.py 里内联做), 是因为 pyart/cfgrib/herbie 这套原生库
和 torch(尤其是CUDA初始化后)在同一个进程里混用时会偶发底层段错误(段错误发生在
Herbie下载/cfgrib读取那一段, 具体原因没深究, 但两次单独测试都各自正常, 混一起跑
就会崩), 干脆让"数据准备"和"模型训练"分成两个进程, 训练脚本只加载缓存的.npz,
完全不导入 pyart/herbie, 从根源上避开这个冲突。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # case_data.py lives one level up (shared across cases)
from case_data import load_radar_rainrate_stack, load_ec_cumulative_precip

RADAR_DIR = r"F:\WRF_backup\radar_demo\KLSX_20260710_flashflood"
EC_INIT = "2026-07-10 00:00"
EC_FXX = [6, 9, 12]
GRID_CENTER_XY_KM = (0, 0)
HALF_EXTENT_KM = 180
RES_KM = 2.0
LEVEL_KM = 1.5
OUT_NPZ = "dl_dataset_cache.npz"

print("=== gridding radar volumes ===")
radar_times, rainrate_stack, lat2d, lon2d = load_radar_rainrate_stack(
    RADAR_DIR, GRID_CENTER_XY_KM, HALF_EXTENT_KM, RES_KM, LEVEL_KM)
print("radar_rainrate_stack:", rainrate_stack.shape)

print("=== downloading + regridding EC precip ===")
tp_on_radar_grid = load_ec_cumulative_precip(EC_INIT, EC_FXX, lat2d, lon2d)
ec_rate_0_3 = tp_on_radar_grid[1] / 3.0
ec_rate_3_6 = (tp_on_radar_grid[2] - tp_on_radar_grid[1]) / 3.0

radar_times_str = np.array([t.isoformat() for t in radar_times])
np.savez(OUT_NPZ,
         radar_times=radar_times_str,
         rainrate_stack=rainrate_stack.astype(np.float32),
         ec_rate_0_3=ec_rate_0_3.astype(np.float32),
         ec_rate_3_6=ec_rate_3_6.astype(np.float32),
         lat2d=lat2d, lon2d=lon2d)
print(f"\nsaved -> {OUT_NPZ}")
