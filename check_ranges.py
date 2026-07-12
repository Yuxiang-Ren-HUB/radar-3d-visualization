import glob
import numpy as np
import pyart

for kind, folder in [
    ("Moore/KTLX", r"F:\WRF_backup\radar_demo\KTLX_20130520_Moore_tornado"),
    ("RollingFork/KDGX", r"F:\WRF_backup\radar_demo\KDGX_20230325_RollingFork_tornado"),
]:
    print(f"--- {kind} ---")
    for f in sorted(glob.glob(folder + r"\*V06*")):
        if f.endswith("_MDM"):
            continue
        try:
            radar = pyart.io.read_nexrad_archive(f)
        except Exception as e:
            print(f"{f}: FAILED to read ({e})")
            continue
        sweep = 0
        refl = radar.get_field(sweep, 'reflectivity')
        sl = radar.get_slice(sweep)
        x = radar.gate_x['data'][sl]
        y = radar.gate_y['data'][sl]
        rng = np.sqrt(x**2 + y**2) / 1000
        refl_filled = np.ma.filled(refl, -999)
        mask = refl_filled > 40
        if mask.any():
            core_range = rng[mask].min()
            core_range_max = rng[mask].max()
        else:
            core_range = core_range_max = float('nan')
        print(f"{f.split(chr(92))[-1]}: strong-echo(>40dBZ) range {core_range:.1f}-{core_range_max:.1f} km")
