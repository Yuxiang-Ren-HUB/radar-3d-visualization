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
            print(f"{f}: FAILED ({e})")
            continue
        sweep = 0
        refl = np.ma.filled(radar.get_field(sweep, 'reflectivity'), -999)
        sl = radar.get_slice(sweep)
        x = radar.gate_x['data'][sl] / 1000
        y = radar.gate_y['data'][sl] / 1000
        rng = np.sqrt(x ** 2 + y ** 2)
        # exclude near-radar clutter (<8km) and anything beyond a sane storm-scale search (150km)
        valid = (rng > 8) & (rng < 150) & (refl > 45)
        if valid.any():
            r_at_max = rng[valid][np.argmax(refl[valid])]
        else:
            r_at_max = float('nan')
        print(f"{f.split(chr(92))[-1]}: core range ~{r_at_max:.1f} km")
