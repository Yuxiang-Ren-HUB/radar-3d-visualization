"""
下载 2026-07-10 密苏里东南部(Iron/Reynolds County)特大暴雨山洪个例的 KLSX(圣路易斯)
NEXRAD Level II 体扫数据, 数据源: Unidata THREDDS 实时归档镜像(公开可访问, S3桶本身禁止匿名
list, 但THREDDS这边有完整目录可以枚举+直接下载)。

事件参考: NWS圣路易斯办公室 2026-07-10 对 Iron/Reynolds County 发布"catastrophic flash
flood"(灾难级山洪), 累计雨量6-11英寸, 集中在约 05:44 CDT (~10:44 UTC) 前后。

下载窗口: 05:15-13:30 UTC。传统融合方案(radar_ec_blend.py)只需要 T0=06:00 UTC 前后±6h,
但深度学习方案要在雷达序列上滑窗生成多个训练样本(每个样本都要求N_INPUT帧历史+6h未来),
原来只到12:15的话滑窗后一个样本都凑不出来, 只需要往后延一点点就够用(不需要更多)。

(注: 一度尝试下载全天286个体扫, 中途 THREDDS 服务器(thredds-test, 看名字应该是个测试/
开发实例)明显被限速, 单文件下载速度从~10MB/s掉到~64KB/s(单文件103秒), 慢了100多倍,
下全天不现实。这里改成只多要十几个文件, 刚好够生成个位数~十几个滑窗样本做框架冒烟测试。)
"""
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

import requests

STATION = "KLSX"
DATE_STR = "20260710"
START_MIN = 5 * 60 + 15   # 05:15 UTC
END_MIN = 13 * 60 + 30    # 13:30 UTC

OUT_DIR = r"F:\WRF_backup\radar_demo\KLSX_20260710_flashflood"
CATALOG_URL = f"https://thredds-test.unidata.ucar.edu/thredds/catalog/nexrad/level2/{STATION}/{DATE_STR}/catalog.xml"
FILESERVER_BASE = f"https://thredds-test.unidata.ucar.edu/thredds/fileServer/nexrad/level2/{STATION}/{DATE_STR}/"
NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"

os.makedirs(OUT_DIR, exist_ok=True)

print("=== fetching THREDDS catalog ===")
with urllib.request.urlopen(CATALOG_URL) as resp:
    catalog_xml = resp.read()
root = ET.fromstring(catalog_xml)

pattern = re.compile(rf"Level2_{STATION}_{DATE_STR}_(\d{{2}})(\d{{2}})\.ar2v")
selected = []
for ds in root.iter(f"{NS}dataset"):
    name = ds.attrib.get("name", "")
    m = pattern.match(name)
    if not m:
        continue
    minute_of_day = int(m.group(1)) * 60 + int(m.group(2))
    if START_MIN <= minute_of_day <= END_MIN:
        selected.append(name)
selected.sort()
print(f"{len(selected)} volumes found in window {START_MIN // 60:02d}:{START_MIN % 60:02d}-"
      f"{END_MIN // 60:02d}:{END_MIN % 60:02d} UTC")

print("=== downloading ===")
session = requests.Session()
for name in selected:
    out_path = os.path.join(OUT_DIR, name)
    if os.path.exists(out_path):
        print(f"  skip (exists): {name}")
        continue
    for attempt in range(4):
        try:
            r = session.get(FILESERVER_BASE + name, timeout=180)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"  downloaded: {name} ({len(r.content) / 1024:.0f} KB)")
            break
        except (requests.exceptions.RequestException,) as e:
            print(f"  attempt {attempt + 1} failed for {name}: {e}")
            time.sleep(2)
    else:
        print(f"  GAVE UP on {name}")

print(f"\ndone. {len(selected)} volumes in {OUT_DIR}")
