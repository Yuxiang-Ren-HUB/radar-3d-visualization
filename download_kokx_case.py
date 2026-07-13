"""
下载 2023-03-13/14 东北沿岸低压(Nor'easter)个例的 KOKX(纽约Upton)NEXRAD Level II 体扫数据。
这是慢速移动、结构稳定的大范围层状云降水个例，跟Rolling Fork(快速移动、会重组的强对流)形成对比，
用来检验"STEPS集合融合确实更好"这个结论是不是个例特定的。

用nexradaws查scan列表(桶本身禁止匿名list，但nexradaws内部走的是别的索引方式)，
下载改用boto3直接GetObject(nexradaws自带的下载方法测试时卡死不动，原因不明，
换成自己写的直接S3下载更可控、能看到实时进度)。
"""
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import nexradaws

STATION = "KOKX"
YEAR, MONTH, DAY = "2023", "03", "14"
START_MIN = 0
END_MIN = 7 * 60 + 10  # 07:10 UTC

OUT_DIR = r"F:\WRF_backup\radar_demo\KOKX_20230314_noreaster"
os.makedirs(OUT_DIR, exist_ok=True)

conn = nexradaws.NexradAwsInterface()
print("=== querying available scans ===", flush=True)
scans = conn.get_avail_scans(YEAR, MONTH, DAY, STATION)
print(f"found {len(scans)} scans total for {STATION} {YEAR}-{MONTH}-{DAY}", flush=True)

selected = []
for s in scans:
    if s.key.endswith("_MDM"):
        continue
    fname = s.filename
    hhmmss = fname.split("_")[1]
    minute_of_day = int(hhmmss[0:2]) * 60 + int(hhmmss[2:4])
    if START_MIN <= minute_of_day <= END_MIN:
        selected.append(s)
print(f"{len(selected)} scans in window {START_MIN//60:02d}:{START_MIN%60:02d}-{END_MIN//60:02d}:{END_MIN%60:02d} UTC", flush=True)

s3 = boto3.client('s3', config=Config(
    signature_version=UNSIGNED,
    connect_timeout=30, read_timeout=180,
    retries={'max_attempts': 6, 'mode': 'standard'},
))
print("=== downloading via boto3 ===", flush=True)
ok, failed = 0, 0
for i, s in enumerate(selected):
    out_path = os.path.join(OUT_DIR, s.filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        ok += 1
        continue
    # download_file() calls HeadObject first, which this bucket forbids for anonymous access
    # even though GetObject works -- fetch the body directly instead. Network in this environment
    # is slow/flaky (seen before with GFS downloads), so retry a few times before giving up.
    for attempt in range(4):
        try:
            resp = s3.get_object(Bucket='unidata-nexrad-level2', Key=s.key)
            with open(out_path, 'wb') as f:
                f.write(resp['Body'].read())
            ok += 1
            break
        except Exception as e:
            if attempt == 3:
                failed += 1
                print(f"  FAILED {s.filename} after 4 attempts: {e}", flush=True)
            else:
                print(f"  retry {attempt + 1} for {s.filename}: {e}", flush=True)
    if (i + 1) % 10 == 0 or (i + 1) == len(selected):
        print(f"  progress: {i + 1}/{len(selected)} (ok={ok} failed={failed})", flush=True)

print(f"\ndone. ok={ok} failed={failed}", flush=True)
