"""
雷达 + EC 融合短临预报 —— 深度学习方案 (0-6小时), 跟 radar_ec_blend.py 用同一个个例
(2026-07-10 密苏里东南部特大暴雨山洪, KLSX雷达 + ECMWF IFS Open Data)。

架构: ConvLSTM编码器(过去N_INPUT帧雷达降雨率, dB空间) + CNN编码EC背景场(0-3h/3-6h平均降雨率)
      -> 拼接融合 -> 两个解码头分别直接回归 +3h / +6h 的降雨率场(dB空间)
      预测目标跟 radar_ec_blend.py 完全一致的3个检验点(0/3/6h), CSI阈值也一致(1/10/25mm/h),
      方便跟传统融合方案直接对比。

*** 重要局限 ***
这里只有一个个例(一次真实EC起报), 训练样本靠在79帧雷达序列上滑窗生成, 每个样本的
"过去帧->未来帧"雷达自监督对是真实多样的, 但EC那两个背景场对所有样本都是同一份
(毕竟只下载了一次EC预报), 相当于告诉模型"大概这片区域这几小时会下多少雨"这个固定的
大尺度先验, 而不是每个样本各自对应的真实EC起报。也就是说这只是打通"雷达+EC双分支融合"
这个网络结构和训练流程的框架验证(overfit到这一个个例是预期行为), 要真正训练出有泛化能力
的模型, 需要重复 download_klsx_case.py + Herbie 下载流程, 积累多个独立个例(不同起报时次、
不同地点)才有意义。

*** 依赖 ***
先跑一次 prepare_dl_dataset.py 生成 dl_dataset_cache.npz。这里没有直接调用 case_data.py
去读雷达/下载EC(那样等于在本进程里同时导入 pyart/cfgrib/herbie 和 torch, 实测两者混在
一起偶发底层段错误, 具体机制没深究但拆开跑各自都稳定), 训练脚本只读缓存好的数组,
不导入 pyart/herbie, 从根源上避开这个冲突。
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pysteps.utils import transformation

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

DATASET_CACHE = "dl_dataset_cache.npz"
OUT_PREFIX = "dl_nowcast"

N_INPUT = 10             # input radar frames per sample
LEAD_HOURS = (3, 6)      # the two lead times the decoder heads predict
CSI_THRESHOLDS = [1.0, 10.0, 25.0]  # mm/h, same as radar_ec_blend.py
VAL_FRACTION = 0.2       # last 20% of samples (in time) held out -- see limitation note above
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# 1) load cached radar rain-rate stack + EC background rate fields
# ================================================================
print("=== loading cached dataset ===")
cache = np.load(DATASET_CACHE, allow_pickle=False)
radar_times = [datetime.fromisoformat(s) for s in cache["radar_times"]]
rainrate_stack = cache["rainrate_stack"]
ec_rate_0_3 = cache["ec_rate_0_3"]
ec_rate_3_6 = cache["ec_rate_3_6"]

R_db_all, _ = transformation.dB_transform(rainrate_stack, threshold=0.1, zerovalue=-15.0)
R_db_all = np.nan_to_num(R_db_all, nan=-15.0, posinf=-15.0, neginf=-15.0).astype(np.float32)

dt_min = np.median([(radar_times[i + 1] - radar_times[i]).total_seconds() / 60.0
                     for i in range(len(radar_times) - 1)])
print(f"{len(radar_times)} radar frames, dt~{dt_min:.1f}min")

# EC background channels, dB-transformed the same way so the network sees one consistent unit system
ec_db = np.stack([
    transformation.dB_transform(ec_rate_0_3, threshold=0.1, zerovalue=-15.0)[0],
    transformation.dB_transform(ec_rate_3_6, threshold=0.1, zerovalue=-15.0)[0],
]).astype(np.float32)
ec_db = np.nan_to_num(ec_db, nan=-15.0, posinf=-15.0, neginf=-15.0)

# ================================================================
# 2) build sliding-window samples: (N_INPUT past frames) -> (+3h frame, +6h frame)
# ================================================================
steps_3h = int(round(3 * 60 / dt_min))
steps_6h = int(round(6 * 60 / dt_min))

samples = []  # (input_idx_slice, target_idx_3h, target_idx_6h)
for t0_idx in range(N_INPUT - 1, len(radar_times)):
    idx3 = t0_idx + steps_3h
    idx6 = t0_idx + steps_6h
    if idx6 >= len(radar_times):
        break
    samples.append((t0_idx, idx3, idx6))
print(f"{len(samples)} sliding-window samples "
      f"(input {N_INPUT} frames, targets at +{steps_3h}/+{steps_6h} steps ~ +3h/+6h)")

n_val = max(1, int(len(samples) * VAL_FRACTION))
train_samples, val_samples = samples[:-n_val], samples[-n_val:]
print(f"train={len(train_samples)}  val={len(val_samples)}")


class NowcastDataset(torch.utils.data.Dataset):
    def __init__(self, sample_list):
        self.samples = sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        t0_idx, idx3, idx6 = self.samples[i]
        x = R_db_all[t0_idx - N_INPUT + 1: t0_idx + 1]      # (N_INPUT, H, W)
        y3 = R_db_all[idx3]
        y6 = R_db_all[idx6]
        return (torch.from_numpy(x).unsqueeze(1),            # (N_INPUT, 1, H, W)
                torch.from_numpy(ec_db),                      # (2, H, W)
                torch.from_numpy(y3).unsqueeze(0),
                torch.from_numpy(y6).unsqueeze(0))


# ================================================================
# 3) model: ConvLSTM encoder (radar) + CNN encoder (EC) -> fuse -> two regression heads
# ================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=pad)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class RadarECNet(nn.Module):
    def __init__(self, hid_ch=32):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.convlstm = ConvLSTMCell(16, hid_ch)
        self.ec_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, hid_ch, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.fuse = nn.Conv2d(hid_ch * 2, hid_ch, 1)
        self.head_3h = self._make_head(hid_ch)
        self.head_6h = self._make_head(hid_ch)

    @staticmethod
    def _make_head(hid_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(hid_ch, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x_seq, ec):
        # x_seq: (B, N_INPUT, 1, H, W), ec: (B, 2, H, W)
        b, t, _, h, w = x_seq.shape
        h_state = c_state = None
        for k in range(t):
            feat = self.down(x_seq[:, k])
            if h_state is None:
                h_state = torch.zeros(b, self.convlstm.hid_ch, *feat.shape[-2:], device=feat.device)
                c_state = torch.zeros_like(h_state)
            h_state, c_state = self.convlstm(feat, h_state, c_state)
        ec_feat = self.ec_encoder(ec)
        fused = self.fuse(torch.cat([h_state, ec_feat], dim=1))
        out3 = self.head_3h(fused)
        out6 = self.head_6h(fused)
        # crop/pad to exactly match input H,W (stride-4 down/up-sampling can round oddly for odd sizes)
        out3 = nn.functional.interpolate(out3, size=(h, w), mode='bilinear', align_corners=False)
        out6 = nn.functional.interpolate(out6, size=(h, w), mode='bilinear', align_corners=False)
        return out3, out6


# ================================================================
# 4) train (smoke test on this single case -- see limitation note in the module docstring)
# ================================================================
print(f"\n=== training on {DEVICE} ===")
train_loader = torch.utils.data.DataLoader(NowcastDataset(train_samples), batch_size=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(NowcastDataset(val_samples), batch_size=4, shuffle=False)

model = RadarECNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

N_EPOCHS = 60
for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0.0
    for x, ec, y3, y6 in train_loader:
        x, ec, y3, y6 = x.to(DEVICE), ec.to(DEVICE), y3.to(DEVICE), y6.to(DEVICE)
        opt.zero_grad()
        p3, p6 = model(x, ec)
        loss = loss_fn(p3, y3) + loss_fn(p6, y6)
        loss.backward()
        opt.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_samples)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, ec, y3, y6 in val_loader:
                x, ec, y3, y6 = x.to(DEVICE), ec.to(DEVICE), y3.to(DEVICE), y6.to(DEVICE)
                p3, p6 = model(x, ec)
                val_loss += (loss_fn(p3, y3) + loss_fn(p6, y6)).item() * x.size(0)
        val_loss /= max(1, len(val_samples))
        print(f"epoch {epoch + 1:3d}/{N_EPOCHS}  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")

# ================================================================
# 5) evaluate: CSI on the held-out validation samples, same thresholds as radar_ec_blend.py
# ================================================================
print("\n=== verification (held-out validation samples) ===")


def csi(forecast, obs, th):
    fc_bin = forecast > th
    obs_bin = obs > th
    hits = np.sum(fc_bin & obs_bin)
    misses = np.sum(~fc_bin & obs_bin)
    false_alarms = np.sum(fc_bin & ~obs_bin)
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else np.nan


model.eval()
csi_3h = {th: [] for th in CSI_THRESHOLDS}
csi_6h = {th: [] for th in CSI_THRESHOLDS}
last_batch_for_plot = None
with torch.no_grad():
    for x, ec, y3, y6 in val_loader:
        x, ec = x.to(DEVICE), ec.to(DEVICE)
        p3, p6 = model(x, ec)
        p3_rr = transformation.dB_transform(p3.cpu().numpy(), inverse=True, threshold=-10, zerovalue=-15.0)[0]
        p6_rr = transformation.dB_transform(p6.cpu().numpy(), inverse=True, threshold=-10, zerovalue=-15.0)[0]
        y3_rr = transformation.dB_transform(y3.numpy(), inverse=True, threshold=-10, zerovalue=-15.0)[0]
        y6_rr = transformation.dB_transform(y6.numpy(), inverse=True, threshold=-10, zerovalue=-15.0)[0]
        for th in CSI_THRESHOLDS:
            for b in range(p3_rr.shape[0]):
                csi_3h[th].append(csi(p3_rr[b, 0], y3_rr[b, 0], th))
                csi_6h[th].append(csi(p6_rr[b, 0], y6_rr[b, 0], th))
        last_batch_for_plot = (p3_rr, p6_rr, y3_rr, y6_rr)

print("--- 验证集 CSI 均值 ---")
for th in CSI_THRESHOLDS:
    print(f"{th:g}mm/h: +3h={np.nanmean(csi_3h[th]):.3f}  +6h={np.nanmean(csi_6h[th]):.3f}")

# ================================================================
# 6) qualitative snapshot: last validation sample, +3h and +6h, obs vs DL prediction
# ================================================================
p3_rr, p6_rr, y3_rr, y6_rr = last_batch_for_plot
levels = [0.1, 1, 2, 5, 10, 20, 40, 80, 150]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
panels = [
    (y3_rr[-1, 0], "实况观测 +3h", axes[0, 0]),
    (p3_rr[-1, 0], "深度学习预测 +3h", axes[0, 1]),
    (y6_rr[-1, 0], "实况观测 +6h", axes[1, 0]),
    (p6_rr[-1, 0], "深度学习预测 +6h", axes[1, 1]),
]
for field, title, ax in panels:
    # 'NWSRef' is a colormap pyart registers as an import side-effect; this script deliberately
    # avoids importing pyart (see module docstring re: the torch/pyart segfault), so use a
    # standard matplotlib colormap instead here.
    cf = ax.contourf(field, levels=levels, cmap='turbo', extend='max')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(cf, ax=axes, label="降雨率 (mm/h)", shrink=0.6, location='right')
fig.suptitle("深度学习(ConvLSTM+EC融合) 验证集样本示例\n(密苏里东南部特大暴雨个例, 2026-07-10 -- 单个例训练, 仅框架验证)")
plt.savefig(f"{OUT_PREFIX}_val_snapshot.png", dpi=160, bbox_inches='tight')
plt.close(fig)

print("\ndone:", f"{OUT_PREFIX}_val_snapshot.png")
