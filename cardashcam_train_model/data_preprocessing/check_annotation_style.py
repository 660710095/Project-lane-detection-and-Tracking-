"""
เช็คว่า mask เฉลยของไทยวาดแบบเดียวกับ CULane หรือเปล่า

ทำไมต้องเช็คก่อนไปเก็บข้อมูลเพิ่ม: ถ้า convention การ label ต่างกัน เก็บภาพเพิ่มอีกเท่าไหร่ก็ไม่ช่วย
เพราะโมเดลเรียนจาก CULane มาแล้วถูกสอนอีกแบบ — เป็นการเช็ค ~2 นาทีที่กันการ label เสียเปล่าเป็นสัปดาห์

วัดเฉพาะ **mask เฉลย** ไม่เกี่ยวกับโมเดลเลย ใช้ extract_lanes ตัวเดียวกับที่ใช้วัดผล
โหลดผ่าน LaneDatasetFromCSV เพื่อให้ได้ขนาดและการ binarize เหมือนตอนเทรนเป๊ะ

ใช้: python3 check_annotation_style.py [--n-per-cat 200]
"""
import argparse
import os
import sys

import cv2
cv2.setNumThreads(0)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append('.')
from dataset import LaneDatasetFromCSV, IMG_SIZE
from slope_metrics import extract_lanes

CULANE_ROOT = "/home/jovyan/shared/donut/CarDashCam/CULane"
THAI_ROOT = "thai_road_lane"
CATS = ("normal", "curve", "night")
OUT_DIR = "results/thai"
H, W = IMG_SIZE
COLORS = {"CULane": "#2a78d6", "Thai": "#eb6834"}


def sample(root, split, cat, n):
    """หยิบ index หมวดนั้นด้วย stride คงที่ — รันซ้ำได้ชุดเดิม"""
    ds = LaneDatasetFromCSV(os.path.join(root, f"{split}_list.csv"),
                            os.path.join(root, "images"), os.path.join(root, "masks"),
                            augment=False)
    idx = [i for i, (_, c) in enumerate(ds.samples) if c == cat]
    if not idx:
        return ds, []
    step = max(len(idx) // n, 1)
    return ds, idx[::step][:n]


def measure(ds, picks):
    """คืน dict ของสถิติต่อเส้นและต่อภาพ จาก mask เฉลยล้วนๆ"""
    per_lane = dict(nrows=[], area=[], w=[], y_top=[], y_bot=[], rms=[], ang=[])
    per_img = dict(n_lanes=[], mask_frac=[])
    for i in picks:
        _, mask_t = ds[i]
        gt = mask_t[0].numpy() > 0.5
        per_img["mask_frac"].append(gt.mean())
        lanes = extract_lanes(gt)
        per_img["n_lanes"].append(len(lanes))
        for d in lanes:
            per_lane["nrows"].append(d["nrows"])
            per_lane["area"].append(d["area"])
            per_lane["w"].append(d["area"] / max(d["nrows"], 1))
            per_lane["y_top"].append(d["y_top"] / H)
            per_lane["y_bot"].append(d["y_bot"] / H)
            per_lane["rms"].append(d["rms"])
            per_lane["ang"].append(d["ang"])
    return {k: np.array(v) for k, v in {**per_lane, **per_img}.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=200)
    args = ap.parse_args()

    stats = {}
    for name, root, split in (("CULane", CULANE_ROOT, "train"), ("Thai", THAI_ROOT, "train")):
        for cat in CATS:
            ds, picks = sample(root, split, cat, args.n_per_cat)
            if not picks:
                print(f"⚠️  {name} {cat}: ไม่มีภาพ")
                continue
            stats[(name, cat)] = measure(ds, picks)
            print(f"   {name:7s} {cat:7s} {len(picks):4d} ภาพ "
                  f"{len(stats[(name,cat)]['nrows']):5d} เส้น")

    KEYS = [("n_lanes", "เส้นต่อภาพ", 2), ("w", "ความหนาเส้น (px)", 1),
            ("nrows", "ความยาวเส้น (แถว)", 0), ("y_top", "ปลายบน (สัดส่วนความสูง)", 3),
            ("y_bot", "ปลายล่าง (สัดส่วนความสูง)", 3), ("rms", "ความคดของเส้น (px)", 2),
            ("mask_frac", "สัดส่วนพิกเซลที่เป็นเลน", 4)]

    print("\n" + "=" * 92)
    print("median ของแต่ละค่า — เทียบ CULane กับ ไทย ในหมวดเดียวกัน")
    print(f"{'ค่าที่วัด':>26s}" + "".join(f"{c:>21s}" for c in CATS))
    print(" " * 26 + "".join(f"{'CULane':>10s}{'Thai':>11s}" for _ in CATS))
    flags = []
    for k, label, dec in KEYS:
        line = f"{label:>26s}"
        for cat in CATS:
            a = stats.get(("CULane", cat)), stats.get(("Thai", cat))
            if not all(a):
                line += f"{'—':>10s}{'—':>11s}"
                continue
            ma, mb = np.median(a[0][k]), np.median(a[1][k])
            line += f"{ma:>10.{dec}f}{mb:>11.{dec}f}"
            rel = abs(mb - ma) / max(abs(ma), 1e-9)
            if rel > 0.25:
                flags.append((label, cat, ma, mb, rel))
        print(line)

    print("\n" + "=" * 92)
    if flags:
        print("⚠️  ค่าที่ต่างกันเกิน 25% — ตรงนี้คือจุดที่ convention อาจไม่ตรงกัน")
        for label, cat, ma, mb, rel in flags:
            direction = "มากกว่า" if mb > ma else "น้อยกว่า"
            print(f"   {label} ({cat}): ไทย {mb:.3f} {direction} CULane {ma:.3f} — ต่าง {rel:.0%}")
    else:
        print("✓ ทุกค่าต่างกันไม่เกิน 25% — convention การ label ถือว่าใกล้เคียงกัน")

    # ---------- รูป: การกระจาย ไม่ใช่แค่ median ----------
    plot_keys = [k for k, _, _ in KEYS if k != "mask_frac"]
    fig, axes = plt.subplots(len(CATS), len(plot_keys),
                             figsize=(3.0 * len(plot_keys), 2.7 * len(CATS)))
    en = {"n_lanes": "lanes per image", "w": "lane thickness (px)",
          "nrows": "lane length (rows)", "y_top": "top end (frac of height)",
          "y_bot": "bottom end (frac of height)", "rms": "straight-line rms (px)"}
    for r, cat in enumerate(CATS):
        for c, k in enumerate(plot_keys):
            ax = axes[r, c]
            for name in ("CULane", "Thai"):
                s = stats.get((name, cat))
                if s is None:
                    continue
                v = s[k]
                hi = np.percentile(np.concatenate([stats[(n, cat)][k] for n in ("CULane", "Thai")
                                                   if (n, cat) in stats]), 99)
                ax.hist(v, bins=30, range=(0, max(hi, 1e-6)), density=True, alpha=0.55,
                        color=COLORS[name], label=name)
            if r == 0:
                ax.set_title(en[k], fontsize=9)
            if c == 0:
                ax.set_ylabel(cat, fontsize=10)
            ax.set_yticks([])
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Annotation style — ground-truth masks only, no model involved\n"
                 "if these distributions differ, more Thai labels will not close the gap",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/annotation_style.png"
    plt.savefig(out, dpi=115)
    plt.close(fig)
    print(f"\n📁 {out}")

    import csv
    with open(f"{OUT_DIR}/annotation_style.csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["dataset", "category", "metric", "median", "p25", "p75", "n"])
        for (name, cat), s in stats.items():
            for k, _, _ in KEYS:
                v = s[k]
                w_.writerow([name, cat, k, np.median(v), *np.percentile(v, [25, 75]), len(v)])
    print(f"📁 {OUT_DIR}/annotation_style.csv")


if __name__ == "__main__":
    main()
