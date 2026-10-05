"""
วาดรูปตัวอย่างถนนไทย: ภาพต้นฉบับ / เฉลย GT / โมเดลทำนาย (สามวิธีเทียบกัน)

ทำไมต้องมีไฟล์นี้: ผลถนนไทยทั้งหมดใน results/thai/ เป็นตัวเลขล้วน ยังไม่เคยเห็นว่า
โมเดลวาดเส้นออกมาหน้าตายังไง และการวาด overlay เดิมติดอยู่ในเซลล์โน้ตบุ๊ก เรียกใช้ซ้ำไม่ได้

ข้อบังคับที่ห้ามแก้ (ทั้งสามข้อเคยทำพังมาแล้ว):
  1. โหลดภาพผ่าน LaneDatasetFromCSV เท่านั้น ห้าม cv2.imread + cv2.resize เอง
     dataset ย่อด้วย Kornia และ binarize mask ที่ >127 ถ้าทางโหลดต่างจากตอนวัดผล
     รูปจะไม่ตรงกับตัวเลขใน REPORT_THAI.md
  2. ห้าม normalize — weights ทุกตัวในโปรเจกต์นี้กิน input ดิบ [0,1]
     finetune_thai.evaluate() ก็ป้อน tensor จาก DataLoader เข้าตรงๆ เหมือนกัน
  3. np.ascontiguousarray ก่อน cv2.line เสมอ — permute ของ torch ให้ view ที่ไม่ C-contiguous

ใช้:
  python3 plot_thai_examples.py
  python3 plot_thai_examples.py --n-per-cat 3 --per-category
"""
import argparse
import os
import sys

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append('.')
from dataset import LaneDatasetFromCSV
from finetune_thai import build_model          # เดาสถาปัตยกรรมจาก state_dict ไม่เชื่อชื่อไฟล์
from slope_metrics import extract_lanes, slope_metrics_for_pair, mask_iou

THAI_ROOT = "thai_road_lane"
OUT_DIR = "results/thai"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ใช้ checkpoint เกณฑ์ "มุมคลาด" ทั้งสองตัว ให้ตรงกับ REPORT_THAI.md หัวข้อ 3
# ถ้าสลับไปใช้ best_f1 กลางคืนจะกลายเป็น 26.47° แทน 4.28° รูปจะเล่าคนละเรื่องกับตารางในไฟล์เดียวกัน
MODELS = [
    ("zero-shot (CULane)",     "models/best_model_resnet50_ssl_bs16_ep30.pth"),
    ("fine-tune (best_slope)", "models/thai_finetune_ssl_best_slope.pth"),
    ("Thai-only (best_slope)", "models/thai_scratch_ssl_best_slope.pth"),
]
ENCODER = "resnet50"
THRESHOLD = 0.5
CATS = ["normal", "curve", "night"]

# สีตามแบบเดิมในโน้ตบุ๊ก (เซลล์ plot_slope_blindtest) อ่านออกแล้ว ไม่คิดใหม่
C_GT = np.array([0, 230, 0])
C_PRED = np.array([255, 40, 40])
C_BOTH = np.array([255, 230, 0])
LINE_GT = (0, 255, 120)
LINE_PRED = (255, 255, 255)


def pick_indices(ds, n_per_cat):
    """เลือก index หมวดละ n ด้วย stride คงที่ — รันซ้ำต้องได้รูปเดิม ห้ามสุ่ม"""
    bycat = {}
    for i, (_, c) in enumerate(ds.samples):
        bycat.setdefault(c, []).append(i)

    picks = []
    for c in CATS:
        idxs = bycat.get(c, [])
        if not idxs:
            print(f"⚠️  ไม่มีภาพหมวด {c}")
            continue
        n = min(n_per_cat, len(idxs))
        # กระจายทั่วหมวด ไม่ใช่หยิบ n ตัวแรก (ตัวแรกๆ มักมาจากคลิปเดียวกัน ภาพจะซ้ำกัน)
        step = len(idxs) / n
        picks += [(idxs[int(k * step)], c) for k in range(n)]
    return picks


def draw_panel(rgb, gt, pred=None):
    """ทับสีลงบนภาพ — pred=None คือช่องเฉลย แสดงแค่ GT"""
    vis = (rgb * 0.5).astype(np.uint8)
    if pred is None:
        vis[gt] = (0.35 * vis[gt] + 0.65 * C_GT).astype(np.uint8)
        lanes = extract_lanes(gt)
        for d in lanes:
            y0, y1 = int(d["y_top"]), int(d["y_bot"])
            cv2.line(vis, (int(d["a"] * y0 + d["b"]), y0),
                     (int(d["a"] * y1 + d["b"]), y1), LINE_GT, 2)
        return vis, lanes

    vis[gt] = (0.35 * vis[gt] + 0.65 * C_GT).astype(np.uint8)
    vis[pred] = (0.35 * vis[pred] + 0.65 * C_PRED).astype(np.uint8)
    both = gt & pred
    vis[both] = (0.30 * vis[both] + 0.70 * C_BOTH).astype(np.uint8)

    # extract_lanes ทิ้ง centerline ไปแล้ว เก็บแค่ a, b, y_top, y_bot
    # ปลายเส้นจึงต้องประกอบเองจาก x = a*y + b — เป็นทางเดียวที่ทำได้
    r = slope_metrics_for_pair(gt, pred)
    for lanes, color in ((r["gt_lanes"], LINE_GT), (r["pred_lanes"], LINE_PRED)):
        for d in lanes:
            y0, y1 = int(d["y_top"]), int(d["y_bot"])
            cv2.line(vis, (int(d["a"] * y0 + d["b"]), y0),
                     (int(d["a"] * y1 + d["b"]), y1), color, 2)
    return vis, r


@torch.inference_mode()
def predict_all(picks, ds):
    """โหลดโมเดลทีละตัว ทำนายเฉพาะเฟรมที่เลือก แล้วปล่อยทิ้ง (ไม่ถือ U-Net สามตัวพร้อมกัน)"""
    preds = {}
    for name, path in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ไม่พบ weights: {path}")
        print(f"\n▶ {name}")
        model = build_model(ENCODER, path)
        model.eval()
        out = []
        for idx, _ in picks:
            image_t, _ = ds[idx]
            # ป้อน input ดิบ [0,1] ไม่ normalize — ตรงกับ finetune_thai.evaluate()
            logit = model(image_t.unsqueeze(0).to(DEVICE))
            prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()
            out.append(prob > THRESHOLD)
        preds[name] = out
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return preds


def render(picks, ds, preds, out_path, title_extra=""):
    rows, cols = len(picks), 2 + len(MODELS)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.55 * rows))
    axes = np.atleast_2d(axes)

    stats = []
    for r, (idx, cat) in enumerate(picks):
        image_t, mask_t = ds[idx]
        filename = ds.samples[idx][0]
        rgb = np.ascontiguousarray((image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        gt = mask_t[0].numpy() > 0.5

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{cat} | input 448x800", fontsize=8)

        vis_gt, gt_lanes = draw_panel(rgb, gt)
        axes[r, 1].imshow(vis_gt)
        axes[r, 1].set_title(f"ground truth | {len(gt_lanes)} lanes", fontsize=8)

        for m, (name, _) in enumerate(MODELS):
            pred = preds[name][r]
            vis, res = draw_panel(rgb, gt, pred)
            iou = mask_iou(gt, pred)
            errs = [d["err"] for d in res["pairs"]]
            # median ของเฟรมเดียว = median "ภายในเฟรม" คนละตัวกับ slope_median_pooled ในตาราง
            # จึงต้องเขียน (this frame) กำกับ ห้ามเอาไปลบกับเลขในรายงานตรงๆ
            ang = f"{np.median(errs):.1f}d" if errs else "no match"
            axes[r, 2 + m].imshow(vis)
            axes[r, 2 + m].set_title(
                f"{name}\nIoU {iou:.2f} | angle {ang} (this frame) | match {res['n_matched']}/{res['n_gt']}",
                fontsize=8)
            stats.append(dict(cat=cat, file=filename, model=name, iou=iou,
                              err=float(np.median(errs)) if errs else float("nan"),
                              n_gt=res["n_gt"], n_matched=res["n_matched"]))

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        "Thai road test set — original | ground truth | three training methods" + title_extra
        + "\ngreen = GT only   red = prediction only   yellow = overlap   |   "
          "green line = GT direction   white line = predicted direction"
          "   |   column 1 is the resized 448x800 model input, not the raw camera frame",
        fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.2)
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\n📁 บันทึกภาพไว้ที่: {out_path}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=2, help="จำนวนภาพต่อหมวดถนน")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--per-category", action="store_true",
                    help="แยกไฟล์รูปตามหมวด แทนการรวมเป็นรูปเดียว")
    args = ap.parse_args()

    ds = LaneDatasetFromCSV(os.path.join(THAI_ROOT, f"{args.split}_list.csv"),
                            os.path.join(THAI_ROOT, "images"),
                            os.path.join(THAI_ROOT, "masks"),
                            augment=False)
    print(f"📦 {args.split}: {len(ds)} รูป {ds.category_counts()}")

    picks = pick_indices(ds, args.n_per_cat)
    print(f"🎯 เลือก {len(picks)} เฟรม (stride คงที่ รันซ้ำได้ชุดเดิม):")
    for idx, cat in picks:
        print(f"   [{idx:3d}] {cat:6s} {ds.samples[idx][0]}")

    preds = predict_all(picks, ds)

    if args.per_category:
        allstats = []
        for c in CATS:
            sub = [p for p in picks if p[1] == c]
            if not sub:
                continue
            subpreds = {n: [preds[n][picks.index(p)] for p in sub] for n, _ in MODELS}
            allstats += render(sub, ds, subpreds, f"{OUT_DIR}/examples_thai_{c}.png",
                               f"  ({c})")
        stats = allstats
    else:
        stats = render(picks, ds, preds, f"{OUT_DIR}/examples_thai_3methods.png")

    print("\n📊 สรุปเฟรมที่วาด (ไว้เทียบทิศทางกับ CSV ในรายงาน):")
    print(f"{'model':24s} {'cat':7s} {'IoU':>6s} {'angle':>7s} {'match/gt':>9s}")
    for s in stats:
        print(f"{s['model']:24s} {s['cat']:7s} {s['iou']:6.3f} {s['err']:7.2f} "
              f"{s['n_matched']:4d}/{s['n_gt']:<4d}")


if __name__ == "__main__":
    main()
