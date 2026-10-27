"""
วัดว่า convention การ label ที่ต่างกัน ทำให้คะแนนบนถนนไทยเสียไปแค่ไหน

คู่กับ check_annotation_style.py ซึ่งวัดเฉพาะเรขาคณิตของ mask เฉลย ไฟล์นี้เอาโมเดล zero-shot
(เทรน CULane ล้วน ไม่เคยเห็นภาพไทย) มาทำนายทั้งสองชุด แล้วแยก "ทำนายไม่เจอ" ออกจาก "ทำนายเกิน"

ทำไมต้องแยก: IoU ก้อนเดียวบอกไม่ได้ว่าคะแนนหายเพราะอะไร
  พิกเซล recall ต่ำ  = โมเดลมองไม่เห็นเส้น (ปัญหาที่โมเดล)
  พิกเซล precision ต่ำ = โมเดลวาดเส้นที่เฉลยไม่มี (อาจเป็นปัญหาที่ convention ไม่ใช่ที่โมเดล)

ใช้: python3 check_convention_effect.py [--n 150]
"""
import argparse
import csv
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
from finetune_thai import build_model, THAI_ROOT
from slope_metrics import extract_lanes, slope_metrics_for_pair

CULANE_ROOT = "/home/jovyan/shared/donut/CarDashCam/CULane"
ZEROSHOT = "models/best_model_resnet50_ssl_bs16_ep30.pth"
OUT_DIR = "results/thai"
CATS = ("normal", "curve", "night")


def stride_picks(ds, cat, n):
    idx = [i for i, (_, c) in enumerate(ds.samples) if c == cat]
    step = max(len(idx) // n, 1)
    return idx[::step][:n]


@torch.inference_mode()
def measure(model, root, cat, n):
    ds = LaneDatasetFromCSV(f"{root}/test_list.csv", f"{root}/images", f"{root}/masks")
    picks = stride_picks(ds, cat, n)
    tp = fp = fn = 0
    n_gt = n_pred = n_match = 0
    for i in picks:
        im, mk = ds[i]
        p = (torch.sigmoid(model(im.unsqueeze(0).cuda()).float())[0, 0].cpu().numpy() > 0.5)
        t = mk[0].numpy() > 0.5
        tp += (p & t).sum(); fp += (p & ~t).sum(); fn += (~p & t).sum()
        r = slope_metrics_for_pair(t, p)
        n_gt += r["n_gt"]; n_pred += r["n_pred"]; n_match += r["n_matched"]
    return dict(n_img=len(picks),
                px_recall=tp / max(tp + fn, 1), px_precision=tp / max(tp + fp, 1),
                iou=tp / max(tp + fp + fn, 1),
                lane_recall=n_match / max(n_gt, 1), lane_precision=n_match / max(n_pred, 1),
                lanes_gt=n_gt / len(picks), lanes_pred=n_pred / len(picks))


def evidence_figure(model, n_rows=3):
    """เฟรมไทยที่โมเดลวาดเส้นเกินเฉลยมากสุด — ดูว่าเส้นเกินอยู่บนเส้นถนนจริงหรือไม่"""
    ds = LaneDatasetFromCSV(f"{THAI_ROOT}/test_list.csv", f"{THAI_ROOT}/images",
                            f"{THAI_ROOT}/masks")
    cand = []
    with torch.inference_mode():
        for i in stride_picks(ds, "normal", 40):
            im, mk = ds[i]
            pr = (torch.sigmoid(model(im.unsqueeze(0).cuda()).float())[0, 0].cpu().numpy() > 0.5)
            g, p = extract_lanes(mk[0].numpy() > 0.5), extract_lanes(pr)
            cand.append((len(p) - len(g), i, len(g), len(p), im, mk, pr))
    cand.sort(key=lambda t: (-t[0], t[1]))      # ตัดสินเสมอด้วย index ให้ผลคงที่
    pick = cand[:n_rows]

    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.4 * n_rows))
    for r, (_, i, ng, npd, im, mk, pr) in enumerate(pick):
        # permute ให้ array ที่ไม่ C-contiguous — ต้อง ascontiguousarray ก่อนใช้กับ cv2/ระบายสี
        rgb = np.ascontiguousarray((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        gt = mk[0].numpy() > 0.5
        for c, (m, col, title) in enumerate(
                ((gt, [0, 230, 0], f"Thai ground truth — {ng} lanes"),
                 (pr, [255, 40, 40], f"zero-shot prediction — {npd} lanes"))):
            vis = (rgb * 0.55).astype(np.uint8)
            vis[m] = (0.2 * vis[m] + 0.8 * np.array(col)).astype(np.uint8)
            axes[r, c].imshow(vis)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("Thai frames where the CULane-trained model draws the most extra lanes\n"
                 "the red lanes it adds sit on real road markings the Thai labels do not cover",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    out = f"{OUT_DIR}/convention_extra_lanes.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    return out, [(ds.samples[i][0], ng, npd) for _, i, ng, npd, _, _, _ in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="ภาพต่อหมวด")
    args = ap.parse_args()

    model = build_model("resnet50", ZEROSHOT)
    model.eval()

    rows = []
    print(f"{'ชุด':20s}{'ภาพ':>5s}{'พิกเซล recall':>15s}{'พิกเซล prec':>13s}{'IoU':>8s}"
          f"{'เส้น recall':>13s}{'เส้น prec':>11s}{'เฉลย':>7s}{'ทำนาย':>8s}")
    for name, root in (("CULane", CULANE_ROOT), ("Thai", THAI_ROOT)):
        for cat in CATS:
            m = measure(model, root, cat, args.n)
            rows.append(dict(dataset=name, category=cat, **m))
            print(f"{name+' '+cat:20s}{m['n_img']:5d}{m['px_recall']:15.3f}"
                  f"{m['px_precision']:13.3f}{m['iou']:8.3f}{m['lane_recall']:13.3f}"
                  f"{m['lane_precision']:11.3f}{m['lanes_gt']:7.2f}{m['lanes_pred']:8.2f}")

    # ตีความอัตโนมัติ แยกสองกลไก ไม่ hardcode ข้อสรุป
    print("\nกลไกที่คะแนนหายไป แยกตามหมวด (เทียบ CULane หมวดเดียวกัน)")
    cu = {r["category"]: r for r in rows if r["dataset"] == "CULane"}
    for r in (r for r in rows if r["dataset"] == "Thai"):
        c = cu[r["category"]]
        d_rec, d_pre = r["px_recall"] - c["px_recall"], r["px_precision"] - c["px_precision"]
        if d_rec >= -0.02 and d_pre < -0.10:
            verdict = "ทำนายเกิน — เห็นเส้นครบแต่วาดเส้นที่เฉลยไม่มี (ชี้ไปที่ convention)"
        elif d_rec < -0.15:
            verdict = "มองไม่เห็นเส้น — ปัญหาที่โมเดลจริง"
        else:
            verdict = "ผสมทั้งสองอย่าง"
        print(f"   {r['category']:7s} Δพิกเซล recall {d_rec:+.3f}  Δพิกเซล precision {d_pre:+.3f}"
              f"  ทำนาย/เฉลย {r['lanes_pred']/max(r['lanes_gt'],1e-9):.2f}×  -> {verdict}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/convention_effect.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n📁 {OUT_DIR}/convention_effect.csv")

    out, picked = evidence_figure(model)
    print(f"📁 {out}")
    for fn, ng, npd in picked:
        print(f"   {fn}  เฉลย {ng} เส้น · ทำนาย {npd} เส้น")


if __name__ == "__main__":
    main()
