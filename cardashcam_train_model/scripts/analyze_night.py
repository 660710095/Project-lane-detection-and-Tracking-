"""
สรุปผลการทดลองแก้ปัญหา fine-tune ลืมกลางคืน — ตาราง + รูป

อ่านจากไฟล์ที่ finetune_thai.py และ eval_retention.py เขียนไว้ ไม่คำนวณโมเดลใหม่
ตัวเลขทุกตัวในรายงานต้องมาจากที่นี่ ไม่พิมพ์มือ

ตัววัดหลักคือ CULane test night 2,553 ภาพ / 16 clip **ไม่ใช่** ไทย test night
เพราะ night ของไทยทั้ง train/val/test เป็นคลิปเดียว (2/1/1 clip) วัดการลืมไม่ได้

ใช้: python3 analyze_night.py
"""
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/thai"
RETENTION = os.path.join(OUT_DIR, "retention_night_full.csv")
CEILING = "zeroshot"            # zero-shot = เพดาน ยังไม่เคยเห็นข้อมูลไทย
CONTROL = "ft_ctrl_s1_slope"    # control = พื้น ใช้ checkpoint เกณฑ์ slope ให้เทียบเป็นธรรม

# สีตาม dataviz: น้ำเงิน/ส้ม/เขียว/ม่วง/แดง
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d64550", "#8a8f98"]

ARMS = [
    ("ft_ctrl_s1", "control (seed 1)"),
    ("ft_ctrl_s2", "control (seed 2)"),
    ("ft_frz_s1",  "freeze encoder"),
    ("ft_bal_s1",  "balanced categories"),
    ("ft_rh_s1",   "rehearse CULane 50%"),
    ("ft_rh_s2",   "rehearse 50% (seed 2)"),
]

# guard แบบสัมบูรณ์ ใช้เมื่อ gap เล็กเกินกว่าจะหารเป็นอัตราส่วนได้
RECALL_GUARD_MARGIN = 0.02      # recall ต้องไม่ต่ำกว่า เพดาน − ค่านี้
THAI_F1_MARGIN = 0.02           # ไทย F1 ต้องไม่ต่ำกว่า control − ค่านี้
MIN_GAP = 0.01                  # gap เล็กกว่านี้ถือว่าไม่มี gap อัตราส่วนใช้ไม่ได้
MIN_CLIPS_BETTER = 9            # จาก 16 clip


def load_retention():
    """คืน dict[tag][scope] -> dict ของตัวเลข"""
    by = collections.defaultdict(dict)
    if not os.path.exists(RETENTION):
        raise FileNotFoundError(f"ไม่พบ {RETENTION} — รัน eval_retention.py ก่อน")
    for r in csv.DictReader(open(RETENTION)):
        by[r["tag"]][r["scope"]] = {k: (float(v) if k not in ("tag", "model", "scope") else v)
                                    for k, v in r.items()}
    return by


def load_thai(tag, criterion="slope"):
    """อ่านผลไทย test + epoch ที่เกณฑ์นั้นเลือก จาก {tag}_by_criterion.csv"""
    path = f"{OUT_DIR}/{tag}_by_criterion.csv"
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path)):
        if r["criterion"] == criterion and r["scope"] == "ALL":
            return dict(epoch=int(r["epoch"]), iou=float(r["iou"]), f1=float(r["f1"]),
                        median_err=float(r["median_err"]), lane_recall=float(r["lane_recall"]))
    return None


def paired_clips(by, tag_a, tag_b):
    """เทียบจับคู่ราย clip: คืน (Δมุม, Δrecall) เฉพาะ clip ที่ทั้งสองฝั่งจับคู่เส้นได้"""
    clips = sorted(s for s in by[tag_a] if s != "ALL")
    d_ang, d_rec = [], []
    for c in clips:
        a, b = by[tag_a][c], by[tag_b].get(c)
        if b is None or a["median_err"] != a["median_err"] or b["median_err"] != b["median_err"]:
            continue
        d_ang.append(b["median_err"] - a["median_err"])
        d_rec.append(b["lane_recall"] - a["lane_recall"])
    return np.array(d_ang), np.array(d_rec)


def main():
    by = load_retention()
    have = [(t, lab) for t, lab in ARMS if f"{t}_slope" in by]
    missing = [t for t, _ in ARMS if f"{t}_slope" not in by]
    if missing:
        print(f"⚠️  ยังไม่มีผลของ: {', '.join(missing)}")

    ceil, ctrl = by[CEILING]["ALL"], by[CONTROL]["ALL"]
    d_ang = ctrl["median_err"] - ceil["median_err"]
    d_rec = ceil["lane_recall"] - ctrl["lane_recall"]
    recall_guard = ceil["lane_recall"] - RECALL_GUARD_MARGIN
    thai_ctrl = load_thai("ft_ctrl_s1")
    f1_guard = thai_ctrl["f1"] - THAI_F1_MARGIN

    print("=" * 96)
    print("ตัววัดหลัก: CULane test night 2,553 ภาพ / 16 clip — checkpoint เกณฑ์ slope ทุก arm")
    print(f"  เพดาน (zero-shot)  มุม {ceil['median_err']:.3f}°  recall {ceil['lane_recall']:.4f}"
          f"  IoU {ceil['iou']:.3f}")
    print(f"  พื้น   (control)    มุม {ctrl['median_err']:.3f}°  recall {ctrl['lane_recall']:.4f}"
          f"  IoU {ctrl['iou']:.3f}  (epoch {thai_ctrl['epoch']})")
    print(f"  ช่องว่าง           มุม {d_ang:+.3f}°  recall {d_rec:+.4f}")

    # เกณฑ์ recall ที่ตั้งไว้ก่อนรันเป็นอัตราส่วนของ gap แต่ gap นี้อาจไม่มีอยู่จริง
    # ห้ามหารแล้วพิมพ์เปอร์เซ็นต์ออกมา มันจะได้เลขหลักร้อยหลักพันที่ไม่มีความหมาย
    recall_ratio_usable = abs(d_rec) >= MIN_GAP
    if not recall_ratio_usable:
        print(f"\n  ⚠️  เกณฑ์ 'กู้ recall คืน >= 50% ของ gap' **ใช้ไม่ได้**")
        print(f"      gap ของ recall = {d_rec:+.4f} เล็กกว่า {MIN_GAP} — control ทำ recall ได้"
              f"{' สูงกว่า' if d_rec < 0 else ' พอๆ กับ'}เพดานอยู่แล้ว")
        print(f"      สาเหตุ: เกณฑ์ slope เลือก epoch {thai_ctrl['epoch']} ซึ่ง recall ยังไม่เสียหาย"
              f" การลืมที่ checkpoint นี้เป็นเรื่องมุมล้วนๆ")
        print(f"      จึงใช้ guard สัมบูรณ์แทน: recall >= เพดาน − {RECALL_GUARD_MARGIN}"
              f" = {recall_guard:.4f}")

    spread_a = spread_r = float("nan")
    if "ft_ctrl_s2_slope" in by:
        c2 = by["ft_ctrl_s2_slope"]["ALL"]
        spread_a = abs(c2["median_err"] - ctrl["median_err"])
        spread_r = abs(c2["lane_recall"] - ctrl["lane_recall"])
        print(f"\n  ช่วงเหวี่ยงระหว่างสอง control  มุม {spread_a:.3f}°  recall {spread_r:.4f}"
              f"   <- ต้องชนะเกินนี้ถึงนับเป็นผล")

    print("\n" + "=" * 96)
    print("                      |------- CULane night (ตัววัดหลัก) -------|--- ไทย test ---|")
    print(f"{'arm':22s}{'ep':>4s}{'มุม':>8s}{'recall':>9s}{'IoU':>8s}{'กู้มุม':>8s}"
          f"{'clipดีขึ้น':>11s}{'IoU':>8s}{'F1':>7s}  คำตัดสิน")
    rows = []
    for tag, label in have:
        a = by[f"{tag}_slope"]["ALL"]
        th = load_thai(tag)
        rec_a = (ctrl["median_err"] - a["median_err"]) / d_ang if abs(d_ang) >= MIN_GAP else float("nan")
        da, _ = paired_clips(by, CONTROL, f"{tag}_slope")
        better = int((da < 0).sum())

        if tag.startswith("ft_ctrl"):
            # control ไม่ใช่ intervention ไม่ตัดสิน — seed 2 มีไว้วัดช่วงเหวี่ยงเท่านั้น
            verdict = "— (พื้น)"
        else:
            beats_noise = (ctrl["median_err"] - a["median_err"]) > spread_a
            ok_ang = rec_a >= 0.5 and beats_noise
            ok_rec = a["lane_recall"] >= recall_guard
            ok_clip = better >= MIN_CLIPS_BETTER
            ok_thai = th is not None and th["f1"] >= f1_guard
            if ok_ang and ok_rec and ok_clip and ok_thai:
                verdict = "ได้ผล"
            elif (rec_a >= 0.2 and beats_noise) or ok_rec:
                verdict = "ได้ผลบางส่วน"
            else:
                verdict = "ล้มเหลว"

        print(f"{label:22s}{th['epoch']:4d}{a['median_err']:7.2f}°{a['lane_recall']:9.3f}"
              f"{a['iou']:8.3f}{rec_a:8.0%}{better:8d}/{len(da)}"
              f"{th['iou']:8.3f}{th['f1']:7.3f}  {verdict}")
        rows.append(dict(arm=tag, label=label, epoch=th["epoch"],
                         cu_night_median_err=a["median_err"], cu_night_recall=a["lane_recall"],
                         cu_night_iou=a["iou"], recovered_angle=rec_a,
                         clips_better=better, clips_compared=len(da),
                         thai_iou=th["iou"], thai_f1=th["f1"],
                         thai_median_err=th["median_err"], thai_recall=th["lane_recall"],
                         verdict=verdict))

    print(f"\nเกณฑ์: กู้มุม >=50% ของ gap และชนะช่วงเหวี่ยง control ({spread_a:.3f}°) ·"
          f" recall >= {recall_guard:.3f} · clip ดีขึ้น >= {MIN_CLIPS_BETTER}/16 ·"
          f" ไทย F1 >= {f1_guard:.3f}")

    # ---------- รูป: เส้นทางความเสียหายรายอีพอค ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (tag, label) in enumerate(have):
        h = f"{OUT_DIR}/history_{tag}.csv"
        if not os.path.exists(h):
            continue
        d = list(csv.DictReader(open(h)))
        x = [float(r["thai_imgs_seen"]) for r in d]
        axes[0].plot(x, [float(r["val_night_median_err"]) for r in d],
                     color=COLORS[i % len(COLORS)], label=label, lw=1.8)
        axes[1].plot(x, [float(r["val_night_recall"]) for r in d],
                     color=COLORS[i % len(COLORS)], label=label, lw=1.8)
    axes[0].set_ylabel("val night angle error (deg)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("val night lane recall")
    for ax in axes:
        # แกน x เป็นจำนวนภาพไทยที่เห็น ไม่ใช่ epoch เพราะ arm ที่ผสม CULane
        # เห็นภาพไทยต่อ epoch น้อยกว่า เลข epoch จึงเทียบข้าม arm ไม่ได้
        ax.set_xlabel("Thai training images seen")
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Thai val night (n=20, 1 clip) during fine-tuning — only rehearsal stays low\n"
                 "secondary view: val night is one scene, the primary metric is CULane night",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    out = f"{OUT_DIR}/night_val_curves.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n📁 {out}")

    # ---------- รูปหลัก: เทียบ arm บนตัววัดหลัก ----------
    lbl = [r["label"].replace(" (seed 1)", " s1").replace(" (seed 2)", " s2")
           .replace(" CULane 50%", " 50%").replace("balanced categories", "balanced")
           for r in rows]
    panels = [
        ("CULane night angle error (deg)\nlower is better", "cu_night_median_err",
         ceil["median_err"], "zero-shot ceiling"),
        ("CULane night lane recall\nhigher is better", "cu_night_recall",
         recall_guard, f"guard (ceiling - {RECALL_GUARD_MARGIN})"),
        ("Thai test IoU\nhigher is better", "thai_iou", None, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (title, key, ref, reflab) in zip(axes, panels):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), vals,
                      color=[COLORS[i % len(COLORS)] for i in range(len(rows))], width=0.62)
        if ref is not None:
            ax.axhline(ref, color="#555", ls="--", lw=1.1, label=reflab)
            ax.legend(fontsize=8, frameon=False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(lbl, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("Anti-forgetting arms — slope-selected checkpoint of each run", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = f"{OUT_DIR}/night_comparison.png"
    plt.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"📁 {out2}")

    with open(f"{OUT_DIR}/night_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"📁 {OUT_DIR}/night_summary.csv")


if __name__ == "__main__":
    main()
