"""
สรุปการทดลอง slope loss — ตาราง + รูป + คำตัดสินตามเกณฑ์ที่ตั้งไว้ก่อนรัน

คำถามที่ตอบ: เอา "มุมของเส้น" เข้าไปใน loss แล้วดีขึ้นจริงไหม เทียบกับเทรนปกติแล้ววัดด้วยความชัน

เทียบที่ checkpoint เกณฑ์ f1 ทุก arm เพราะเกณฑ์ slope เลือก epoch กระโดดมาก (วัดแล้ว 17/8/25
ในการทดลองปริมาณข้อมูล) ถ้าเทียบด้วยเกณฑ์นั้นจะแยกไม่ออกว่าส่วนต่างมาจาก loss หรือจาก epoch ที่หยิบ

ใช้: python3 make_slopeloss_report.py
"""
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/thai"
CRITERION = "f1"
CONTROL = "ft_rh_s1"
ARMS = [(CONTROL, 0.0, "control (BCE+Dice)"),
        ("ft_sl1_s1", 0.1, "+ slope loss 0.1"),
        ("ft_sl3_s1", 0.3, "+ slope loss 0.3")]

# เกณฑ์ตั้งไว้ก่อนรัน ห้ามแก้หลังเห็นผล
MIN_ANGLE_GAIN = 0.10      # องศา — ช่วงเหวี่ยงระหว่าง seed ที่วัดไว้คือ 0.03°
MAX_IOU_DROP = 0.02
MIN_CATS_BETTER = 2        # จาก 3 หมวด — กัน composition effect ของแถว ALL
CATS = ("normal", "curve", "night")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def by_criterion(tag, scope="ALL"):
    path = f"{OUT_DIR}/{tag}_by_criterion.csv"
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path)):
        if r["criterion"] == CRITERION and r["scope"] == scope:
            return r
    return None


def retention(tag):
    want = f"{tag}_{CRITERION}"
    path = f"{OUT_DIR}/retention_night_full.csv"
    for r in csv.DictReader(open(path)):
        if r["tag"] == want and r["scope"] == "ALL":
            return r
    return None


def history(tag):
    return list(csv.DictReader(open(f"{OUT_DIR}/history_{tag}.csv")))


def main():
    rows = []
    for tag, lam, label in ARMS:
        a = by_criterion(tag)
        if a is None:
            print(f"⚠️  ยังไม่มีผลของ {tag}")
            continue
        h = history(tag)
        pairs = [int(x.get("train_slope_pairs") or 0) for x in h]
        sl = [float(x.get("train_slope_loss") or 0.0) for x in h]
        rows.append(dict(tag=tag, lam=lam, label=label, epoch=int(a["epoch"]),
                         iou=float(a["iou"]), f1=float(a["f1"]),
                         median_err=float(a["median_err"]),
                         lane_recall=float(a["lane_recall"]),
                         min_pairs=min(pairs) if pairs else 0,
                         mean_slope_loss=float(np.mean(sl)) if sl else 0.0,
                         ret=retention(tag),
                         percat={c: by_criterion(tag, c) for c in CATS}))

    ctrl = next(r for r in rows if r["tag"] == CONTROL)
    print("=" * 96)
    print(f"checkpoint เกณฑ์ {CRITERION} ทุก arm · ทุกตัวใช้ --rehearse-frac 0.5 --seed 1 "
          f"ต่างกันแค่ lambda")
    print(f"{'arm':22s}{'λ':>5s}{'ep':>4s}{'ไทย IoU':>10s}{'ไทย F1':>9s}{'ไทย มุม':>10s}"
          f"{'recall':>9s}{'CUnight มุม':>13s}{'คู่เส้นต่ำสุด':>14s}")
    for r in rows:
        rt = f"{float(r['ret']['median_err']):.2f}°" if r["ret"] else "—"
        print(f"{r['label']:22s}{r['lam']:5.1f}{r['epoch']:4d}{r['iou']:10.3f}{r['f1']:9.3f}"
              f"{r['median_err']:9.2f}°{r['lane_recall']:9.3f}{rt:>13s}"
              f"{r['min_pairs']:>14d}")

    # แถว ALL เป็น pooled median ของทุกคู่เส้น ถ้า recall เปลี่ยน องค์ประกอบของคู่ก็เปลี่ยน
    # ทำให้ ALL ขยับได้ทั้งที่ไม่มีหมวดไหนดีขึ้นจริง ต้องตรวจรายหมวดเสมอ
    print("\nมุมคลาดแยกหมวด (เกณฑ์ f1) — ใช้ตรวจว่าแถว ALL ที่ดีขึ้นเป็นของจริงหรือ composition effect")
    print(f"{'arm':22s}" + "".join(f"{c:>12s}" for c in CATS))
    for r in rows:
        line = f"{r['label']:22s}"
        for c in CATS:
            pc = r["percat"][c]
            line += f"{float(pc['median_err']):11.2f}°" if pc else f"{'—':>12s}"
        print(line)

    print("\n" + "=" * 96)
    print(f"คำตัดสิน — เกณฑ์ตั้งไว้ก่อนรัน: มุมดีขึ้น >= {MIN_ANGLE_GAIN}° ·"
          f" IoU ไม่ตกเกิน {MAX_IOU_DROP} · คู่เส้นไม่เป็น 0")
    print(f"   เพิ่มด่านตรวจหลังเห็นผล: ต้องมีหมวดที่ดีขึ้น >= {MIN_CATS_BETTER}/3")
    print("   (เพิ่มเพราะพบว่าแถว ALL ขยับได้จาก recall ที่เปลี่ยน ไม่ใช่จากการเล็งที่ดีขึ้น")
    print("    เป็นการทำให้เกณฑ์**เข้มขึ้น** ไม่ใช่ผ่อนให้ผลผ่าน — บันทึกไว้ตามตรง)")
    for r in rows:
        if r["tag"] == CONTROL:
            print(f"   {r['label']:22s} — (ตัวควบคุม)")
            continue
        d_ang = ctrl["median_err"] - r["median_err"]       # บวก = ดีขึ้น
        d_iou = r["iou"] - ctrl["iou"]
        ok_ang = d_ang >= MIN_ANGLE_GAIN
        ok_iou = d_iou >= -MAX_IOU_DROP
        ok_run = r["min_pairs"] > 0
        n_better = sum(1 for c in CATS
                       if r["percat"][c] and ctrl["percat"][c]
                       and float(r["percat"][c]["median_err"]) < float(ctrl["percat"][c]["median_err"]))
        ok_cat = n_better >= MIN_CATS_BETTER
        if not ok_run:
            verdict = "ผลไม่มีความหมาย (เทอมไม่ทำงาน)"
        elif ok_ang and ok_iou and ok_cat:
            verdict = "ได้ผล"
        elif ok_ang and ok_iou and not ok_cat:
            verdict = "ไม่สรุป — ALL ดีขึ้นแต่ไม่มีหมวดไหนยืนยัน (composition effect)"
        else:
            verdict = "ล้มเหลว"
        print(f"   {r['label']:22s} Δมุม {d_ang:+.2f}° {'✓' if ok_ang else '✗'}"
              f"   ΔIoU {d_iou:+.3f} {'✓' if ok_iou else '✗'}"
              f"   หมวดที่ดีขึ้น {n_better}/3 {'✓' if ok_cat else '✗'}"
              f"   คู่เส้น {'✓' if ok_run else '✗'}\n{'':25s}-> {verdict}")

    # ---------- รูป ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    labels = [r["label"].replace("control (BCE+Dice)", "control").replace("+ slope loss ", "λ=")
              for r in rows]
    panels = [("Thai test angle error (deg)\nlower is better", "median_err", True),
              ("Thai test IoU\nhigher is better", "iou", False),
              ("Thai test lane recall\nhigher is better", "lane_recall", False)]
    for ax, (title, key, lower_better) in zip(axes, panels):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=COLORS[:len(rows)], width=0.6)
        ax.axhline(ctrl[key], color="#555", ls="--", lw=1.1, label="control")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Slope in the loss vs slope as a metric only — f1-selected checkpoint, "
                 "same data and schedule, only lambda differs", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{OUT_DIR}/slopeloss_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n📁 {out}")

    with open(f"{OUT_DIR}/slopeloss_summary.csv", "w", newline="") as f:
        cols = ["tag", "lam", "label", "epoch", "iou", "f1", "median_err", "lane_recall",
                "min_pairs", "mean_slope_loss", "cu_night_median_err", "cu_night_recall"]
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r["tag"], r["lam"], r["label"], r["epoch"], r["iou"], r["f1"],
                        r["median_err"], r["lane_recall"], r["min_pairs"],
                        r["mean_slope_loss"],
                        r["ret"]["median_err"] if r["ret"] else "",
                        r["ret"]["lane_recall"] if r["ret"] else ""])
    print(f"📁 {OUT_DIR}/slopeloss_summary.csv")


if __name__ == "__main__":
    main()
