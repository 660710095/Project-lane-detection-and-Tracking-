"""
เส้นโค้งปริมาณข้อมูล — สรุปผลจาก 3 จุด (245 / 505 / 749 ภาพไทย) เป็นตาราง + รูป + รายงาน

ตอบคำถามให้ทีม: เพื่อนควร label ภาพไทยเพิ่มอีกไหม
เส้นยังชันที่ 749 = คุ้ม · เริ่มแบน = ปัญหาอยู่ที่อื่น ควรไปทำ augmentation หรือเปลี่ยนสถาปัตยกรรม

ทุกจุดใช้ config เดียวกัน: --rehearse-frac 0.5 --seed 1, 30 epoch, checkpoint เกณฑ์ slope
จุดที่ 3 คือรัน ft_rh_s1 ที่ทำไว้แล้ว (ตรวจแล้วว่า rehearse_frac ตรงกันจาก thai_imgs_seen)

ใช้: python3 make_datasize_report.py
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/thai"
# อ่านสองเกณฑ์เสมอ ไม่ใช่เกณฑ์เดียว
# เพราะเกณฑ์ slope เลือก epoch ต่างกันมากในสามจุดนี้ (17 / 8 / 25) ซึ่งเป็นตัวปน
# ถ้าดูเกณฑ์เดียวจะแยกไม่ออกว่าส่วนต่างมาจากปริมาณข้อมูลหรือจาก epoch ที่หยิบ
CRITERIA = ("slope", "f1")
MIN_MEANINGFUL_IOU = 0.02      # เพิ่มข้อมูล 3 เท่าแล้วได้น้อยกว่านี้ ถือว่าแบน
# (จำนวนภาพไทยที่ใช้เทรน, tag)
POINTS = [(245, "ft_n245"), (505, "ft_n505"), (749, "ft_rh_s1")]
# รากของชุดเทรนแต่ละจุด ใช้นับว่าหมวดไหนได้ภาพเพิ่มเท่าไหร่จริง
ROOTS = {245: "data_subsets/thai_250", 505: "data_subsets/thai_500", 749: "thai_road_lane"}
CATS = ("normal", "curve", "night")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def thai_rows(tag, criterion):
    """คืน dict[scope] ของผลไทย test ที่ checkpoint เกณฑ์ที่ระบุ"""
    out = {}
    for r in csv.DictReader(open(f"{OUT_DIR}/{tag}_by_criterion.csv")):
        if r["criterion"] == criterion:
            out[r["scope"]] = r
    return out


def retention(tag):
    """
    CULane night ของ checkpoint เกณฑ์ slope — ดูว่าข้อมูลไทยน้อยลงกระทบการลืมไหม

    วัด retention ไว้เฉพาะ checkpoint เกณฑ์ slope ของแต่ละจุด ตารางเกณฑ์ f1 จึงแสดง
    ค่าเดียวกัน ต้องพิมพ์หัวคอลัมน์กำกับว่าเป็นของ slope ไม่ให้เข้าใจผิดว่าเป็นของ f1
    """
    want = f"{tag}_slope"
    for r in csv.DictReader(open(f"{OUT_DIR}/retention_night_full.csv")):
        if r["tag"] == want and r["scope"] == "ALL":
            return r
    return None


def check_config():
    """ทุกจุดต้องมาจาก rehearse_frac เดียวกัน ไม่งั้นเทียบกันไม่ได้"""
    fr = {}
    for n, tag in POINTS:
        rows = list(csv.DictReader(open(f"{OUT_DIR}/history_{tag}.csv")))
        seen = float(rows[0]["thai_imgs_seen"])
        fr[tag] = round(1 - seen / n, 2)
    if len(set(fr.values())) != 1:
        raise SystemExit(f"rehearse_frac ไม่ตรงกันทุกจุด {fr} — เทียบกันไม่ได้ ต้องรันใหม่")
    return list(fr.values())[0]


def main():
    frac = check_config()
    print(f"ทุกจุดใช้ rehearse_frac = {frac} ✓\n")

    data = {}
    for crit in CRITERIA:
        rows = []
        for n, tag in POINTS:
            t = thai_rows(tag, crit)
            rows.append(dict(n=n, tag=tag, epoch=int(t["ALL"]["epoch"]), all=t["ALL"],
                             ret=retention(tag)))
        data[crit] = rows

    f = lambda v, k=3: f"{float(v):.{k}f}"
    for crit in CRITERIA:
        print(f"--- checkpoint เกณฑ์ {crit} ---")
        print(f"{'ภาพไทย':>8s}{'ep':>4s}{'IoU':>8s}{'F1':>8s}{'มุม':>8s}{'recall':>9s}"
              f"{'CUnight มุม(slope)':>18s}{'recall':>9s}")
        for d in data[crit]:
            a, r = d["all"], d["ret"]
            rt = f"{float(r['median_err']):.2f}°" if r else "—"
            rr = f(r["lane_recall"]) if r else "—"
            print(f"{d['n']:8d}{d['epoch']:4d}{float(a['iou']):8.3f}{float(a['f1']):8.3f}"
                  f"{float(a['median_err']):7.2f}°{float(a['lane_recall']):9.3f}{rt:>18s}{rr:>9s}")
        print()

    # ตัดสินจาก "ได้เท่าไหร่จริง" เมื่อข้อมูลเพิ่ม 3 เท่า ไม่ใช่จากความชันเปรียบเทียบ
    # ความชันเปรียบเทียบไวต่อ epoch ที่เกณฑ์หยิบ จนบอกว่า "เร่งขึ้น" ได้ทั้งที่ค่าจริงแทบไม่ขยับ
    xs = [d["n"] for d in data[CRITERIA[0]]]
    gains = {}
    for crit in CRITERIA:
        rows = data[crit]
        gains[crit] = {k: float(rows[-1]["all"][k]) - float(rows[0]["all"][k])
                       for k in ("iou", "f1", "median_err", "lane_recall")}
    ratio_n = xs[-1] / xs[0]
    print(f"ข้อมูลเพิ่ม {ratio_n:.2f} เท่า ({xs[0]} เป็น {xs[-1]} ภาพ) ได้อะไรเพิ่ม")
    print(f"{'เกณฑ์':>9s}{'Δ IoU':>9s}{'Δ F1':>9s}{'Δ มุม':>9s}{'Δ recall':>11s}")
    for crit in CRITERIA:
        g = gains[crit]
        print(f"{crit:>9s}{g['iou']:+9.3f}{g['f1']:+9.3f}{g['median_err']:+8.2f}°{g['lane_recall']:+11.3f}")

    iou_gain = max(g["iou"] for g in gains.values())
    both_positive = all(g["iou"] > 0 for g in gains.values())
    angle_worse = all(g["median_err"] > 0 for g in gains.values())
    recall_worse = all(g["lane_recall"] < 0 for g in gains.values())

    if iou_gain >= MIN_MEANINGFUL_IOU and both_positive:
        verdict = "ยังชัน — label เพิ่มคุ้ม"
    elif both_positive:
        verdict = "แบน — label เพิ่มไม่ใช่ตัวแปรหลัก"
    else:
        verdict = "ไม่สรุป — สองเกณฑ์ให้ทิศต่างกัน"
    print(f"\nคำตัดสิน: {verdict}")
    print(f"  (เกณฑ์: Δ IoU ต้อง >= {MIN_MEANINGFUL_IOU} และบวกทั้งสองเกณฑ์ ถึงจะนับว่าชัน)")
    if angle_worse:
        print("  ⚠️  มุมคลาด **แย่ลง** ทั้งสองเกณฑ์เมื่อข้อมูลเพิ่ม")
    if recall_worse:
        print("  ⚠️  lane_recall **ตก** ทั้งสองเกณฑ์เมื่อข้อมูลเพิ่ม")

    # ---------- แยกหมวด — แถว ALL ถูก normal ครอบงำ (82% ของ test) จนบังโครงสร้างจริง ----------
    print("\nแยกหมวด (เกณฑ์ f1) — ภาพเทรนที่หมวดนั้นได้จริง เทียบกับผลที่ได้")
    print(f"{'หมวด':>8s}{'ภาพ 245':>10s}{'ภาพ 749':>10s}{'IoU 245':>9s}{'IoU 749':>9s}"
          f"{'Δ IoU':>9s}{'Δ recall':>11s}")
    percat = {}
    for c in CATS:
        n_img = {}
        for n in (245, 749):
            rows_ = list(csv.DictReader(open(f"{ROOTS[n]}/train_list.csv")))
            n_img[n] = sum(1 for r in rows_ if r["category"] == c)
        lo = thai_rows("ft_n245", "f1")[c]
        hi = thai_rows("ft_rh_s1", "f1")[c]
        d_iou = float(hi["iou"]) - float(lo["iou"])
        d_rec = float(hi["lane_recall"]) - float(lo["lane_recall"])
        percat[c] = dict(n_lo=n_img[245], n_hi=n_img[749], iou_lo=float(lo["iou"]),
                         iou_hi=float(hi["iou"]), d_iou=d_iou, d_recall=d_rec)
        print(f"{c:>8s}{n_img[245]:>10d}{n_img[749]:>10d}{float(lo['iou']):9.3f}"
              f"{float(hi['iou']):9.3f}{d_iou:+9.3f}{d_rec:+11.3f}")

    grow = [c for c in CATS if percat[c]["d_iou"] >= MIN_MEANINGFUL_IOU]
    flat = [c for c in CATS if percat[c]["d_iou"] < MIN_MEANINGFUL_IOU]
    if grow:
        print(f"\n  หมวดที่ยังได้ผลจากข้อมูลเพิ่ม (Δ IoU >= {MIN_MEANINGFUL_IOU}): {', '.join(grow)}")
    if flat:
        print(f"  หมวดที่อิ่มตัวแล้ว: {', '.join(flat)}")
    if grow and flat:
        print("  => คำตอบไม่ใช่ 'label เพิ่มคุ้ม/ไม่คุ้ม' แต่เป็น 'label หมวดไหนเพิ่ม'")

    rows = data["slope"]
    iou = [float(d["all"]["iou"]) for d in rows]
    f1 = [float(d["all"]["f1"]) for d in rows]
    data_slope, data_f1 = data["slope"], data["f1"]

    # ---------- รูป ----------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    # ตรึงแกน y ให้ครอบช่วงที่มีความหมายจริงในโปรเจกต์นี้ ไม่ปล่อย auto-scale
    # ไม่งั้นส่วนต่าง 0.003 จะถูกยืดเต็มแกนจนดูเหมือนแนวโน้มชัด ทั้งที่ข้อสรุปคือ "แบน"
    # ค่าอ้างอิง: zero-shot ไทย IoU 0.354 / F1 0.523 / มุม 1.90° / recall 0.953
    keys = [("iou", "Thai test IoU", "higher is better", (0.30, 0.55), 0.354),
            ("f1", "Thai test F1", "higher is better", (0.45, 0.70), 0.523),
            ("median_err", "Thai test angle error (deg)", "lower is better", (0.0, 3.0), 1.90),
            ("lane_recall", "Thai test lane recall", "higher is better", (0.70, 1.00), 0.953)]
    for ax, (k, title, hint, ylim, ref) in zip(axes, keys):
        ax.axhline(ref, color="#8a8f98", ls="--", lw=1.0)
        ax.annotate("zero-shot", (xs[0], ref), textcoords="offset points", xytext=(2, 3),
                    fontsize=7, color="#8a8f98")
        ax.set_ylim(*ylim)
        for i, crit in enumerate(CRITERIA):
            ys = [float(d["all"][k]) for d in data[crit]]
            ax.plot(xs, ys, "o-", color=COLORS[i], lw=2, ms=6,
                    label=f"{crit}-selected checkpoint")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 7 if i == 0 else -13), ha="center", fontsize=7,
                            color=COLORS[i])
        ax.set_xlabel("Thai training images")
        ax.set_title(f"{title}\n{hint}", fontsize=10)
        ax.set_xticks(xs)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Data-scaling curve — Thai fine-tuning with 50% CULane rehearsal\n"
                 "two checkpoint criteria shown, because the epoch each picks differs "
                 "(slope: 17 / 8 / 25) and that alone can fake a trend", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out = f"{OUT_DIR}/datasize_curve.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n📁 {out}")

    with open(f"{OUT_DIR}/datasize_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["criterion", "n_thai_train", "tag", "epoch", "thai_iou", "thai_f1", "thai_median_err",
                    "thai_recall", "cu_night_median_err", "cu_night_recall", "cu_night_iou"])
        for crit in CRITERIA:
            for d in data[crit]:
                a, r = d["all"], d["ret"]
                w.writerow([crit, d["n"], d["tag"], d["epoch"], a["iou"], a["f1"],
                            a["median_err"], a["lane_recall"],
                            r["median_err"] if r else "", r["lane_recall"] if r else "",
                            r["iou"] if r else ""])
    print(f"📁 {OUT_DIR}/datasize_summary.csv")
    return data, verdict, gains


if __name__ == "__main__":
    main()
