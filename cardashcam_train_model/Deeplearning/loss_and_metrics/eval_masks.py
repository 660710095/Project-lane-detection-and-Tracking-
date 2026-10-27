"""
วัดผล mask ที่มาจากวิธีไหนก็ได้ — ไม่ต้องมีโมเดล

ทำไมต้องมี: `finetune_thai.evaluate()` รับ model เป็นอาร์กิวเมนต์แรก ต้องเป็นโมเดล torch
ฝั่ง image processing ได้ผลออกมาเป็นไฟล์ PNG เรียกใช้ไม่ได้ ไฟล์นี้คือตัวเชื่อม
ให้ทั้งสองวิธีวัดด้วยไม้บรรทัดอันเดียวกัน แล้วเอาตัวเลขมาวางข้างกันได้จริง

คายคอลัมน์ชุดเดียวกับ `{tag}_by_criterion.csv` ของฝั่ง DL เป๊ะ

ใช้:
  python3 eval_masks.py --pred-dir /path/to/my_masks --tag ip_hough
  python3 eval_masks.py --pred-dir ... --root thai_road_lane --split test
"""
import argparse
import csv
import os
import sys

import cv2
cv2.setNumThreads(0)
import numpy as np

sys.path.append('.')
from dataset import LaneDatasetFromCSV, IMG_SIZE
from slope_metrics import slope_metrics_for_pair

OUT_DIR = "results/ip"
H, W = IMG_SIZE


def load_pred(path):
    """
    โหลด mask ที่ทำนายมา แล้วทำให้เหมือนกับทางที่เฉลยถูกโหลด

    resize เป็น IMG_SIZE ด้วย INTER_NEAREST แล้ว binarize ที่ >127
    (เฉลยใน LaneDatasetFromCSV ใช้ nearest + >127 เหมือนกัน — ถ้าใช้คนละแบบ เทียบกันไม่ได้)

    วัดแล้ว: ย่อจากขนาดที่เป็น**ทวีคูณลงตัว** ของ 448x800 ได้ผลเท่ากันเป๊ะ (ต่าง 0.00)
    แต่ขนาดที่ไม่ลงตัว (เช่น 1080x1920) ทำให้ IoU เพี้ยน 0.005-0.010 เพราะพิกเซลเลื่อนตอน resample
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, None
    orig = m.shape
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m > 127, orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="โฟลเดอร์ mask ที่ทำนายมา (.png)")
    ap.add_argument("--root", default="thai_road_lane", help="รากชุดข้อมูลที่มีเฉลย")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tag", required=True, help="ชื่อวิธี ใช้ตั้งชื่อไฟล์ผล")
    ap.add_argument("--out", default=None, help="ไม่ใส่ = results/ip/{tag}.csv")
    args = ap.parse_args()
    out = args.out or os.path.join(OUT_DIR, f"{args.tag}.csv")

    rows = list(csv.DictReader(open(f"{args.root}/{args.split}_list.csv")))
    ds = LaneDatasetFromCSV(f"{args.root}/{args.split}_list.csv",
                            f"{args.root}/images", f"{args.root}/masks", augment=False)
    assert [r["filename"] for r in rows] == [s[0] for s in ds.samples], \
        "ลำดับ CSV กับ dataset ไม่ตรงกัน"

    acc = {}
    def slot(s):
        return acc.setdefault(s, dict(tp=0., fp=0., fn=0., inter=0., uni=0.,
                                      err=[], gt=0, pred=0, match=0, img=0))

    missing, resized, odd_size = [], 0, set()
    for i, r in enumerate(rows):
        pred_path = os.path.join(args.pred_dir, r["filename"].replace(".jpg", ".png"))
        P, orig = load_pred(pred_path)
        if P is None:
            missing.append(r["filename"])
            continue
        if orig != (H, W):
            resized += 1
            # ทวีคูณลงตัวย่อกลับได้เป๊ะ ที่ไม่ลงตัวพิกเซลจะเลื่อน ต้องเตือน
            if orig[0] % H or orig[1] % W:
                odd_size.add(orig)
        _, mask_t = ds[i]
        T = mask_t[0].numpy() > 0.5
        res = slope_metrics_for_pair(T, P)
        for s in ("ALL", r["category"]):
            a = slot(s)
            a["img"] += 1
            a["tp"] += (P & T).sum(); a["fp"] += (P & ~T).sum(); a["fn"] += (~P & T).sum()
            a["inter"] += (P & T).sum(); a["uni"] += (P | T).sum()
            a["gt"] += res["n_gt"]; a["pred"] += res["n_pred"]; a["match"] += res["n_matched"]
            a["err"] += [d["err"] for d in res["pairs"]]

    # บอกให้ชัดว่าวัดจากกี่ภาพ และมีอะไรหายไหม — เงียบแล้วคายเลขออกมาคืออันตราย
    n_found = len(rows) - len(missing)
    print(f"📁 {args.pred_dir}")
    print(f"   จับคู่ได้ {n_found}/{len(rows)} ภาพ" + (f" · resize ให้ {resized} ภาพ" if resized else ""))
    if missing:
        print(f"   ⚠️  ไม่พบไฟล์ทำนาย {len(missing)} ภาพ — ตัวเลขข้างล่างวัดจากที่เหลือเท่านั้น")
        for m in missing[:5]:
            print(f"        {m.replace('.jpg', '.png')}")
        if len(missing) > 5:
            print(f"        ... อีก {len(missing)-5} ไฟล์")
    if odd_size:
        sz = ", ".join(f"{h}x{w}" for h, w in sorted(odd_size))
        print(f"   ⚠️  ขนาด {sz} ไม่ใช่ทวีคูณลงตัวของ {H}x{W}")
        print(f"        พิกเซลเลื่อนตอน resize ทำให้ IoU เพี้ยนราว 0.005-0.010 (วัดแล้ว)")
        print(f"        ส่ง mask ที่ {H}x{W} หรือทวีคูณลงตัว (เช่น {H*2}x{W*2}) จะได้เลขที่เทียบได้ตรง")
    if not acc:
        raise SystemExit("ไม่มีภาพไหนจับคู่ได้เลย — เช็ค --pred-dir และชื่อไฟล์")

    out_rows = []
    for s in ("ALL", "normal", "curve", "night"):
        if s not in acc:
            continue
        a = acc[s]
        e = np.array(a["err"]) if a["err"] else np.array([np.nan])
        out_rows.append(dict(
            scope=s, n_img=a["img"],
            iou=a["inter"] / max(a["uni"], 1),
            f1=2 * a["tp"] / max(2 * a["tp"] + a["fp"] + a["fn"], 1),
            median_err=float(np.median(e)), acc5=float(np.mean(e <= 5.0)),
            lane_recall=a["match"] / a["gt"] if a["gt"] else float("nan"),
            lane_precision=a["match"] / a["pred"] if a["pred"] else float("nan"),
            px_recall=a["tp"] / max(a["tp"] + a["fn"], 1),
            px_precision=a["tp"] / max(a["tp"] + a["fp"], 1)))

    print(f"\n{'หมวด':>8s}{'ภาพ':>6s}{'IoU':>8s}{'F1':>8s}{'มุมคลาด':>10s}{'acc@5°':>9s}"
          f"{'lane rec':>10s}{'lane prec':>11s}{'px rec':>9s}{'px prec':>9s}")
    for r in out_rows:
        print(f"{r['scope']:>8s}{r['n_img']:6d}{r['iou']:8.3f}{r['f1']:8.3f}"
              f"{r['median_err']:9.2f}°{r['acc5']:9.3f}{r['lane_recall']:10.3f}"
              f"{r['lane_precision']:11.3f}{r['px_recall']:9.3f}{r['px_precision']:9.3f}")

    print("\nอ่านยังไง: px recall ต่ำ = มองไม่เห็นเส้น · px precision ต่ำ = วาดเส้นที่เฉลยไม่มี")
    print("เฉลยไทย label แค่เลนที่รถวิ่งอยู่ ถ้าวิธีของคุณจับเลนข้างเคียงด้วย precision จะต่ำ")
    print("โดยที่ไม่ได้ผิด — ดู results/thai/REPORT_ANNOTATION.md")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print(f"\n📁 {out}")


if __name__ == "__main__":
    main()
