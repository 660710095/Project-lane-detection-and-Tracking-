"""
วัด retention: โมเดลยังทำถนนกลางคืนของ CULane ได้ดีแค่ไหนหลัง fine-tune ด้วยข้อมูลไทย

ทำไมต้องมีไฟล์นี้: ไทย test หมวด night มีแค่ 17 ภาพ (~30-40 คู่เส้น) เล็กเกินกว่าจะสรุป
ว่า intervention ตัวไหนแก้ปัญหา "ลืมกลางคืน" ได้จริง คำว่าลืมแปลว่า retention
จึงต้องวัดบนชุดที่โมเดลเคยทำได้และมีขนาดพอ — CULane test หมวด night

ชุดที่ใช้: CULane test หมวด night **ทั้งหมด 2,553 ภาพ / 16 clip** (ใช้เวลา ~70 วินาที
ไม่คุ้มที่จะ subsample) และแยกผล**ราย clip** ด้วย

ทำไมต้องรายคลิป: ไทย night test 17 เฟรมมาจาก **clip เดียว** (sathon_road_night01)
train ก็มีแค่ 2 clip val 1 clip ตัวเลข night ของไทยจึงเป็น "หนึ่งฉาก" ไม่ใช่กลุ่มตัวอย่าง
ต้องมีค่ากระจายราย clip ของ CULane มาบอกว่าส่วนต่างที่เห็นใหญ่กว่าความเหวี่ยงจริงหรือเปล่า

เคล็ด: evaluate() จัดกลุ่มตาม cat_of[filename] อยู่แล้ว ส่ง clip_id เข้าไปแทน category
ก็ได้ผลราย clip ฟรี โดยไม่ต้องแก้ evaluate() เลย

วัดหมวดอื่นได้ด้วย `--category normal|curve` เพื่อตอบว่าการลืมจำกัดอยู่แค่กลางคืน
หรือเป็น domain shift ทั้งโดเมน

ใช้:
  python3 eval_retention.py --model models/best_model_resnet50_ssl_bs16_ep30.pth --tag zeroshot
  python3 eval_retention.py --model ... --tag zeroshot --category normal
  python3 eval_retention.py --model models/thai_ft_ctrl_s1_best_slope.pth --tag ft_ctrl_s1_slope
"""
import argparse
import csv
import os
import sys

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append('.')
from dataset import LaneDatasetFromCSV
from finetune_thai import build_model, evaluate

CULANE_ROOT = "/home/jovyan/shared/donut/CarDashCam/CULane"
OUT_DIR = "results/thai"


def category_set(category="night"):
    """
    คืน (dataset ย่อย, รายชื่อไฟล์, dict filename -> clip_id) ของภาพหมวดนั้นทั้งหมดใน CULane test

    ส่ง clip_id เป็น "หมวด" ให้ evaluate() เพื่อให้ได้ scope ราย clip มาด้วย
    """
    rows = list(csv.DictReader(open(os.path.join(CULANE_ROOT, "test_list.csv"))))
    ds = LaneDatasetFromCSV(os.path.join(CULANE_ROOT, "test_list.csv"),
                            os.path.join(CULANE_ROOT, "images"),
                            os.path.join(CULANE_ROOT, "masks"), augment=False)
    picks = [i for i, r in enumerate(rows) if r["category"] == category]
    if not picks:
        raise ValueError(f"ไม่มีภาพหมวด {category!r} ใน CULane test")
    assert [rows[i]["filename"] for i in picks] == [ds.samples[i][0] for i in picks], \
        "ลำดับ CSV กับ dataset ไม่ตรงกัน"
    names = [rows[i]["filename"] for i in picks]
    clip_of = {rows[i]["filename"]: rows[i]["clip_id"] for i in picks}
    return Subset(ds, picks), names, clip_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ไฟล์ .pth ที่จะวัด")
    ap.add_argument("--tag", required=True, help="ชื่อสั้นๆ ใช้ตั้งชื่อแถวในไฟล์ผล")
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--category", default="night", choices=["night", "normal", "curve"],
                    help="หมวดถนนใน CULane test ที่จะวัด (default night)")
    ap.add_argument("--out", default=None,
                    help="ไม่ใส่ = results/thai/retention_{category}_full.csv")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(OUT_DIR, f"retention_{args.category}_full.csv")

    sub, names, clip_of = category_set(args.category)
    clips = sorted(set(clip_of.values()))
    print(f"🌙 CULane test {args.category}: {len(sub)} ภาพ / {len(clips)} clip")

    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=torch.cuda.is_available())
    model = build_model(args.encoder, args.model)
    res = evaluate(model, loader, names, clip_of)

    os.makedirs(OUT_DIR, exist_ok=True)
    header = ["tag", "model", "scope", "n_img", "iou", "f1", "median_err", "acc5", "lane_recall"]
    exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        for s_ in ["ALL"] + clips:
            r = res[s_]
            w.writerow([args.tag, os.path.basename(args.model), s_, r["n_img"], r["iou"],
                        r["f1"], r["median_err"], r["acc5"], r["lane_recall"]])

    a = res["ALL"]
    # clip ที่จับคู่ไม่ได้เลยจะได้ nan (เจอจริง 1 clip ใน CULane night) ต้องใช้ nan-aware
    per_clip = np.array([res[c]["median_err"] for c in clips])
    n_nan = int(np.isnan(per_clip).sum())
    q1, q3 = np.nanpercentile(per_clip, [25, 75])
    print(f"\n{'scope':22s}{'IoU':>8s}{'F1':>8s}{'มุมคลาด':>10s}{'acc@5°':>9s}{'recall':>9s}")
    print(f"{'ALL':22s}{a['iou']:8.3f}{a['f1']:8.3f}"
          f"{a['median_err']:9.2f}°{a['acc5']:9.3f}{a['lane_recall']:9.3f}")
    print(f"ราย clip ({len(clips)} clip, จับคู่ไม่ได้ {n_nan}): "
          f"มุมคลาด median {np.nanmedian(per_clip):.2f}° "
          f"IQR {q1:.2f}-{q3:.2f}° ต่ำสุด {np.nanmin(per_clip):.2f}° "
          f"สูงสุด {np.nanmax(per_clip):.2f}°")
    print(f"\n📁 {args.out}")


if __name__ == "__main__":
    main()
