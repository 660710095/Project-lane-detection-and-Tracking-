"""
เส้นโค้งปริมาณข้อมูล — เทรนด้วยภาพไทย 245 / 505 / 749 ภาพ แล้วเทียบผลบน test ชุดเดียวกัน

ตอบคำถามให้ทีม: label เพิ่มแล้วคุ้มไหม
เส้นโค้งยังชัน = คุ้ม · เริ่มแบน = ปัญหาอยู่ที่อื่น ควรไปทำ augmentation หรือเปลี่ยนสถาปัตยกรรม

subset สร้างจาก make_thai_subsets.py (แบ่งตาม clip_id ไม่ใช่รายเฟรม)
วิธีชี้ finetune_thai ไปที่ subset: แทนที่ค่า THAI_ROOT ในโมดูล ไม่ต้องแก้ไฟล์นั้น
val/test ของทุก subset เป็นชุดเดิมทั้งหมด เปลี่ยนแค่ชุดเทรน

ใช้: python3 run_datasize.py --rehearse-frac 0.5
"""
import argparse
import os
import sys

sys.path.append('.')
import finetune_thai as F

INIT = "models/best_model_resnet50_ssl_bs16_ep30.pth"
SUBSETS = [("data_subsets/thai_250", "ft_n245"), ("data_subsets/thai_500", "ft_n505")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    original = F.THAI_ROOT
    for root, tag in SUBSETS:
        if not os.path.exists(f"{root}/train_list.csv"):
            raise FileNotFoundError(f"ไม่พบ {root} — รัน make_thai_subsets.py ก่อน")
        print(f"\n{'='*70}\n{tag}  (ชุดเทรนจาก {root})\n{'='*70}")
        F.THAI_ROOT = root
        try:
            F.run(mode="finetune", init=INIT, tag=tag, seed=args.seed,
                  epochs=args.epochs, rehearse_frac=args.rehearse_frac)
        finally:
            F.THAI_ROOT = original      # คืนค่าเสมอ ไม่งั้นรอบถัดไปใช้ราก subset เดิม


if __name__ == "__main__":
    main()
