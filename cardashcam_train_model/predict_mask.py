#!/usr/bin/env python3
"""
predict_mask.py: โมดูลแปลงภาพถนนเป็น Binary Mask สำหรับฝั่ง Image Processing (IP)

โมเดลหลัก: DeepLabV3+ (ResNet-50 ImageNet) จาก Joint Training (IoU 55.12%, F1 68.84%)
รับภาพ BGR (numpy array) คืนค่า Binary Mask (0/255) ขนาดเท่าภาพต้นฉบับ

วิธีใช้งานในโค้ด Python:
-----------------------
    from predict_mask import load_model, predict_mask
    import cv2

    model, device = load_model()
    frame = cv2.imread("test.jpg")
    mask = predict_mask(model, frame, device) # ได้ numpy array uint8 (0 หรือ 255)

วิธีรันผ่าน Terminal / CLI:
--------------------------
    # 1. ทำนายภาพเดี่ยว
    python predict_mask.py --input path/to/image.jpg --output path/to/mask.png

    # 2. ทำนายทั้งโฟลเดอร์ (Batch)
    python predict_mask.py --batch --csv dataset/Thai_road_lane/test_list.csv --out-dir results/ip/pred_masks_dl
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# ขนาดภาพมาตรฐานที่โมเดลใช้เทรน
IMG_SIZE = (448, 800)  # (height, width)

# ค่า ImageNet Normalization สำหรับ ResNet-50
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# โมเดลตัวแชมป์ค่าเริ่มต้น
DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "joint_culane_thai_deeplabv3plus_resnet50_imagenet_best_f1.pth"
)


def load_model(checkpoint_path=DEFAULT_CKPT, device=None):
    """
    โหลดโมเดล DeepLabV3+ (ResNet-50 ImageNet) เข้า Memory
    เรียกเพียงครั้งเดียวตอนเริ่มต้นโปรแกรม
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ Checkpoint: {checkpoint_path}")

    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"✅ โหลดโมเดล DeepLabV3+ (ResNet-50) สำเร็จจาก: {os.path.basename(checkpoint_path)} บน [{device}]")
    return model, device


@torch.inference_mode()
def predict_mask(model, frame, device=None, threshold=0.5):
    """
    รับภาพ 1 เฟรม (BGR, uint8) คืนค่า Binary Mask (0/255) ขนาดเท่าภาพต้นฉบับ

    Parameters
    ----------
    model : โมเดลจาก load_model()
    frame : numpy array (H, W, 3) BGR uint8
    device : torch.device (ถ้าไม่ระบุจะดูจากโมเดล)
    threshold : เกณฑ์ตัดสินพิกเซลเส้นเลน (ค่าเริ่มต้น 0.5)

    Returns
    -------
    numpy array (H, W) uint8 ค่า 0 (พื้นหลัง) หรือ 255 (เส้นเลน)
    """
    if device is None:
        device = next(model.parameters()).device
    orig_h, orig_w = frame.shape[:2]

    # 1. Resize เข้าสู่ขนาดโมเดล 448x800
    resized = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)

    # 2. แปลง BGR -> RGB -> Normalize ImageNet -> Tensor (1, 3, 448, 800)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    tensor = (tensor - MEAN.to(device)) / STD.to(device)

    # 3. Forward pass + Sigmoid + Threshold
    logits = model(tensor)
    prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
    mask_small = (prob > threshold).astype(np.uint8) * 255

    # 4. ขยายกลับเป็นขนาดเท่าภาพต้นฉบับ (INTER_NEAREST เพื่อรักษาขอบคมชัด)
    mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


def main():
    parser = argparse.ArgumentParser(description="ทำนาย Binary Mask จากกล้องหน้ารถด้วยโมเดล DeepLabV3+ แชมป์")
    parser.add_argument("--input", type=str, help="ที่อยู่ไฟล์ภาพต้นฉบับ (.jpg / .png)")
    parser.add_argument("--output", type=str, default="output_mask.png", help="ที่อยู่ไฟล์เซฟ Mask ปลายทาง")
    parser.add_argument("--model", type=str, default=DEFAULT_CKPT, help="ที่อยู่ไฟล์ checkpoint .pth")
    parser.add_argument("--batch", action="store_true", help="โหมดแปลงภาพชุดจากไฟล์ CSV")
    parser.add_argument("--csv", type=str, default="dataset/Thai_road_lane/test_list.csv", help="ไฟล์รายการ CSV")
    parser.add_argument("--img-dir", type=str, default="dataset/Thai_road_lane/images", help="โฟลเดอร์ภาพต้นฉบับ")
    parser.add_argument("--out-dir", type=str, default="results/ip/pred_masks_dl", help="โฟลเดอร์บันทึก Mask ปลายทาง")
    args = parser.parse_args()

    model, device = load_model(args.model)

    if args.batch:
        import pandas as pd
        from tqdm import tqdm

        if not os.path.exists(args.csv):
            print(f"❌ ไม่พบไฟล์ CSV: {args.csv}")
            return
        df = pd.read_csv(args.csv)
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"🚀 กำลังแปลงภาพทั้งหมด {len(df)} ภาพ จาก {args.csv}...")

        t0 = time.time()
        for fname in tqdm(df["filename"]):
            img_p = os.path.join(args.img_dir, fname)
            f = cv2.imread(img_p)
            if f is None:
                continue
            m = predict_mask(model, f, device)
            out_name = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(args.out_dir, out_name), m)
        total_time = time.time() - t0
        print(f"🎉 เสร็จสิ้น! บันทึก Binary Mask ครบ {len(df)} ภาพ ที่: {args.out_dir} (ใช้เวลา {total_time:.2f} วินาที)")

    elif args.input:
        if not os.path.exists(args.input):
            print(f"❌ ไม่พบภาพ: {args.input}")
            return
        frame = cv2.imread(args.input)
        t0 = time.time()
        mask = predict_mask(model, frame, device)
        ms = (time.time() - t0) * 1000

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        cv2.imwrite(args.output, mask)
        lane_px = int((mask == 255).sum())
        print(f"🎉 สำเร็จ! บันทึกผลไปที่: {args.output}")
        print(f"   ขนาดภาพ: {frame.shape[1]}x{frame.shape[0]} | เวลาประมวลผล: {ms:.1f} ms | พิกเซลเส้นเลน: {lane_px:,} px")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
