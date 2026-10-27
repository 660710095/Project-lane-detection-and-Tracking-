#!/usr/bin/env python3
"""
train_thai_2lanes.py: สคริปต์เทรนโมเดลตรวจจับเส้นเลน บังคับตีกรอบเฉพาะ 2 เลนหน้ารถ (Ego-Lane)

หลักการบังคับ 2 เลน:
------------------
1. ใช้ Ground Truth ของชุดข้อมูลไทย (Thai Road Lane) ซึ่งมีเฉพาะ 2 เส้นเลนของรถเรา (Ego-lane)
2. Loss Function (BCE + Dice) ทำโทษพิกเซลเลนข้างเคียง (Adjacent lanes) ในฐานะ False Positive
3. ปรับจูน (Fine-tune) จากโมเดลตั้งต้น CULane (ที่เคยวาด 3-4 เลน) ให้หุบเหลือเฉพาะ 2 เลนของรถเรา

วิธีใช้งาน:
----------
    # 1. เทรนโมเดลแชมป์ DeepLabV3+ ResNet-50 บังคับ 2 เลน (ค่าเริ่มต้น)
    python train_thai_2lanes.py --epochs 30 --batch-size 16

    # 2. ทดสอบระบบแบบรวดเร็ว (Smoke Test / Dry Run)
    python train_thai_2lanes.py --dry-run

    # 3. ระบุโมเดลตั้งต้น และไฟล์บันทึกเอาต์พุตเอง
    python train_thai_2lanes.py --init models/best_model_deeplabv3plus_resnet50_imagenet_bs16_ep30.pth \
                                --output models/my_thai_2lanes_best.pth \
                                --lr 5e-5
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# นำเข้าโมดูลจากในโปรเจกต์
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "data_preprocessing"))
from dataset import LaneDatasetFromCSV

import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet Normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


class DiceLoss(nn.Module):
    """Dice Loss พร้อม Laplace smoothing กันการหารด้วยศูนย์"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs.float()).view(-1)
        targets = targets.float().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


def build_model(arch="DeepLabV3Plus", encoder="resnet50", init_path=None):
    """สร้างโมเดล พร้อมตรวจจับสถาปัตยกรรมอัตโนมัติจากไฟล์ Checkpoint"""
    if init_path and os.path.exists(init_path):
        obj = torch.load(init_path, map_location="cpu", weights_only=False)
        state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        # ตรวจสอบว่ามีโมดูล ASPP ของ DeepLabV3+ หรือไม่
        detected_arch = "DeepLabV3Plus" if any("decoder.aspp" in k for k in state) else "Unet"
        model = getattr(smp, detected_arch)(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        model.load_state_dict(state)
        print(f"📦 โหลด Base Weights จาก: {os.path.basename(init_path)} ({detected_arch} + {encoder})")
    else:
        model = getattr(smp, arch)(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        print(f"🆕 สร้างโมเดลใหม่จาก ImageNet Encoder ({arch} + {encoder})")

    return model.to(DEVICE)


@torch.inference_mode()
def evaluate(model, loader, cats=None, threshold=0.5):
    """ประเมินผล IoU, Precision, Recall, F1 บนชุดข้อมูล"""
    model.eval()
    if cats is None:
        tp = fp = fn = 0
        for images, masks in loader:
            images = ((images.to(DEVICE) - MEAN) / STD).float()
            preds = (torch.sigmoid(model(images).float()) > threshold).cpu()
            targets = (masks > 0.5)
            tp += (preds & targets).sum().item()
            fp += (preds & ~targets).sum().item()
            fn += (~preds & targets).sum().item()
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        return dict(iou=iou, prec=prec, rec=rec, f1=f1)
    else:
        stats = {c: dict(tp=0, fp=0, fn=0) for c in ['ALL', 'normal', 'curve', 'night']}
        k = 0
        for images, masks in loader:
            images = ((images.to(DEVICE) - MEAN) / STD).float()
            preds = (torch.sigmoid(model(images).float()) > threshold).cpu()
            targets = (masks > 0.5)
            for b in range(len(preds)):
                c = cats[k] if k < len(cats) else "normal"
                k += 1
                p = preds[b]
                t = targets[b]
                p_tp = (p & t).sum().item()
                p_fp = (p & ~t).sum().item()
                p_fn = (~p & t).sum().item()
                for grp in ['ALL', c]:
                    stats[grp]['tp'] += p_tp
                    stats[grp]['fp'] += p_fp
                    stats[grp]['fn'] += p_fn
        results = {}
        for grp in ['ALL', 'normal', 'curve', 'night']:
            tp = stats[grp]['tp']
            fp = stats[grp]['fp']
            fn = stats[grp]['fn']
            iou = tp / max(tp + fp + fn, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            results[grp] = dict(iou=iou, prec=prec, rec=rec, f1=f1)
        return results


def main():
    parser = argparse.ArgumentParser(description="สคริปต์เทรนโมเดลบังคับเฉพาะ 2 เลนหน้ารถ (Thai 2-Lane Fine-Tuning)")
    parser.add_argument("--arch", default="DeepLabV3Plus", choices=["DeepLabV3Plus", "Unet"], help="สถาปัตยกรรมโมเดล")
    parser.add_argument("--encoder", default="resnet50", help="Backbone Encoder (เช่น resnet50, resnet34)")
    parser.add_argument("--init", default=os.path.join(PROJECT_ROOT, "models", "best_model_deeplabv3plus_resnet50_imagenet_bs16_ep30.pth"),
                        help="ไฟล์ Checkpoint ตั้งต้น (CULane pretrained เพื่อ fine-tune บังคับ 2 เลน)")
    parser.add_argument("--epochs", type=int, default=30, help="จำนวน Epoch ในการเทรน")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning Rate")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "models", "finetuned_deeplabv3plus_resnet50_2lanes_best.pth"),
                        help="ปลายทางเซฟโมเดล 2 เลนที่ดีที่สุด")
    parser.add_argument("--rehearse-culane", type=float, default=0.0,
                        help="สัดส่วนข้อมูล CULane ที่นำมาช่วยเตือนความจำกันลืม (0.0 = ไทยล้วน 100%%)")
    parser.add_argument("--dry-run", action="store_true", help="ทดสอบรัน 1 Batch เพื่อตรวจเช็คความพร้อมของระบบ")

    args = parser.parse_args()

    print("=" * 75)
    print("🚗 THAI EGO-LANE (2-LANES) FINE-TUNING PIPELINE")
    print("=" * 75)
    print(f"Target: Fine-tune model to strictly predict 2 Ego-lanes on Thai roads")
    print(f"Architecture : {args.arch} ({args.encoder})")
    print(f"Base Weights : {args.init}")
    print(f"Batch Size   : {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"Output Checkpoint : {args.output}")
    print("=" * 75)

    # 1. โหลดข้อมูลชุดถนนไทย (ที่มีเฉพาะ 2 เลน Ego-lane)
    thai_data_dir = os.path.join(PROJECT_ROOT, "dataset", "Thai_road_lane")
    train_csv = os.path.join(thai_data_dir, "train_list.csv")
    val_csv   = os.path.join(thai_data_dir, "val_list.csv")
    test_csv  = os.path.join(thai_data_dir, "test_list.csv")

    train_ds = LaneDatasetFromCSV(train_csv, os.path.join(thai_data_dir, "images"), os.path.join(thai_data_dir, "masks"), augment=True)
    val_ds   = LaneDatasetFromCSV(val_csv, os.path.join(thai_data_dir, "images"), os.path.join(thai_data_dir, "masks"), augment=False)
    test_ds  = LaneDatasetFromCSV(test_csv, os.path.join(thai_data_dir, "images"), os.path.join(thai_data_dir, "masks"), augment=False)

    print(f"📊 Thai Dataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"🎯 Target ground truth: Ego-lane only (Average 2.00 lines/image)")

    # 2. จัดการ Rehearsal ผสม CULane (หากกำหนด > 0)
    if args.rehearse_culane > 0:
        culane_root = "/home/jovyan/shared/donut/CarDashCam/CULane"
        if os.path.exists(culane_root):
            culane_csv = os.path.join(culane_root, "train_list.csv")
            culane_ds = LaneDatasetFromCSV(culane_csv, os.path.join(culane_root, "images"), os.path.join(culane_root, "masks"), augment=True)
            # สุ่มดึงภาพ CULane ตามสัดส่วน
            n_culane = int(len(train_ds) * args.rehearse_culane)
            indices = np.random.choice(len(culane_ds), n_culane, replace=False)
            culane_sub = torch.utils.data.Subset(culane_ds, indices)
            train_ds = ConcatDataset([train_ds, culane_sub])
            print(f"🔄 Rehearsal Enabled: ผสมข้อมูล CULane เข้ามา {n_culane} ภาพ (รวม {len(train_ds)} ภาพ)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_meta = pd.read_csv(test_csv)
    test_cats = test_meta["category"].tolist()

    # 3. สร้างโมเดลและ Optimizer
    model = build_model(arch=args.arch, encoder=args.encoder, init_path=args.init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    scaler = torch.amp.GradScaler('cuda')

    # 4. โหมด Smoke Test (Dry Run)
    if args.dry_run:
        print("\n🧪 Running 1-Batch Dry Run...")
        model.train()
        images, masks = next(iter(train_loader))
        images = ((images.to(DEVICE) - MEAN) / STD).float()
        masks = masks.to(DEVICE).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = model(images)
            loss = bce_loss(out, masks) + dice_loss(out, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"  ✓ Forward & Backward Success! Loss: {loss.item():.4f}")
        val_res = evaluate(model, val_loader)
        print(f"  ✓ Validation Step Success! Val IoU: {val_res['iou']*100:.2f}% | Val F1: {val_res['f1']*100:.2f}%")
        print("\n✅ DRY RUN PASSED COMPLETELY! Script is fully operational.")
        return

    # 5. Training Loop
    best_val_f1 = -1.0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    history = []

    print("\n🚀 เริ่มต้นการฝึกสอนโมเดลบีบ 2 เลน...")
    for ep in range(args.epochs):
        model.train()
        tot_loss = 0.0
        t0 = time.time()

        for images, masks in train_loader:
            images = ((images.to(DEVICE) - MEAN) / STD).float()
            masks = masks.to(DEVICE).float()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                out = model(images)
                loss = bce_loss(out, masks) + dice_loss(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item()

        epoch_time = time.time() - t0
        avg_loss = tot_loss / len(train_loader)

        # วัดผลทุก Epoch บน Validation Set
        val_res = evaluate(model, val_loader)
        is_best = val_res['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_res['f1']
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch": args.arch,
                "encoder": args.encoder,
                "epoch": ep + 1,
                "val_f1": best_val_f1,
                "val_iou": val_res['iou']
            }, args.output)
            star = " ★ [BEST MODEL SAVED]"
        else:
            star = ""

        print(f"Epoch {ep+1:02d}/{args.epochs:02d} ({epoch_time:.1f}s) | Train Loss: {avg_loss:.4f} | "
              f"Val IoU: {val_res['iou']*100:.2f}% | Val F1: {val_res['f1']*100:.2f}%{star}", flush=True)

        history.append({
            "epoch": ep + 1,
            "loss": avg_loss,
            "val_iou": val_res['iou'],
            "val_f1": val_res['f1']
        })

    print(f"\n🎉 ฝึกสอนเสร็จสิ้น! บันทึกโมเดล 2 เลนตัวจริงที่: {args.output}")

    # 6. ประเมินผลบนชุดทดสอบทางการถนนไทย (Test Set 191 ภาพ)
    print("\n" + "=" * 75)
    print("📊 การประเมินผลบนชุดทดสอบถนนไทยทางการ (Test Set 191 ภาพ)")
    print("=" * 75)
    ckpt = torch.load(args.output, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)

    test_res = evaluate(model, test_loader, cats=test_cats)
    print(f"🏆 ALL (รวมทุกหมวด 191 ภาพ):")
    print(f"   IoU: {test_res['ALL']['iou']*100:.2f}% | F1: {test_res['ALL']['f1']*100:.2f}% | "
          f"Precision: {test_res['ALL']['prec']*100:.2f}% | Recall: {test_res['ALL']['rec']*100:.2f}%")
    print("-" * 75)
    for cat in ["normal", "curve", "night"]:
        print(f"   {cat.upper():7s} -> IoU: {test_res[cat]['iou']*100:.2f}% | F1: {test_res[cat]['f1']*100:.2f}% | "
              f"Prec: {test_res[cat]['prec']*100:.2f}% | Rec: {test_res[cat]['rec']*100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
