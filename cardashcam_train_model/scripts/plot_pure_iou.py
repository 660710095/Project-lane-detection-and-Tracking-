"""
พล็อตภาพตัวอย่าง 9 ภาพ ตัด Slope ทิ้งทั้งหมด วัดเฉพาะ IoU และ Mask
ใช้ข้อความภาษาอังกฤษเพื่อไม่ให้ฟอนต์เพี้ยนใน Matplotlib
แสดง: Original | Ground Truth | Before Fine-tune (Zero-shot) | After Fine-tune (Best F1)
"""
import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

sys.path.append('.')
from dataset import LaneDatasetFromCSV

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    ("Before Fine-tune (Zero-shot)", "models/best_model_resnet50_ssl_bs16_ep30.pth"),
    ("After Fine-tune (Best F1)",     "models/thai_ft_rh_s1_best_f1.pth"),
]

def load_model(path):
    m = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    m.load_state_dict(ckpt)
    m.to(DEVICE).eval()
    return m

def calc_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / max(union, 1)

def make_overlay(rgb, gt, pred=None):
    vis = rgb.copy()
    if pred is None:
        vis[gt] = [0, 230, 0]
        return vis
    
    gt_only = np.logical_and(gt, ~pred)
    pred_only = np.logical_and(~gt, pred)
    both = np.logical_and(gt, pred)

    vis[gt_only] = [0, 230, 0]    # Green = False Negative
    vis[pred_only] = [255, 50, 50] # Red = False Positive
    vis[both] = [255, 230, 0]     # Yellow = True Positive (IoU)
    return vis

def main():
    ds = LaneDatasetFromCSV("thai_road_lane/test_list.csv",
                            "thai_road_lane/images",
                            "thai_road_lane/masks",
                            augment=False)
    
    # สุ่มเลือกภาพแบบคงที่ (3 normal, 3 curve, 3 night)
    picks = [
        (34, "Normal", "normal__Pratunam_road_0000.jpg"),
        (86, "Normal", "normal__lumphaya_road02_0025.jpg"),
        (138, "Normal", "normal__lumphaya_road05_0039.jpg"),
        (0, "Curve",  "curve__Anusaowalee_curve01_0000.jpg"),
        (5, "Curve",  "curve__Anusaowalee_curve01_0005.jpg"),
        (11, "Curve", "curve__Chatuchak_road_curve03_0003.jpg"),
        (17, "Night", "night__sathon_road_night01_0000.jpg"),
        (22, "Night", "night__sathon_road_night01_0005.jpg"),
        (28, "Night", "night__sathon_road_night01_0011.jpg"),
    ]

    loaded_models = {name: load_model(path) for name, path in MODELS}

    rows = len(picks)
    cols = 2 + len(MODELS)  # Original, GT, Zero-shot, Fine-tune

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows))
    
    for r, (idx, cat_name, fname) in enumerate(picks):
        image_t, mask_t = ds[idx]
        rgb = np.ascontiguousarray((image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        gt = mask_t[0].numpy() > 0.5

        # 1. Original Image
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"[{cat_name}]\n{fname.split('.')[0]}", fontsize=9, pad=4)
        axes[r, 0].axis("off")

        # 2. Ground Truth
        gt_vis = make_overlay(rgb, gt)
        axes[r, 1].imshow(gt_vis)
        axes[r, 1].set_title("Ground Truth\n(Green = True Lane)", fontsize=9, pad=4)
        axes[r, 1].axis("off")

        # Inference for each model
        with torch.inference_mode():
            tensor = image_t.unsqueeze(0).to(DEVICE)
            for m_idx, (m_name, _) in enumerate(MODELS):
                model = loaded_models[m_name]
                logit = model(tensor)
                prob = torch.sigmoid(logit.float())[0, 0].cpu().numpy()
                pred = prob > 0.5

                iou = calc_iou(pred, gt)
                vis = make_overlay(rgb, gt, pred)

                axes[r, 2 + m_idx].imshow(vis)
                axes[r, 2 + m_idx].set_title(f"{m_name}\nIoU = {iou:.3f}", fontsize=10, fontweight="bold", pad=4)
                axes[r, 2 + m_idx].axis("off")

    fig.suptitle(
        "Thai Dashcam Lane Segmentation — Before vs After Fine-tuning (IoU Metric)\n"
        "Green: Ground Truth Only (FN)   |   Red: Over-predicted (FP)   |   Yellow: Overlap Match (IoU)",
        fontsize=12, y=0.995, fontweight="bold"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.5)
    
    out_dir = "results/thai"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "examples_thai_iou_clean.png")
    plt.savefig(out_path, dpi=120)
    plt.savefig("./examples_thai_iou_clean.png", dpi=120)
    plt.close(fig)
    print(f"✅ Generated clean IoU image successfully:")
    print(f"   1. {out_path}")
    print(f"   2. ./examples_thai_iou_clean.png")

if __name__ == "__main__":
    main()
