import os
import sys
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import segmentation_models_pytorch as smp

sys.path.append("/home/jovyan/cardashcam_train_model/data_preprocessing")
from dataset import LaneDatasetFromCSV

ROOT = "/home/jovyan/cardashcam_train_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 16
LR = 5e-5

# Datasets
train_ds = LaneDatasetFromCSV(f"{ROOT}/dataset/Thai_road_lane/train_list.csv",
                              f"{ROOT}/dataset/Thai_road_lane/images",
                              f"{ROOT}/dataset/Thai_road_lane/masks", augment=True)
val_ds   = LaneDatasetFromCSV(f"{ROOT}/dataset/Thai_road_lane/val_list.csv",
                              f"{ROOT}/dataset/Thai_road_lane/images",
                              f"{ROOT}/dataset/Thai_road_lane/masks", augment=False)
test_ds  = LaneDatasetFromCSV(f"{ROOT}/dataset/Thai_road_lane/test_list.csv",
                              f"{ROOT}/dataset/Thai_road_lane/images",
                              f"{ROOT}/dataset/Thai_road_lane/masks", augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

test_df = pd.read_csv(f"{ROOT}/dataset/Thai_road_lane/test_list.csv")
test_cats = test_df['category'].tolist()

# Loss
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs.float()).view(-1)
        targets = targets.float().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

# Normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

MODELS = [
    ("U-Net (ResNet-34, ImageNet)", "Unet", "resnet34", "unet_r34_imagenet", f"{ROOT}/models/unet_lane_model_bs32_ep30.pth"),
    ("U-Net (ResNet-50, ImageNet)", "Unet", "resnet50", "unet_r50_imagenet", f"{ROOT}/models/best_model_resnet50_imagenet_bs16_ep30.pth"),
    ("U-Net (ResNet-50, SSL)", "Unet", "resnet50", "unet_r50_ssl", f"{ROOT}/models/best_model_resnet50_ssl_bs16_ep30.pth"),
    ("DeepLabV3+ (ResNet-34, ImageNet)", "DeepLabV3Plus", "resnet34", "dlv3p_r34_imagenet", f"{ROOT}/models/final_model_deeplabv3plus_resnet34_imagenet_bs32_ep30.pth"),
    ("DeepLabV3+ (ResNet-50, ImageNet)", "DeepLabV3Plus", "resnet50", "dlv3p_r50_imagenet", f"{ROOT}/models/best_model_deeplabv3plus_resnet50_imagenet_bs16_ep30.pth"),
    ("DeepLabV3+ (ResNet-50, SSL)", "DeepLabV3Plus", "resnet50", "dlv3p_r50_ssl", f"{ROOT}/models/final_model_deeplabv3plus_resnet50_ssl_bs16_ep30.pth"),
]

@torch.inference_mode()
def evaluate_model(model, loader, cats=None):
    model.eval()
    if cats is None:
        tp = fp = fn = 0
        for images, masks in loader:
            images = ((images.to(DEVICE) - mean) / std).float()
            preds = (torch.sigmoid(model(images).float()) > 0.5).cpu()
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
            images = ((images.to(DEVICE) - mean) / std).float()
            preds = (torch.sigmoid(model(images).float()) > 0.5).cpu()
            targets = (masks > 0.5)
            for b in range(len(preds)):
                c = cats[k]
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

out_csv = f"{ROOT}/results/thai/finetune_sequential_6_models_comparison.csv"
records = []

print("="*70)
print(f"🚀 STARTING SEQUENTIAL FINE-TUNING FOR ALL 6 MODELS (EPOCHS={EPOCHS})")
print("="*70)

for disp_name, arch, enc, tag, ckpt_path in MODELS:
    print(f"\n▶ Training Model: {disp_name}")
    print(f"  Init from: {os.path.basename(ckpt_path)}")
    model = getattr(smp, arch)(encoder_name=enc, encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    best_ckpt = f"{ROOT}/models/finetuned_{tag}_best.pth"
    
    if os.path.exists(best_ckpt):
        print(f"  ⚡ Found existing checkpoint: {best_ckpt} -> Skipping training and evaluating directly!", flush=True)
    else:
        for ep in range(EPOCHS):
            model.train()
            tot_loss = 0.0
            t0 = time.time()
            for images, masks in train_loader:
                images = ((images.to(DEVICE) - mean) / std).float()
                masks = masks.to(DEVICE).float()
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    out = model(images)
                    loss = bce_loss(out, masks) + dice_loss(out, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tot_loss += loss.item()
            
            val_res = evaluate_model(model, val_loader)
            t_ep = time.time() - t0
            if val_res['f1'] > best_val_f1:
                best_val_f1 = val_res['f1']
                torch.save(model.state_dict(), best_ckpt)
                star = " ★ BEST"
            else:
                star = ""
            print(f"   Epoch {ep+1:02d}/{EPOCHS:02d} ({t_ep:.1f}s) | Loss: {tot_loss/len(train_loader):.4f} | Val IoU: {val_res['iou']*100:.2f}% | Val F1: {val_res['f1']*100:.2f}%{star}", flush=True)
    
    # Evaluate best checkpoint on test set
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    test_res = evaluate_model(model, test_loader, cats=test_cats)
    print(f"  🏆 TEST SET (191 imgs) -> IoU: {test_res['ALL']['iou']*100:.2f}% | F1: {test_res['ALL']['f1']*100:.2f}% | Precision: {test_res['ALL']['prec']*100:.2f}% | Recall: {test_res['ALL']['rec']*100:.2f}%", flush=True)
    print(f"     Normal: {test_res['normal']['iou']*100:.2f}% | Curve: {test_res['curve']['iou']*100:.2f}% | Night: {test_res['night']['iou']*100:.2f}%", flush=True)
    
    records.append({
        "Model": disp_name,
        "Architecture": arch,
        "Encoder": enc,
        "All_IoU": test_res['ALL']['iou'] * 100,
        "All_Prec": test_res['ALL']['prec'] * 100,
        "All_Rec": test_res['ALL']['rec'] * 100,
        "All_F1": test_res['ALL']['f1'] * 100,
        "Normal_IoU": test_res['normal']['iou'] * 100,
        "Curve_IoU": test_res['curve']['iou'] * 100,
        "Night_IoU": test_res['night']['iou'] * 100,
    })

# Save to CSV
df_out = pd.DataFrame(records)
df_out = df_out.sort_values(by="All_IoU", ascending=False)
df_out.to_csv(out_csv, index=False)
print("\n" + "="*70)
print(f"✅ FINISHED ALL 6 FINE-TUNING RUNS! SAVED TO: {out_csv}")
print("="*70)
print(df_out.to_string(index=False))
