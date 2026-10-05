"""
เทรนโมเดลบนชุดข้อมูลถนนไทย — รองรับสองโหมด

  fine-tune : เริ่มจาก weights ที่เทรนมาจาก CULane แล้วเทรนต่อด้วยไทย (lr ต่ำ)
  scratch   : เริ่มจาก encoder ที่ pretrain ImageNet มาเท่านั้น ไม่เคยเห็นเส้นเลนมาก่อน

ทำไมต้องเขียนไฟล์นี้แยก: train() ในโน้ตบุ๊กโหลดต่อได้จาก checkpoint ของตัวเองเท่านั้น
ใส่ weights ตั้งต้นจากไฟล์อื่นไม่ได้ ซึ่งเป็นหัวใจของการ fine-tune

ใช้:
  python finetune_thai.py --mode finetune --init models/best_model_resnet50_ssl_bs16_ep30.pth
  python finetune_thai.py --mode scratch

แก้ปัญหา fine-tune ลืมกลางคืน (Thai train มี night แค่ 11%) มีสามปุ่ม ใช้รวมกันได้:
  --freeze-encoder      เทรนแค่ decoder + head
  --balance-cats        ถ่วงน้ำหนักให้ทุกหมวดถูกหยิบเท่ากัน
  --rehearse-frac 0.5   ผสมภาพ CULane เข้าไปครึ่งหนึ่งของแต่ละ batch
  --slope-loss-weight L เอา "มุมของเส้น" เข้าไปใน loss จริง (ดู slope_loss.py)
ทั้งหมดเป็น opt-in ไม่ใส่แล้วพฤติกรรมเหมือนเดิมทุกประการ ผลเก่าจึงยังทำซ้ำได้
"""
import argparse
import csv
import os
import sys

import cv2
cv2.setNumThreads(0)      # DataLoader แตก worker อยู่แล้ว อย่าให้ cv2 แย่ง CPU (เครื่องนี้ 4 cores)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler
from tqdm import tqdm
import segmentation_models_pytorch as smp

sys.path.append('.')
from dataset import LaneDatasetFromCSV
from slope_metrics import extract_lanes, slope_metrics_for_lanes
from slope_loss import slope_loss   # ใช้เฉพาะตอน slope_loss_weight > 0

THAI_ROOT = "thai_road_lane"
CULANE_ROOT = "/home/jovyan/shared/donut/CarDashCam/CULane"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        # .float() กัน fp16 overflow ตอนรวมค่า (เหมือนในโน้ตบุ๊ก ห้ามตัดออก)
        inputs = torch.sigmoid(inputs.float()).view(-1)
        targets = targets.float().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


def build_model(encoder_name, init_path=None):
    """
    สร้างโมเดล ถ้าให้ init_path มาก็โหลด weights ทับ

    สถาปัตยกรรมเดาจาก key ในไฟล์ ไม่เชื่อชื่อไฟล์ (เหมือน load_seg_model ในโน้ตบุ๊ก)
    """
    if init_path:
        obj = torch.load(init_path, map_location="cpu", weights_only=False)
        state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        arch = "DeepLabV3Plus" if any("decoder.aspp" in k for k in state) else "Unet"
        model = getattr(smp, arch)(encoder_name=encoder_name, encoder_weights=None,
                                   in_channels=3, classes=1)
        model.load_state_dict(state)
        print(f"🔁 fine-tune จาก {os.path.basename(init_path)} ({arch} + {encoder_name})")
    else:
        # encoder_weights="imagenet" คือจุดเริ่มเดียวกับตอนเทรน CULane รอบแรก
        model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet",
                         in_channels=3, classes=1)
        print(f"🆕 เทรนใหม่จาก ImageNet encoder (Unet + {encoder_name})")
    return model.to(DEVICE)


@torch.inference_mode()
def evaluate(model, loader, names, cat_of, threshold=0.5):
    """วัด IoU / F1 / มุมคลาด / lane_recall แยกตามหมวดถนน"""
    model.eval()
    acc = {}

    def slot(s):
        return acc.setdefault(s, dict(tp=0., fp=0., fn=0., inter=0., uni=0.,
                                      err=[], gt=0, match=0, img=0))

    k = 0
    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        pred = (torch.sigmoid(model(images).float())[:, 0] > threshold).cpu().numpy()
        gt = (masks[:, 0] > 0.5).numpy()
        for b in range(len(pred)):
            cat = cat_of[names[k]]
            k += 1
            P, T = pred[b], gt[b]
            res = slope_metrics_for_lanes(extract_lanes(T), extract_lanes(P))
            for s in ("ALL", cat):
                a = slot(s)
                a["img"] += 1
                a["tp"] += (P & T).sum()
                a["fp"] += (P & ~T).sum()
                a["fn"] += (~P & T).sum()
                a["inter"] += (P & T).sum()
                a["uni"] += (P | T).sum()
                a["gt"] += res["n_gt"]
                a["match"] += res["n_matched"]
                a["err"] += [d["err"] for d in res["pairs"]]

    out = {}
    for s, a in acc.items():
        e = np.array(a["err"]) if a["err"] else np.array([np.nan])
        out[s] = dict(n_img=a["img"],
                      iou=a["inter"] / max(a["uni"], 1),
                      f1=2 * a["tp"] / max(2 * a["tp"] + a["fp"] + a["fn"], 1),
                      median_err=float(np.median(e)),
                      acc5=float(np.mean(e <= 5.0)),
                      lane_recall=a["match"] / a["gt"] if a["gt"] else float("nan"))
    return out


def set_seed(seed):
    """ตรึง seed ทุกตัว — ใช้เพื่อรัน control ซ้ำด้วย seed ต่างกันอย่างตั้งใจ
    ไม่ใช่เพื่อบังคับให้ผลเท่ากันเป๊ะ (cudnn ยังไม่ deterministic)"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def culane_rehearsal_subset(n_per_cat):
    """
    หยิบภาพ CULane มาเตือนความจำ หมวดละ n_per_cat ภาพ ด้วย stride คงที่

    เอาหมวดละเท่าๆ กัน ไม่ใช่ตามสัดส่วนเดิมของ CULane (normal 63%) เพราะที่ต้องกัน
    ไม่ให้ลืมคือกลางคืน ถ้าตามสัดส่วนเดิมจะได้ night น้อยตามไปด้วย

    stride ไม่ใช่สุ่ม เพราะต้องได้ชุดเดิมทุกรัน ไม่งั้นเพิ่ม noise ให้งานที่ noise เป็นปัญหาอยู่แล้ว
    """
    ds = LaneDatasetFromCSV(os.path.join(CULANE_ROOT, "train_list.csv"),
                            os.path.join(CULANE_ROOT, "images"),
                            os.path.join(CULANE_ROOT, "masks"), augment=True)
    bycat = {}
    for i, (_, c) in enumerate(ds.samples):
        bycat.setdefault(c, []).append(i)
    picks = []
    for c in sorted(bycat):
        idxs = bycat[c]
        step = max(len(idxs) // n_per_cat, 1)
        picks += idxs[::step][:n_per_cat]
    return Subset(ds, sorted(picks))


def run(mode, init=None, encoder="resnet50", epochs=30, batch_size=16, lr=None, tag=None,
        freeze_encoder=False, balance_cats=False, rehearse_frac=0.0, seed=None,
        slope_loss_weight=0.0):
    """
    เทรนแล้ววัดผลบนไทย คืน dict ผลแยกตามหมวด

    แยกออกมาจาก main() เพื่อให้โน้ตบุ๊กเรียกใช้โค้ดชุดเดียวกับที่รันจาก command line
    ไม่ต้องก๊อปโค้ดไปไว้ในเซลล์ (ตามแบบ slope_metrics.py ที่ CLAUDE.md กำหนด)
    """
    if mode not in ("finetune", "scratch"):
        raise ValueError(f"mode ต้องเป็น finetune หรือ scratch ไม่ใช่ {mode!r}")
    if mode == "finetune" and not init:
        raise ValueError("โหมด finetune ต้องระบุ init (path ของ .pth ตั้งต้น)")
    if mode == "scratch" and init:
        raise ValueError("โหมด scratch ห้ามระบุ init — มันจะไม่ใช่การเทรนใหม่แล้ว")
    if slope_loss_weight < 0:
        raise ValueError(f"slope_loss_weight ต้องไม่ติดลบ ไม่ใช่ {slope_loss_weight}")
    if not 0.0 <= rehearse_frac < 1.0:
        raise ValueError(f"rehearse_frac ต้องอยู่ใน [0,1) ไม่ใช่ {rehearse_frac}")
    if freeze_encoder and mode == "scratch":
        raise ValueError("โหมด scratch ห้าม freeze encoder — encoder ยังไม่เคยเห็นเส้นเลน")
    if seed is not None:
        set_seed(seed)

    # lr ของ fine-tune ต้องต่ำกว่ามาก ไม่งั้นลืม feature จาก CULane จนกลายเป็น scratch กลายๆ
    if lr is None:
        lr = 1e-5 if mode == "finetune" else 1e-4
    tag = tag or f"{mode}_{encoder}"

    class _A:
        pass
    args = _A()
    args.mode, args.init, args.encoder = mode, init, encoder
    args.epochs, args.batch_size = epochs, batch_size

    print(f"🚀 {DEVICE.type.upper()} | mode={mode} | encoder={encoder} "
          f"| lr={lr} | epochs={epochs} | bs={batch_size}")

    img_dir = os.path.join(THAI_ROOT, "images")
    mask_dir = os.path.join(THAI_ROOT, "masks")
    ds = {s: LaneDatasetFromCSV(os.path.join(THAI_ROOT, f"{s}_list.csv"), img_dir, mask_dir,
                                augment=(s == "train"))
          for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        print(f"   {s:5s} {len(ds[s])} รูป {ds[s].category_counts()}")

    lk = dict(num_workers=4, pin_memory=(DEVICE.type == "cuda"))
    if lk["num_workers"] > 0:
        lk.update(persistent_workers=True, prefetch_factor=2)

    # balance_cats กับ rehearse_frac เป็นเรื่องเดียวกัน คือ "ถ่วงน้ำหนักการสุ่ม"
    # จึงใช้ WeightedRandomSampler ตัวเดียวคุมทั้งคู่
    n_thai = len(ds["train"])
    train_ds, train_kw = ds["train"], dict(shuffle=True)
    if balance_cats or rehearse_frac > 0:
        cnt = ds["train"].category_counts()
        # น้ำหนักฝั่งไทยรวมกันได้ (1 - rehearse_frac) เสมอ ไม่ว่าจะ balance หรือไม่
        if balance_cats:
            w_thai = [1.0 / cnt[c] for _, c in ds["train"].samples]
        else:
            w_thai = [1.0] * n_thai
        w_thai = [w / sum(w_thai) * (1.0 - rehearse_frac) for w in w_thai]
        weights = list(w_thai)

        if rehearse_frac > 0:
            # หมวดละ 1/3 ของจำนวนภาพไทย ทำให้ pool ของ CULane ใหญ่พอไม่ซ้ำรูปเดิมทุก epoch
            reh = culane_rehearsal_subset(max(n_thai // 3, 1))
            train_ds = ConcatDataset([ds["train"], reh])
            weights += [rehearse_frac / len(reh)] * len(reh)
            print(f"   เตือนความจำด้วย CULane {len(reh)} ภาพ (สัดส่วนใน batch {rehearse_frac:.0%})")

        # num_samples คงที่เท่าจำนวนภาพไทยเสมอ — จำนวน step ต่อ epoch จึงเท่า control
        # ไม่งั้น "epoch 3" ของแต่ละการทดลองคนละความหมายกัน เทียบไม่ได้
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        train_kw = dict(shuffle=False,
                        sampler=WeightedRandomSampler(weights, num_samples=n_thai,
                                                      replacement=True, generator=gen))

    dl = {"train": DataLoader(train_ds, batch_size=args.batch_size, **train_kw, **lk),
          "val":   DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False, **lk),
          "test":  DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False, **lk)}

    meta = {s: list(csv.DictReader(open(os.path.join(THAI_ROOT, f"{s}_list.csv"))))
            for s in ("val", "test")}
    names = {s: [r["filename"] for r in meta[s]] for s in meta}
    cat_of = {r["filename"]: r["category"] for s in meta for r in meta[s]}

    model = build_model(args.encoder, args.init)
    if freeze_encoder:
        for p_ in model.encoder.parameters():
            p_.requires_grad = False
        n_frozen = sum(1 for p_ in model.encoder.parameters())
        print(f"   ❄️  freeze encoder {n_frozen} tensor (decoder + head เท่านั้นที่เทรน)")
    bce, dice = nn.BCEWithLogitsLoss(), DiceLoss()
    # กรองพารามิเตอร์ที่ freeze ออกจาก optimizer — requires_grad=False อย่างเดียวไม่พอ
    # Adam จะยังจอง state ให้มัน และโค้ดอ่านแล้วเข้าใจผิดว่ายังเทรนอยู่
    opt = optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=lr)

    os.makedirs("results/thai", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    hist_path = f"results/thai/history_{tag}.csv"
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_f1", "val_f1",
                                "val_iou", "val_median_err", "val_lane_recall",
                                "val_night_f1", "val_night_median_err", "val_night_recall",
                                "val_curve_median_err", "val_normal_median_err",
                                "thai_imgs_seen", "train_slope_loss", "train_slope_pairs"])

    # เซฟ checkpoint สามเกณฑ์พร้อมกันในรันเดียว
    # เพราะวัดแล้วว่า val_f1 อย่างเดียวเลือก epoch ที่ทิศทางแย่ที่สุด:
    # fine-tune รอบแรกได้ ep5 (มุมคลาด 4.65° recall 0.781) ขณะ ep1-2 ได้ 1.86° / 0.925
    # เก็บทั้งสามแล้ววัดบน test เทียบกัน จะได้รู้ว่าเกณฑ์ไหนให้โมเดลดีกว่าจริง
    CRITERIA = {
        "f1":     dict(key="f1",          better=max, best=-1.0, ep=0),
        "slope":  dict(key="median_err",  better=min, best=float("inf"), ep=0),
        "recall": dict(key="lane_recall", better=max, best=-1.0, ep=0),
    }
    for name in CRITERIA:
        CRITERIA[name]["path"] = f"models/thai_{tag}_best_{name}.pth"

    thai_seen = 0.0
    for ep in range(args.epochs):
        model.train()
        if freeze_encoder:
            # จำเป็นต่างหากจาก requires_grad=False — BatchNorm อัปเดต running stats
            # ของตัวเองโดยไม่ผ่าน gradient encoder จะเพี้ยนตามข้อมูลไทยอยู่ดี
            model.encoder.eval()
        tot_loss = tp = fp = fn = 0.0
        sl_sum, sl_pairs = 0.0, 0
        for images, masks in tqdm(dl["train"], desc=f"Epoch {ep+1}/{args.epochs}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(images)
            loss = bce(out, masks) + dice(out, masks)
            if slope_loss_weight > 0:
                # จับคู่เส้นโดยไม่เอา gradient แล้วคิดมุมแบบมี gradient (ดู slope_loss.py)
                # ถ้าไม่มีคู่ไหนจับได้ sl เป็น 0 ที่ยังต่อ graph อยู่ backward จึงไม่พัง
                sl, npair = slope_loss(out, masks)
                loss = loss + slope_loss_weight * sl
                sl_sum += float(sl.detach()); sl_pairs += npair
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            with torch.no_grad():
                p = torch.sigmoid(out.float()) > 0.5
                t = masks > 0.5
                tp += (p & t).sum().item()
                fp += (p & ~t).sum().item()
                fn += (~p & t).sum().item()

        train_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        v_all = evaluate(model, dl["val"], names["val"], cat_of)
        v = v_all["ALL"]
        # จำนวนภาพไทยที่เห็นสะสม — ตอน rehearse ครึ่ง batch เป็น CULane เลข epoch
        # จึงไม่ใช่หน่วยที่เทียบข้ามการทดลองได้ ต้องใช้เลขนี้เป็นแกน x แทน
        thai_seen += n_thai * (1.0 - rehearse_frac)

        def cat(scope, key):
            return v_all[scope][key] if scope in v_all else float("nan")

        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, tot_loss / len(dl["train"]), train_f1,
                                    v["f1"], v["iou"], v["median_err"], v["lane_recall"],
                                    cat("night", "f1"), cat("night", "median_err"),
                                    cat("night", "lane_recall"),
                                    cat("curve", "median_err"), cat("normal", "median_err"),
                                    thai_seen,
                                    sl_sum / len(dl["train"]), sl_pairs])
        marks = []
        for name, c in CRITERIA.items():
            val = v[c["key"]]
            # ข้าม NaN (เกิดได้ถ้า epoch นั้นไม่มีเส้นไหนจับคู่ได้เลย)
            if val != val:
                continue
            if c["better"](val, c["best"]) == val and val != c["best"]:
                c["best"], c["ep"] = val, ep + 1
                torch.save(model.state_dict(), c["path"])
                marks.append(name)
        mark = ("  <- best " + "/".join(marks)) if marks else ""
        print(f"[{ep+1:02d}/{args.epochs}] loss {tot_loss/len(dl['train']):.4f} "
              f"train_f1 {train_f1:.4f} | val_f1 {v['f1']:.4f} "
              f"val_มุม {v['median_err']:.2f}° | กลางคืน {cat('night','median_err'):.2f}° "
              f"recall {cat('night','lane_recall'):.3f}"
              f" gap {train_f1 - v['f1']:+.3f}{mark}")

    print("\n🏆 checkpoint ที่เลือกได้จากแต่ละเกณฑ์")
    for name, c in CRITERIA.items():
        print(f"   {name:7s} epoch {c['ep']:2d}  (val {c['key']} = {c['best']:.4f})")

    # วัดทั้งสาม checkpoint บน test ชุดเดียวกัน แล้วเขียนไฟล์เทียบกัน
    all_res = {}
    for name, c in CRITERIA.items():
        model.load_state_dict(torch.load(c["path"], map_location=DEVICE, weights_only=False))
        all_res[name] = evaluate(model, dl["test"], names["test"], cat_of)

    cmp_csv = f"results/thai/{tag}_by_criterion.csv"
    with open(cmp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "epoch", "scope", "n_img", "iou", "f1",
                    "median_err", "acc5", "lane_recall"])
        for name, c in CRITERIA.items():
            for s in ("ALL", "normal", "curve", "night"):
                if s not in all_res[name]:
                    continue
                r = all_res[name][s]
                w.writerow([name, c["ep"], s, r["n_img"], r["iou"], r["f1"],
                            r["median_err"], r["acc5"], r["lane_recall"]])

    print(f"\n📊 เทียบสามเกณฑ์บนไทย test {all_res['f1']['ALL']['n_img']} รูป")
    print(f"{'เกณฑ์':10s}{'ep':>4s}{'IoU':>8s}{'F1':>8s}{'มุมคลาด':>10s}{'recall':>9s}")
    for name, c in CRITERIA.items():
        r = all_res[name]["ALL"]
        print(f"  {name:8s}{c['ep']:4d}{r['iou']:8.3f}{r['f1']:8.3f}"
              f"{r['median_err']:9.2f}°{r['lane_recall']:9.3f}")

    # ไฟล์หลักยังใช้เกณฑ์ f1 เหมือนเดิม เพื่อไม่ให้ผลที่รายงานไปแล้วเปลี่ยนความหมาย
    res = all_res["f1"]
    out_csv = f"results/thai/{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "n_img", "iou", "f1", "median_err", "acc5", "lane_recall"])
        for s in ("ALL", "normal", "curve", "night"):
            if s not in res:
                continue
            r = res[s]
            w.writerow([s, r["n_img"], r["iou"], r["f1"], r["median_err"],
                        r["acc5"], r["lane_recall"]])

    print(f"\n📊 แยกหมวด (เกณฑ์ f1 — ไฟล์หลัก {tag}.csv)")
    print(f"{'ถนน':10s}{'รูป':>6s}{'IoU':>8s}{'F1':>8s}{'มุมคลาด':>10s}{'recall':>9s}")
    for s in ("ALL", "normal", "curve", "night"):
        if s not in res:
            continue
        r = res[s]
        print(f"  {s:8s}{r['n_img']:6d}{r['iou']:8.3f}{r['f1']:8.3f}"
              f"{r['median_err']:9.2f}°{r['lane_recall']:9.3f}")
    print(f"\n📁 {out_csv}\n📁 {cmp_csv}\n📁 {hist_path}")
    for name, c in CRITERIA.items():
        print(f"📁 {c['path']}")
    return all_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finetune", "scratch"], required=True)
    ap.add_argument("--init", default=None, help="ไฟล์ .pth ตั้งต้น (โหมด finetune เท่านั้น)")
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None,
                    help="ไม่ใส่ = 1e-5 สำหรับ finetune / 1e-4 สำหรับ scratch")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="เทรนเฉพาะ decoder + head — กัน encoder ลืม feature จาก CULane")
    ap.add_argument("--balance-cats", action="store_true",
                    help="ถ่วงน้ำหนักให้ทุกหมวดถนนถูกหยิบเท่ากัน (night 11%% เป็น ~33%%)")
    ap.add_argument("--rehearse-frac", type=float, default=0.0,
                    help="สัดส่วนภาพ CULane ที่ผสมเข้าไปเตือนความจำ เช่น 0.5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--slope-loss-weight", type=float, default=0.0,
                    help="น้ำหนักของ slope loss ที่หา gradient ได้ (0 = ปิด เหมือนเดิมทุกประการ)")
    a = ap.parse_args()
    run(mode=a.mode, init=a.init, encoder=a.encoder, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, tag=a.tag,
        freeze_encoder=a.freeze_encoder, balance_cats=a.balance_cats,
        rehearse_frac=a.rehearse_frac, seed=a.seed,
        slope_loss_weight=a.slope_loss_weight)


if __name__ == "__main__":
    main()
