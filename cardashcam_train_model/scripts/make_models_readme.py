"""
สร้าง models/README.md — ตารางบอกว่าแต่ละ .pth คืออะไร

ทำไมต้องมี: ชื่อไฟล์ในโปรเจกต์นี้ **ไม่บอกสถาปัตยกรรม** — `unet_lane_model_bs32_ep30.pth`
เป็น resnet34 U-Net แต่ชื่อไม่มี token ของ encoder เลย และไม่บอกว่าต้อง normalize หรือไม่
ซึ่งถ้าใส่ผิดตัวเลขจะเพี้ยนเงียบๆ (วัดแล้ว F1 0.676 เทียบ 0.414)

ทุกคอลัมน์อ่านจากไฟล์จริง ไม่เชื่อชื่อไฟล์ ยกเว้น normalize ซึ่งอ่านจากผลที่วัดไว้แล้ว

generate ใหม่ได้ตลอด: python3 make_models_readme.py
"""
import csv
import glob
import os
import time

import torch

MODELS_DIR = "models"
OUT = os.path.join(MODELS_DIR, "README.md")

# วัดแล้วทุกตัว: ไม่มี checkpoint ไหนในโปรเจกต์นี้ที่เทรนด้วย ImageNet normalization
# (ตรวจโดยให้คะแนนทั้งสองแบบบน val 200 ภาพ ช่องว่าง F1 ~0.70 เทียบ 0.004-0.40 ไม่มีทางเดาผิด)
NORMALIZE = False


def inspect(path):
    """อ่าน state_dict แล้วบอกสถาปัตยกรรม / encoder / รูปแบบไฟล์ โดยไม่เชื่อชื่อไฟล์"""
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        # ไฟล์เก่าบางตัว serialize คนละแบบ mmap ใช้ไม่ได้ ยอมโหลดเต็มทีละไฟล์
        obj = torch.load(path, map_location="cpu", weights_only=False)

    wrapped = isinstance(obj, dict) and "model_state_dict" in obj
    state = obj["model_state_dict"] if wrapped else obj
    keys = list(state.keys())

    if any("decoder.aspp" in k for k in keys):
        arch = "DeepLabV3Plus"
    elif any("decoder.blocks" in k for k in keys):
        arch = "Unet"
    else:
        arch = "?"

    # BasicBlock มี conv1/conv2 ต่อบล็อก, Bottleneck มี conv1/conv2/conv3
    # resnet34 = BasicBlock, resnet50 = Bottleneck — แยกได้จากการมี layer1.0.conv3
    if any(k.startswith("encoder.layer1.0.conv3") for k in keys):
        block, enc = "Bottleneck", "resnet50"
    elif any(k.startswith("encoder.layer1.0.conv2") for k in keys):
        block, enc = "BasicBlock", "resnet34"
    else:
        block, enc = "?", "?"

    nblocks = []
    for li in (1, 2, 3, 4):
        ids = {k.split(".")[2] for k in keys if k.startswith(f"encoder.layer{li}.")}
        nblocks.append(len(ids))

    epoch = obj.get("epoch") if wrapped else None
    return dict(arch=arch, encoder=enc, block=block, layers="/".join(map(str, nblocks)),
                wrapped=wrapped, epoch=epoch, n_keys=len(keys))


def find_metrics(stem):
    """หาแถว ALL จาก CSV ที่ชื่อพาดพิงถึง checkpoint นี้ ไม่เจอก็เว้น ไม่เดา"""
    for pat in (f"results/**/*{stem}*.csv",):
        for path in sorted(glob.glob(pat, recursive=True)):
            try:
                rows = list(csv.DictReader(open(path)))
            except Exception:
                continue
            for r in rows:
                if r.get("scope") == "ALL" and "f1" in r:
                    return os.path.relpath(path), r
    return None, None


def main():
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")))
    print(f"ตรวจ {len(paths)} ไฟล์...")

    rows = []
    for p in paths:
        st = os.stat(p)
        info = inspect(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        src, met = find_metrics(stem)
        rows.append(dict(name=os.path.basename(p), mb=st.st_size / 1e6,
                         date=time.strftime("%Y-%m-%d", time.localtime(st.st_mtime)),
                         src=src, met=met, **info))
        print(f"  {os.path.basename(p):58s} {info['arch']:14s} {info['encoder']}")

    lines = [
        "# `models/` — ไฟล์น้ำหนักทั้งหมด",
        "",
        "> **ไฟล์นี้ generate จากสคริปต์ อย่าแก้มือ** — สร้างใหม่ด้วย `python3 make_models_readme.py`",
        f"> อัปเดตล่าสุด {time.strftime('%Y-%m-%d %H:%M')} · {len(rows)} ไฟล์ · "
        f"{sum(r['mb'] for r in rows)/1000:.1f} GB",
        "",
        "ชื่อไฟล์ในโปรเจกต์นี้ **ไม่บอกสถาปัตยกรรม** เช่น `unet_lane_model_bs32_ep30.pth`",
        "เป็น resnet34 U-Net ทั้งที่ชื่อไม่มี token ของ encoder เลย",
        "คอลัมน์ข้างล่างจึงอ่านจาก state_dict จริง ไม่ใช่จากชื่อ",
        "",
        "**ทุกไฟล์ต้องใช้ `normalize=False`** (input ดิบช่วง `[0,1]`) — วัดแล้วทั้งหมด",
        "ใส่ ImageNet normalization จะทำให้ตัวเลขพังเงียบๆ (F1 0.676 เทียบ 0.414)",
        "",
        "| ไฟล์ | สถาปัตยกรรม | encoder | layer | รูปแบบ | MB | วันที่ | F1 ที่วัดได้ | จาก |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for r in rows:
        fmt = f"checkpoint (ep {r['epoch']})" if r["wrapped"] else "state_dict"
        f1 = f"{float(r['met']['f1']):.3f}" if r["met"] else "—"
        src = f"`{r['src']}`" if r["src"] else "—"
        lines.append(f"| `{r['name']}` | {r['arch']} | {r['encoder']} | {r['layers']} | "
                     f"{fmt} | {r['mb']:.0f} | {r['date']} | {f1} | {src} |")

    lines += [
        "",
        "## อ่านคอลัมน์ยังไง",
        "",
        "- **สถาปัตยกรรม** ตรวจจาก key: `decoder.aspp.*` = DeepLabV3Plus · `decoder.blocks.*` = Unet",
        "- **encoder** ตรวจจากการมี `encoder.layer1.0.conv3` (Bottleneck = resnet50)",
        "  หรือมีแค่ `conv2` (BasicBlock = resnet34)",
        "- **layer** จำนวนบล็อกใน layer1/2/3/4 — resnet34 ได้ `3/4/6/3`, resnet50 ก็ `3/4/6/3` เหมือนกัน",
        "  ต่างกันที่ชนิดบล็อก ไม่ใช่จำนวน",
        "- **รูปแบบ** `checkpoint` มี optimizer state และประวัติ metric ครบ (ไฟล์ที่ `train()` ใช้ resume)",
        "  ส่วน `state_dict` เป็นน้ำหนักล้วน",
        "- **F1 ที่วัดได้** จับคู่กับ CSV ใน `results/` ตามชื่อไฟล์ ถ้าหาไม่เจอจะเว้น `—` ไม่ใช่เดา",
        "  ตัวเลขนี้มาจากชุดทดสอบคนละชุดกันตามที่มา ดูคอลัมน์ **จาก** ก่อนเอาไปเทียบกัน",
        "",
        "## กับดักที่บันทึกไว้แล้ว",
        "",
        "- `best_model_*.pth` คือ **argmin ของ val_loss ไม่ใช่ argmax ของ F1** ตรวจแล้วว่าต่างกันจริง",
        "  ใน 3 จาก 4 รัน (เช่น resnet50 imagenet เก็บ ep9 แต่ F1 ดีสุดอยู่ ep16)",
        "- `thai_*_best_{f1,slope,recall}.pth` มาจากรันเดียวกัน ต่างกันแค่เกณฑ์เลือก epoch",
        "  เกณฑ์ต่างกันให้ผลต่างกันมาก — ดู `results/thai/REPORT_NIGHT.md`",
        "- รายละเอียดที่มาของแต่ละตระกูลอยู่ใน `CLAUDE.md`",
        "",
    ]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n📁 {OUT} ({len(rows)} แถว)")


if __name__ == "__main__":
    main()
