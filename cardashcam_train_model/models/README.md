# `models/` — ไฟล์น้ำหนักทั้งหมด

> **ไฟล์นี้ generate จากสคริปต์ อย่าแก้มือ** — สร้างใหม่ด้วย `python3 make_models_readme.py`
> อัปเดตล่าสุด 2026-09-25 15:40 · 50 ไฟล์ · 6.7 GB

ชื่อไฟล์ในโปรเจกต์นี้ **ไม่บอกสถาปัตยกรรม** เช่น `unet_lane_model_bs32_ep30.pth`
เป็น resnet34 U-Net ทั้งที่ชื่อไม่มี token ของ encoder เลย
คอลัมน์ข้างล่างจึงอ่านจาก state_dict จริง ไม่ใช่จากชื่อ

**ทุกไฟล์ต้องใช้ `normalize=False`** (input ดิบช่วง `[0,1]`) — วัดแล้วทั้งหมด
ใส่ ImageNet normalization จะทำให้ตัวเลขพังเงียบๆ (F1 0.676 เทียบ 0.414)

| ไฟล์ | สถาปัตยกรรม | encoder | layer | รูปแบบ | MB | วันที่ | F1 ที่วัดได้ | จาก |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |
| `best_model_deeplabv3plus_resnet50_imagenet_bs16_ep30.pth` | DeepLabV3Plus | resnet50 | 3/4/6/3 | checkpoint (ep 4) | 107 | 2026-09-17 | — | — |
| `best_model_resnet34_imagenet_brightonly_bs32_ep30.pth` | Unet | resnet34 | 3/4/6/3 | checkpoint (ep 6) | 98 | 2026-09-17 | — | — |
| `best_model_resnet34_imagenet_bs32_ep30_n2000.pth` | Unet | resnet34 | 3/4/6/3 | checkpoint (ep 24) | 98 | 2026-09-09 | — | — |
| `best_model_resnet50_imagenet_bs16_ep30.pth` | Unet | resnet50 | 3/4/6/3 | checkpoint (ep 9) | 130 | 2026-09-17 | — | — |
| `best_model_resnet50_ssl_bs16_ep30.pth` | Unet | resnet50 | 3/4/6/3 | checkpoint (ep 10) | 130 | 2026-09-06 | — | — |
| `best_unet_lane_model_bs32_ep20.pth` | Unet | resnet34 | 3/4/6/3 | checkpoint (ep 11) | 98 | 2026-09-05 | — | — |
| `final_model_deeplabv3plus_resnet34_imagenet_bs32_ep30.pth` | DeepLabV3Plus | resnet34 | 3/4/6/3 | checkpoint (ep 30) | 90 | 2026-09-17 | — | — |
| `final_model_deeplabv3plus_resnet50_imagenet_bs16_ep30.pth` | DeepLabV3Plus | resnet50 | 3/4/6/3 | checkpoint (ep 30) | 107 | 2026-09-17 | — | — |
| `final_result_epoch_resnet34_imagenet_bs32_ep30_n2000.pth` | Unet | resnet34 | 3/4/6/3 | checkpoint (ep 30) | 294 | 2026-09-09 | — | — |
| `final_result_epoch_resnet50_ssl_bs16_ep30.pth` | Unet | resnet50 | 3/4/6/3 | checkpoint (ep 30) | 391 | 2026-09-07 | — | — |
| `thai_finetune_ssl_best.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-17 | — | — |
| `thai_finetune_ssl_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-21 | — | — |
| `thai_finetune_ssl_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-21 | — | — |
| `thai_finetune_ssl_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-21 | — | — |
| `thai_ft_bal_s1_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_bal_s1_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_bal_s1_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_ctrl_s1_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_ctrl_s1_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_ctrl_s1_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_ctrl_s2_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_ctrl_s2_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_ctrl_s2_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_frz_s1_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_frz_s1_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_frz_s1_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_n245_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_n245_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_n245_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_n505_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_n505_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_n505_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_rh_s1_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_rh_s1_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_rh_s1_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_rh_s2_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_rh_s2_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_rh_s2_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-22 | — | — |
| `thai_ft_sl1_s1_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-23 | — | — |
| `thai_ft_sl1_s1_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-23 | — | — |
| `thai_ft_sl1_s1_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-23 | — | — |
| `thai_ft_sl3_s1_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-23 | — | — |
| `thai_ft_sl3_s1_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-23 | — | — |
| `thai_ft_sl3_s1_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-23 | — | — |
| `thai_scratch_ssl_best_f1.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-21 | — | — |
| `thai_scratch_ssl_best_recall.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-21 | — | — |
| `thai_scratch_ssl_best_slope.pth` | Unet | resnet50 | 3/4/6/3 | state_dict | 130 | 2026-09-21 | — | — |
| `unet_lane_model.pth` | Unet | resnet34 | 3/4/6/3 | state_dict | 98 | 2026-09-05 | 0.490 | `results/thai/zeroshot_unet_lane_model_bs32_ep30.csv` |
| `unet_lane_model_bs32_ep30.pth` | Unet | resnet34 | 3/4/6/3 | state_dict | 98 | 2026-09-09 | 0.490 | `results/thai/zeroshot_unet_lane_model_bs32_ep30.csv` |
| `unet_lane_model_resnet34_imagenet_bs32_ep30_n2000.pth` | Unet | resnet34 | 3/4/6/3 | state_dict | 98 | 2026-09-09 | — | — |

## อ่านคอลัมน์ยังไง

- **สถาปัตยกรรม** ตรวจจาก key: `decoder.aspp.*` = DeepLabV3Plus · `decoder.blocks.*` = Unet
- **encoder** ตรวจจากการมี `encoder.layer1.0.conv3` (Bottleneck = resnet50)
  หรือมีแค่ `conv2` (BasicBlock = resnet34)
- **layer** จำนวนบล็อกใน layer1/2/3/4 — resnet34 ได้ `3/4/6/3`, resnet50 ก็ `3/4/6/3` เหมือนกัน
  ต่างกันที่ชนิดบล็อก ไม่ใช่จำนวน
- **รูปแบบ** `checkpoint` มี optimizer state และประวัติ metric ครบ (ไฟล์ที่ `train()` ใช้ resume)
  ส่วน `state_dict` เป็นน้ำหนักล้วน
- **F1 ที่วัดได้** จับคู่กับ CSV ใน `results/` ตามชื่อไฟล์ ถ้าหาไม่เจอจะเว้น `—` ไม่ใช่เดา
  ตัวเลขนี้มาจากชุดทดสอบคนละชุดกันตามที่มา ดูคอลัมน์ **จาก** ก่อนเอาไปเทียบกัน

## กับดักที่บันทึกไว้แล้ว

- `best_model_*.pth` คือ **argmin ของ val_loss ไม่ใช่ argmax ของ F1** ตรวจแล้วว่าต่างกันจริง
  ใน 3 จาก 4 รัน (เช่น resnet50 imagenet เก็บ ep9 แต่ F1 ดีสุดอยู่ ep16)
- `thai_*_best_{f1,slope,recall}.pth` มาจากรันเดียวกัน ต่างกันแค่เกณฑ์เลือก epoch
  เกณฑ์ต่างกันให้ผลต่างกันมาก — ดู `results/thai/REPORT_NIGHT.md`
- รายละเอียดที่มาของแต่ละตระกูลอยู่ใน `CLAUDE.md`
