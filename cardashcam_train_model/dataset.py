import os
import csv
import cv2
import numpy as np
import torch
import kornia as K
import kornia.augmentation as K_aug
from torch.utils.data import Dataset
 
# ต้องตรงกับ resolution ที่ Antigravity CLI ใช้ตอนวาด mask (เช็คให้ตรงกับที่รายงานมา)
IMG_SIZE = (448, 800)  # (height, width) — หารด้วย 32 ลงตัว (448/32=14, 800/32=25), สัดส่วนตาม 1640x590 ต้นฉบับ
 
 
class LaneDatasetFromCSV(Dataset):
    """
    โหลดข้อมูลจาก train_list.csv / val_list.csv / test_list.csv ที่ Antigravity CLI สร้างไว้
    แทนการสุ่ม split จากทั้งโฟลเดอร์ — ป้องกัน data leakage เพราะ split ถูกกำหนดไว้ล่วงหน้า
    ตาม clip_id และ stratify ตาม category (normal/curve/night) เรียบร้อยแล้ว
 
    CSV ต้องมีคอลัมน์: filename, category, clip_id
    """
 
    def __init__(self, csv_path, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.samples = []  # list of (filename, category)
 
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['filename'], row['category']))
 
        if len(self.samples) == 0:
            raise ValueError(f"ไม่พบข้อมูลใน {csv_path} — เช็คว่าไฟล์ list ถูกสร้างถูกต้องหรือยัง")
 
        # นับจำนวนต่อ category ไว้ print ตอน sanity check ว่าสัดส่วน stratify ถูกต้องจริง
        self._category_counts = {}
        for _, cat in self.samples:
            self._category_counts[cat] = self._category_counts.get(cat, 0) + 1
 
    def category_counts(self):
        return dict(self._category_counts)
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        filename, category = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        mask_filename = filename.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"เปิดไฟล์ไม่ได้: {img_path} หรือ {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # แปลงเป็น Tensor ทันที (shape: C, H, W)
        image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)

        # Resize ด้วย Kornia (รองรับ Tensor โดยตรง)
        image_t = K.geometry.resize(image_t, IMG_SIZE, interpolation='bilinear')
        mask_t = K.geometry.resize(mask_t, IMG_SIZE, interpolation='nearest')

        mask_t = (mask_t > 127.0).float()

        if self.augment:
            image_t, mask_t = self._augment_tensor(image_t, mask_t)

        return image_t, mask_t

    def _augment_tensor(self, image_t, mask_t):
        # รวมภาพและ Mask เป็น Batch ชั่วคราวเพื่อให้ Augmentation พลิกไปในทิศทางเดียวกัน
        # Kornia มีฟังก์ชันสุ่มพลิกซ้ายขวาและปรับแสงที่รับ Input เป็น Tensor ได้เลย
        if torch.rand(1).item() < 0.5:
            image_t = K.geometry.hflip(image_t)
            mask_t = K.geometry.hflip(mask_t)

        if torch.rand(1).item() < 0.3:
            # ใช้ Kornia ปรับแสง (brightness / contrast) เบาๆ
            image_t = K.enhance.adjust_brightness(image_t, torch.empty(1).uniform_(-0.3, 0.3))
            image_t = torch.clamp(image_t, 0.0, 1.0)

        return image_t, mask_t
 
 
if __name__ == "__main__":
    # quick self-test — รันไฟล์นี้ตรงๆ เพื่อเช็คว่าอ่าน CSV และภาพได้ถูกต้องก่อนเอาไปใช้เทรนจริง
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/unified_dataset_laneline_v2/train_list.csv"
    img_dir = sys.argv[2] if len(sys.argv) > 2 else "data/unified_dataset_laneline_v2/images"
    mask_dir = sys.argv[3] if len(sys.argv) > 3 else "data/unified_dataset_laneline_v2/masks"
 
    ds = LaneDatasetFromCSV(csv_path, img_dir, mask_dir, augment=False)
    print(f"จำนวนตัวอย่างทั้งหมด: {len(ds)}")
    print(f"แยกตามหมวดหมู่: {ds.category_counts()}")
    img, mask = ds[0]
    print(f"image shape: {img.shape}, mask shape: {mask.shape}, mask unique values: {mask.unique()}")