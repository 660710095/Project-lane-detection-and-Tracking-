import os
import csv
import cv2
import numpy as np
import torch
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

        # Resize ด้วย OpenCV
        h, w = IMG_SIZE
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        # แปลงเป็น Tensor (shape: C, H, W)
        image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_t = (torch.from_numpy(mask).float().unsqueeze(0) > 127.0).float()

        return image_t, mask_t

    def _augment(self, image, mask):
        # สุ่มพลิกซ้าย-ขวา ทั้งภาพและ Mask พร้อมกัน (50%)
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # สุ่มปรับความสว่าง +-30% (30%)
        if np.random.rand() < 0.3:
            delta = np.random.uniform(-0.3, 0.3) * 255.0
            image = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        return image, mask

    def _augment_tensor(self, image_t, mask_t):
        """Deprecated: เก็บไว้เพื่อ backward compatibility"""
        if torch.rand(1).item() < 0.5:
            image_t = torch.flip(image_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])
        if torch.rand(1).item() < 0.3:
            delta = torch.empty(1).uniform_(-0.3, 0.3).item()
            image_t = torch.clamp(image_t + delta, 0.0, 1.0)
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