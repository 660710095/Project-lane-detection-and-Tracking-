import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class KittiLaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(512, 256)):
        """
        คลาสสำหรับโหลดข้อมูล KITTI เพื่อป้อนให้ AI (PyTorch)
        img_size: ขนาด (Width, Height) ที่ต้องการย่อภาพก่อนเทรน
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # ดึงรายชื่อไฟล์ทั้งหมด (เรียงตามตัวอักษรเพื่อจับคู่ให้ตรงกัน)
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        # บอก AI ว่ามีข้อมูลให้เรียนรู้ทั้งหมดกี่ภาพ
        return len(self.images)
        
    def __getitem__(self, idx):
        # ฟังก์ชันนี้ AI จะเรียกใช้เพื่อขอดึงข้อมูลทีละภาพ
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # 1. โหลดภาพ (OpenCV โหลดมาเป็น BGR เราต้องแปลงเป็น RGB ให้สีตรงกับความจริง)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. โหลดภาพ Mask เฉลย (โหลดเป็นขาวดำเลย)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. ย่อขนาดภาพให้เล็กลงเท่าๆ กัน เพื่อให้เทรนได้เร็วขึ้น
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        # 4. แปลงภาพให้เป็นรูปแบบ Tensor ที่ PyTorch ต้องการ
        # ภาพสีปกติจะมีรูปทรง (Height, Width, Channel) PyTorch ต้องการ (Channel, Height, Width)
        image = image.transpose((2, 0, 1))
        # แปลงเป็น Tensor และหาร 255 เพื่อปรับค่าให้อยู่ในช่วง 0.0 - 1.0 (Normalization)
        image = torch.from_numpy(image).float() / 255.0
        
        # Mask พื้นที่ถนน ปรับค่าให้อยู่ในช่วง 0.0 - 1.0 เช่นกัน (0=ไม่ใช่ถนน, 1=ใช่ถนน)
        mask = torch.from_numpy(mask).float() / 255.0
        mask = mask.unsqueeze(0) # เพิ่มมิติให้ Mask เป็น (1, Height, Width)
        
        return image, mask

# ================= ส่วนทดสอบโค้ด =================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "dataset_for_ai", "images")
    mask_dir = os.path.join(script_dir, "dataset_for_ai", "masks")
    
    # ลองจำลองเรียกใช้งาน Dataset
    dataset = KittiLaneDataset(image_dir, mask_dir)
    print(f"เตรียมข้อมูลสำเร็จ! มีทั้งหมด {len(dataset)} รูป")
    
    # ขอดึงรูปที่ 0 (รูปแรกสุด) มาดู
    img, mask = dataset[0]
    
    # แสดงผลรูปทรงของ Tensor
    print(f"รูปทรงของภาพ (Image Shape): {img.shape} -> (Channel, Height, Width)")
    print(f"รูปทรงของเฉลย (Mask Shape): {mask.shape} -> (Channel, Height, Width)")
    
    print("\nทุกอย่างทำงานถูกต้อง! โค้ดนี้พร้อมนำไปเชื่อมกับสมองกล Neural Network แล้ว")