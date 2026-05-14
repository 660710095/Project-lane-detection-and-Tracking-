import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# 0. แก้ปัญหา ModuleNotFoundError: ค้นหาโฟลเดอร์หลักของโปรเจกต์
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# นำเข้า Dataset และ Model ที่เราจัดระเบียบใหม่
from kitti_ref.kitti_dataset import KittiLaneDataset
from unet_model import UNet

# =====================================================================
# ฟังก์ชันคำนวณความแม่นยำ (IoU - Intersection over Union)
# =====================================================================
def calculate_iou(outputs, labels, threshold=0.5):
    # AI จะให้ค่าความมั่นใจมา เราต้องใช้ Sigmoid บีบให้อยู่ช่วง 0-1 
    # แล้วตัดที่ 0.5 (มากกว่า 0.5 ถือว่าเป็นถนน, น้อยกว่าถือว่าไม่ใช่)
    preds = torch.sigmoid(outputs) > threshold
    labels = labels > threshold
    
    # นับจำนวนพิกเซลที่ทายถูกทับกับเฉลยพอดี (Intersection)
    intersection = (preds & labels).float().sum((1, 2, 3))
    # นับจำนวนพิกเซลทั้งหมดของทั้งคำทายและเฉลยรวมกัน (Union)
    union = (preds | labels).float().sum((1, 2, 3))
    
    # คำนวณ IoU (บวก 1e-6 เพื่อป้องกันการเกิด Error ตัวหารเป็น 0)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# =====================================================================
# ฟังก์ชันหลักสำหรับฝึกสอน AI
# =====================================================================
def train_model():
    # 1. ตั้งค่าพื้นฐาน (Hyperparameters)
    EPOCHS = 10           # จำนวนรอบที่ให้ AI ดูรูปทั้งหมด (เริ่มที่ 10 รอบเพื่อทดสอบระบบ)
    BATCH_SIZE = 8        # สำหรับ RTX 4060 (8GB VRAM) สามารถปรับขึ้นเป็น 8 หรือ 16 ได้เพื่อให้เทรนเร็วขึ้น
    LEARNING_RATE = 1e-4  # ความเร็วในการเรียนรู้ (0.0001)
    
    # เช็คว่าคอมพิวเตอร์มีการ์ดจอ (GPU) ไหม? ถ้ามีมันจะรันเร็วขึ้น 10-20 เท่า!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"กำลังเทรนโมเดลด้วย: {device.type.upper()}")

    # 2. โหลดข้อมูล Dataset
    # ค้นหาโฟลเดอร์ dataset_for_ai ว่าอยู่ข้างใน machine_learning หรือข้างนอก
    if os.path.exists(os.path.join(script_dir, "dataset_for_ai")):
        base_dataset_dir = os.path.join(script_dir, "dataset_for_ai")
    else:
        base_dataset_dir = os.path.join(project_root, "dataset_for_ai")
        
    image_dir = os.path.join(base_dataset_dir, "images")
    mask_dir = os.path.join(base_dataset_dir, "masks")
    
    dataset = KittiLaneDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"เตรียมข้อมูลเสร็จสิ้น! มีภาพให้เรียนรู้ทั้งหมด {len(dataset)} รูป")

    # 3. สร้างโมเดลสมองกล, ตั้งค่าตัวตรวจคำตอบ (Loss), และตัวปรับน้ำหนัก (Optimizer)
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss() # สมการสำหรับวัดความผิดพลาดของภาพขาว-ดำ
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # อัลกอริทึมยอดฮิตในการปรับสมอง AI

    # 4. เริ่มลูปการเทรน (Training Loop)
    print("\n--- เริ่มกระบวนการสอน AI ---")
    for epoch in range(EPOCHS):
        model.train() # สั่งให้โมเดลอยู่ใน "โหมดนักเรียน" (พร้อมจำสิ่งใหม่ๆ)
        running_loss = 0.0
        running_iou = 0.0
        
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()            # 4.1 เคลียร์ความจำจากข้อที่แล้ว
            outputs = model(images)          # 4.2 ให้ AI ลองเดาคำตอบ (วาดสีถนน)
            loss = criterion(outputs, masks) # 4.3 คุณครูตรวจคำตอบว่าผิดไปแค่ไหน
            loss.backward()                  # 4.4 หาจุดบกพร่องในสมอง
            optimizer.step()                 # 4.5 ปรับจูนเส้นประสาทให้เก่งขึ้น
            
            running_loss += loss.item()
            running_iou += calculate_iou(outputs, masks)
            
        # สรุปผลคะแนนสอบประจำรอบ (Epoch)
        epoch_loss = running_loss / len(dataloader)
        epoch_iou = running_iou / len(dataloader)
        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] ข้อผิดพลาด (Loss): {epoch_loss:.4f} | ความแม่นยำ (IoU): {epoch_iou*100:.2f}%")
        
    # 5. เทรนเสร็จแล้ว บันทึกสมองกลเก็บไว้ใช้งาน!
    save_path = os.path.join(script_dir, "unet_lane_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nยินดีด้วย! เทรนสำเร็จและบันทึกโมเดลไว้ที่: {save_path}")

if __name__ == "__main__":
    train_model()