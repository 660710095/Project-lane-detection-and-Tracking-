import os
import cv2
import numpy as np

def prepare_dataset(kitti_base_dir, output_dir):
    """
    ฟังก์ชันสำหรับดึงข้อมูล Ground Truth ของชุดข้อมูล KITTI 
    และแปลงให้อยู่ในรูปแบบ Mask ขาวดำ เพื่อเตรียมเทรน AI
    """
    # 1. กำหนดโฟลเดอร์ต้นทาง
    image_dir = os.path.join(kitti_base_dir, "training", "image_2")
    gt_dir = os.path.join(kitti_base_dir, "training", "gt_image_2")
    
    # 2. กำหนดโฟลเดอร์ปลายทาง (สร้างใหม่เพื่อความสะอาด)
    out_image_dir = os.path.join(output_dir, "images")
    out_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # ดึงรายชื่อไฟล์ภาพต้นฉบับทั้งหมด
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    count = 0
    for img_name in image_files:
        # ชื่อไฟล์ของ KITTI จะเป็นรูปแบบเช่น um_000000.png
        # เราต้องแปลงเป็นชื่อไฟล์ Ground Truth คือ um_road_000000.png หรือ um_lane_000000.png
        category = img_name.split('_')[0] # เช่น 'um', 'umm', 'uu'
        idx = img_name.split('_')[1]      # เช่น '000000.png'
        
        # ในที่นี้เราจะดึงพื้นที่ถนนทั้งหมด (_road_) มาเทรน
        gt_name = f"{category}_road_{idx}"
        
        img_path = os.path.join(image_dir, img_name)
        gt_path = os.path.join(gt_dir, gt_name)
        
        if not os.path.exists(gt_path):
            print(f"ข้ามไฟล์ {img_name} เนื่องจากไม่พบไฟล์ Ground Truth ({gt_name})")
            continue
            
        # 3. โหลดภาพ
        img = cv2.imread(img_path)
        gt_img = cv2.imread(gt_path) # OpenCV โหลดภาพมาเป็น BGR (Blue, Green, Red)
        
        # 4. ดึงเฉพาะ Blue Channel ตามที่ readme.txt ระบุ (Index 0 ของ OpenCV คือ Blue)
        blue_channel = gt_img[:, :, 0]
        
        # แปลงพิกเซลที่มีค่ามากกว่า 0 ให้เป็น 255 (สีขาว) ส่วนที่เหลือเป็น 0 (สีดำ)
        _, binary_mask = cv2.threshold(blue_channel, 0, 255, cv2.THRESH_BINARY)
        
        # 5. บันทึกภาพต้นฉบับและภาพ Mask ลงในโฟลเดอร์ปลายทาง ด้วยชื่อเดียวกันเพื่อจับคู่ง่ายๆ
        # บันทึก Input (X)
        cv2.imwrite(os.path.join(out_image_dir, img_name), img)
        # บันทึก Label Mask (Y)
        cv2.imwrite(os.path.join(out_mask_dir, img_name), binary_mask)
        
        count += 1
        if count % 50 == 0:
            print(f"ประมวลผลไปแล้ว {count} ภาพ...")
            
    print(f"เสร็จสมบูรณ์! เตรียมข้อมูลสำเร็จทั้งหมด {count} ภาพ")
    print(f"เช็คไฟล์ภาพได้ที่: {out_image_dir}")
    print(f"เช็คไฟล์หน้ากาก (Mask) ได้ที่: {out_mask_dir}")

if __name__ == "__main__":
    # กำหนด Path ให้ตรงกับโครงสร้างโปรเจกต์ของคุณ
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # หาตำแหน่งโฟลเดอร์หลักของโปรเจกต์ (รองรับกรณีที่ย้ายไฟล์นี้ไปไว้ในโฟลเดอร์ machine_learning)
    if os.path.basename(script_dir) == "machine_learning":
        project_root = os.path.dirname(script_dir) # ถอยกลับ 1 ชั้น
    else:
        project_root = script_dir
        
    KITTI_BASE_DIR = os.path.join(project_root, "data", "kitti")
    
    # โฟลเดอร์ปลายทางที่เราจะเอาข้อมูลไปให้ AI เทรน
    OUTPUT_DIR = os.path.join(script_dir, "dataset_for_ai")
    
    print("กำลังเริ่มสร้าง Dataset สำหรับสอน AI...")
    prepare_dataset(KITTI_BASE_DIR, OUTPUT_DIR)