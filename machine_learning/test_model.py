import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# ค้นหาโฟลเดอร์หลักของโปรเจกต์
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ดึงฟังก์ชันแปลงมุมมองจากโฟลเดอร์ image_processing เก่าของคุณมาใช้
from image_processing.perspective import get_perspective_matrices, warp_image
from image_processing.lane_tracker import sliding_window, search_from_prior, draw_lane_area
from image_processing.lane_processor import LaneProcessor
from unet_model import UNet

def test_single_image(image_path, model_path):
    # 1. เช็คว่าใช้ GPU หรือ CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"กำลังประมวลผลด้วย: {device.type.upper()}")

    # 2. สร้างสมอง U-Net เปล่าๆ ขึ้นมา
    model = UNet(in_channels=3, out_channels=1).to(device)

    # 3. "เปิด" ไฟล์ .pth และโหลดยัดเข้าไปในสมอง
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # สั่งให้เปลี่ยนเป็น "โหมดทำข้อสอบ" (ปิดระบบเรียนรู้)

    # 4. เตรียมรูปภาพที่จะให้ AI ทำนาย
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ย่อขนาดภาพให้เท่ากับตอนที่สอนมัน (512x256)
    resized_img = cv2.resize(img, (512, 256))
    
    # แปลงภาพเป็น Tensor ที่ AI เข้าใจได้
    img_tensor = resized_img.transpose((2, 0, 1)) # สลับสีมาไว้ข้างหน้า (Channel, Height, Width)
    img_tensor = torch.from_numpy(img_tensor).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # เพิ่มมิติภาพเป็น (1, 3, 256, 512)

    # 5. สั่งให้ AI ทายผล (วาดหน้ากากถนน)
    with torch.no_grad(): # ประหยัด RAM โดยไม่ต้องจำค่าย้อนกลับ
        output = model(img_tensor)
        
        # ผลลัพธ์ที่ได้เป็นตัวเลขความน่าจะเป็น บีบให้อยู่ในช่วง 0-1 ด้วย Sigmoid
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # ถ้าจุดไหน AI มั่นใจมากกว่า 50% (>0.5) ให้ถือว่าเป็นถนน (สีขาว=1) นอกนั้นเป็นสีดำ=0
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

    # 6. แสดงผลลัพธ์บนหน้าจอเปรียบเทียบกัน
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(resized_img)
    plt.title("Original Image (Input)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("AI Predicted Mask (Output)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def test_video(video_path, model_path, output_path=None):
    # 1. เช็คว่าใช้ GPU หรือ CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"กำลังประมวลผลวิดีโอด้วย: {device.type.upper()}")

    # 2. สร้างสมอง U-Net เปล่าๆ และโหลด Weights
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. เปิดไฟล์วิดีโอ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ไม่สามารถเปิดวิดีโอได้: {video_path}")
        return
        
    # 4. เตรียมตัวบันทึกวิดีโอ (ถ้ามีการระบุชื่อไฟล์ปลายทาง)
    out = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w * 2, orig_h)) # บันทึกความกว้าง 2 เท่า (ซ้าย-ขวา)
        
    # เรียกใช้ตัวช่วยจำประวัติเส้นเลน (เพื่อลดอาการเส้นสั่น)
    lane_processor = LaneProcessor(history_length=6)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break # จบวิดีโอ
            
        orig_h, orig_w = frame.shape[:2]
        
        # ย่อขนาดและเตรียมภาพให้ AI
        resized_frame = cv2.resize(frame, (512, 256))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        img_tensor = rgb_frame.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # ให้ AI ทำนาย
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (pred_mask > 0.5).astype(np.uint8)

        # --- 1. ขยายหน้ากาก AI กลับให้เท่ากับขนาดวิดีโอต้นฉบับ ---
        # ใช้ INTER_NEAREST เพื่อให้พิกเซลยังคงเป็นแค่ 0 และ 1 (ขาว/ดำ)
        full_size_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # --- 2. แปลงหน้ากากเป็น Bird's Eye View (BEV) ---
        M, Minv, src_points = get_perspective_matrices(frame.shape)
        bev_mask = warp_image(full_size_mask, M)

        # --- 3. ประยุกต์ใช้ AI เข้ากับระบบ Tracking เดิม (Sliding Window) ---
        bev_binary = (bev_mask * 255).astype(np.uint8) # แปลงเป็น 0-255 ให้โค้ดเก่าเข้าใจ
        
        if lane_processor.detected and lane_processor.current_left_fit is not None:
            left_fit_raw, right_fit_raw, tracker_img = search_from_prior(
                bev_binary, lane_processor.current_left_fit, lane_processor.current_right_fit
            )
            if left_fit_raw is None or right_fit_raw is None:
                left_fit_raw, right_fit_raw, tracker_img = sliding_window(bev_binary)
        else:
            left_fit_raw, right_fit_raw, tracker_img = sliding_window(bev_binary)
            
        # ประมวลผลให้เส้นนิ่งขึ้นด้วย Lane Processor
        left_fit, right_fit = lane_processor.process_fits(left_fit_raw, right_fit_raw, bev_binary.shape)
        
        # วาดพื้นที่สีเขียวที่ผ่านการคำนวณสมการคณิตศาสตร์แล้ว
        if left_fit is not None and right_fit is not None:
            overlay = draw_lane_area(frame, bev_binary, left_fit, right_fit, Minv)
        else:
            overlay = frame.copy() # ถ้าหาเส้นไม่เจอเลย ให้ใช้ภาพเดิม
            
        # นำภาพมาต่อกันแนวนอน (ปรับขนาดฝั่งขวาให้เท่ากับฝั่งซ้ายก่อนนำมาต่อ)
        tracker_img_resized = cv2.resize(tracker_img, (overlay.shape[1], overlay.shape[0]))
        combined_frame = np.hstack((overlay, tracker_img_resized))
        
        # บันทึกเฟรมลงวิดีโอ
        if out:
            out.write(combined_frame)
            
        # แสดงผลบนหน้าต่าง (เพิ่มคุณสมบัติ WINDOW_NORMAL เพื่อให้คุณย่อขยายหน้าต่างได้อย่างอิสระ)
        cv2.namedWindow("AI Lane Detection (Split View)", cv2.WINDOW_NORMAL)
        cv2.imshow("AI Lane Detection (Split View)", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    if out:
        out.release()
        print(f"บันทึกวิดีโอสำเร็จที่: {output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # กำหนดตำแหน่งไฟล์ .pth ที่เราเทรนเสร็จแล้ว
    model_weights_path = os.path.join(script_dir, "unet_lane_model.pth")
    
    # สุ่มหยิบรูปภาพแรกสุดจากในโฟลเดอร์ภาพมาทดสอบ
    base_dataset_dir = os.path.join(script_dir, "dataset_for_ai") if os.path.exists(os.path.join(script_dir, "dataset_for_ai")) else os.path.join(project_root, "dataset_for_ai")
    test_image_dir = os.path.join(base_dataset_dir, "images")
    
    # ดึงชื่อไฟล์รูปภาพแรกมาทดสอบ
    sample_image_file = os.path.join(test_image_dir, os.listdir(test_image_dir)[0])
    
    # 1. ทดสอบกับรูปภาพ (คอมเมนต์ปิดไว้ก่อน)
    # test_single_image(sample_image_file, model_weights_path)
    
    # 2. ทดสอบกับวิดีโอ (เอาคอมเมนต์ออกและเปลี่ยนใส่ Path วิดีโอของคุณได้เลย)
    input_video_path = os.path.join(project_root, "data", "videos", "Video Project 2.mp4")
    output_video_path = os.path.join(script_dir, "ai_output_video.mp4")
    test_video(input_video_path, model_weights_path, output_video_path)