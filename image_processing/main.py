import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import load_image, load_video, save_video, get_kitti_image
from lane_processor import LaneProcessor
from edge_detection import full_pipeline



def test_single_image(img_path):
    img = load_image(img_path)
    

    # ============= downloaded from path ============
    result, edges,roi, binary_warped, tracker_img = full_pipeline(img)

    # ============= run All pipeline util result 3 sections =======
    fig, axes = plt.subplots(1,6,figsize=(30,5))
    #chanel 1  : original image tranformed to BGR --> RGB
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')

    #===========================================================
    #chanel 2 : edge detection result (black and white image)
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Edge Detection (Canny)')

#===========================================================
    #chanel 3 : region of interest (ROI) mask applied to the original image
    axes[2].imshow(roi, cmap='gray')
    axes[2].set_title('Region of Interest (ROI)')

#===========================================================
    #chanel 4 : final lane detection result with detected lanes highlighted
    # green Lines 
    axes[3].imshow(binary_warped, cmap='gray')
    axes[3].set_title("4. Bird's-Eye View")
#===========================================================
    # ภาพที่ 5: Sliding Window (การสแกนหาเส้นโค้ง)
    axes[4].imshow(cv2.cvtColor(tracker_img, cv2.COLOR_BGR2RGB))
    axes[4].set_title('5. Lane Tracker')
#===========================================================
    # ภาพที่ 6: ผลลัพธ์สุดท้าย (ระบายสีเลนถนน)
    axes[5].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[5].set_title('6. Final Result')

    plt.tight_layout()
    plt.show()

def test_video(video_path, output_path, raw_dir=None, processed_dir=None, roi_video_path=None, tracker_video_path=None):
    cap = load_video(video_path)
    # open video
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # สร้าง VideoWriter สำหรับ ROI และ Tracker (ถ้ามี)
    roi_out = None
    if roi_video_path:
        roi_out = cv2.VideoWriter(roi_video_path, fourcc, fps, (width, height))
    tracker_out = None
    if tracker_video_path:
        tracker_out = cv2.VideoWriter(tracker_video_path, fourcc, fps, (width, height))

    # ลด history_length ลงเพื่อให้เส้นตอบสนองกับทางโค้งลึกๆ ได้ทัน (แก้ปัญหาเส้นหลุด/หลอน)
    lane_processor = LaneProcessor(history_length=8)

    count = 0
    #count frames in Video


    while cap.isOpened():
        ret, frame = cap.read()
        #Read frame by frame
        # ret =  True if Read Frames successfully, False if no more frames or error
        # frame = 1 Picture --> 1 Frame

        if not ret:
            break
            
        # ถ้ามีการกำหนดโฟลเดอร์ raw_dir ให้เซฟภาพต้นฉบับลงไป
        if raw_dir:
            cv2.imwrite(os.path.join(raw_dir, f"frame_{count:04d}.png"), frame)
            
        # run pipeline for each frame
        # --- ปรับปรุง: เรียก Pipeline แค่ครั้งเดียวต่อเฟรม ---
        result, _, roi_img, _, tracker_img = full_pipeline(frame, lane_processor=lane_processor)
        out.write(result)

        # ถ้ามีการกำหนดโฟลเดอร์ processed_dir ให้เซฟภาพผลลัพธ์ลงไป
        if processed_dir:
            cv2.imwrite(os.path.join(processed_dir, f"processed_{count:04d}.png"), result)

        # บันทึกวิดีโอ ROI และ Tracker
        if roi_out:
            roi_bgr = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
            roi_out.write(roi_bgr)
        if tracker_out:
            tracker_out.write(tracker_img)

        # --- จัดเรียงภาพ 4 จอ (2x2 Grid) ---
        # 1. ย่อขนาดภาพทั้งหมดลง (เช่น 40% ของขนาดเดิม) เพื่อให้พอดีกับหน้าจอ
        scale = 0.4
        frame_display = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        result_display = cv2.resize(result, (0, 0), fx=scale, fy=scale)
        
        # 2. ย่อขนาดภาพ ROI และ Tracker สำหรับการแสดงผล
        roi_display = cv2.resize(roi_img, (0, 0), fx=scale, fy=scale)
        tracker_display = cv2.resize(tracker_img, (0, 0), fx=scale, fy=scale)

        # 3. แปลงภาพ Grayscale (1 channel) ให้เป็น BGR (3 channels) เพื่อให้ต่อกันได้
        roi_display_bgr = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
        
        # 4. ประกอบร่างเป็นตาราง 2x2
        top_row = cv2.hconcat([frame_display, result_display])
        bottom_row = cv2.hconcat([roi_display_bgr, tracker_display])
        combined_display = cv2.vconcat([top_row, bottom_row])
        
        # 5. แสดงผล
        cv2.imshow('Lane Detection Pipeline (Press Q to exit)', combined_display)
        
        # หน่วงเวลา 1 มิลลิวินาทีให้อัปเดตหน้าจอ และเช็คว่ามีการกดปุ่ม 'q' หรือไม่
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ผู้ใช้กดปุ่มหยุดการทำงาน (Q)")
            break

        count += 1
        if count % 30 == 0:
            print(f'Processed {count}/{total} frames')
        # show progress every 30 frames

    cap.release()
    out.release()
    if roi_out:
        roi_out.release()
    if tracker_out:
        tracker_out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

def test_kitti_batch(kitti_dir, result_dir):
    paths = get_kitti_image(kitti_dir)
    # Pull all images from kitti dataset

    for i, p in enumerate(paths):
        # loop for 5 pictures in kitti dataset
        img = load_image(p)
        result, _, _, _, _ = full_pipeline(img)
        output_path = os.path.join(result_dir, f"kitti_result_{i:03d}.png")
        cv2.imwrite(output_path, result)
        # บันทึกผลลัพธ์เป็นไฟล์รูป
        # i:03d แปลว่าตัวเลข 3 หลัก เช่น 000, 001, 002

def create_video_from_image_sequence(image_dir, output_video_path, fps=10):
    """
    สร้างวิดีโอจากลำดับของไฟล์รูปภาพในไดเรกทอรีที่กำหนด
    
    Args:
        image_dir (str): พาธของไดเรกทอรีที่มีไฟล์รูปภาพอยู่
        output_video_path (str): พาธและชื่อไฟล์สำหรับบันทึกวิดีโอ
        fps (int): จำนวนเฟรมต่อวินาทีของวิดีโอ (ค่าเริ่มต้นคือ 10)
    """
    # ค้นหาไฟล์รูปภาพทั้งหมดที่ขึ้นต้นด้วย 'kitti_result_' และลงท้ายด้วย '.png'
    # และเรียงลำดับตามชื่อไฟล์เพื่อให้ได้ลำดับที่ถูกต้อง
    image_files = sorted([os.path.join(image_dir, f) 
                          for f in os.listdir(image_dir) 
                          if f.startswith('kitti_result_') and f.endswith('.png')])

    if not image_files:
        print(f"ไม่พบไฟล์รูปภาพในไดเรกทอรี '{image_dir}' เพื่อสร้างวิดีโอ")
        return

    # อ่านรูปภาพแรกเพื่อกำหนดขนาดของวิดีโอ
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"ไม่สามารถอ่านรูปภาพแรกได้: {image_files[0]}")
        return

    height, width, _ = first_frame.shape
    
    # กำหนด Codec และสร้าง VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' เป็น Codec สำหรับไฟล์ .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"ข้อผิดพลาด: ไม่สามารถเปิด VideoWriter สำหรับ '{output_video_path}' ได้")
        return

    print(f"กำลังสร้างวิดีโอจาก {len(image_files)} รูปภาพ...")
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"คำเตือน: ไม่สามารถอ่านรูปภาพ {img_path} ได้ ข้ามไป")
            continue
        
        # ตรวจสอบและปรับขนาดของเฟรมให้ตรงกับที่ VideoWriter คาดหวัง
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"ประมวลผลไปแล้ว {i + 1}/{len(image_files)} รูปภาพสำหรับวิดีโอ")

    out.release()
    print(f"สร้างวิดีโอเสร็จสิ้นและบันทึกที่ '{output_video_path}'")

def extract_frames_from_video(video_path, output_dir):
    """
    สกัดเฟรมภาพจากวิดีโอและบันทึกเป็นไฟล์ภาพ Sequence (.png) ทีละเฟรม
    """
    cap = load_video(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    print(f"กำลังสกัดเฟรมจากวิดีโอ '{video_path}' ไปยังโฟลเดอร์ '{output_dir}'...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_dir, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, frame)
        count += 1
        
    cap.release()
    print(f"สกัดเฟรมเสร็จสิ้น! ได้รูปภาพทั้งหมด {count} รูป")

if __name__ == "__main__":
    # หาตำแหน่งของ script main.py นี้
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # หาตำแหน่งโฟลเดอร์โปรเจกต์หลัก (ถอยกลับไป 1 ชั้น)
    project_root = os.path.dirname(script_dir)
    # ให้บันทึกในโฟลเดอร์ image_processing/result ตามที่ต้องการ
    RESULT_DIR = os.path.join(script_dir, "result")
    os.makedirs(RESULT_DIR, exist_ok=True)

    KITTI_DIR = os.path.join(project_root, "data", "kitti")
    # path to dataset

    #test_single_image(f"{KITTI_DIR}/training/image_2/um_000002.png")
    #run model 1 : test single image from kitti dataset

    # =========================================================================
    # โหมดที่ 1: รันชุดข้อมูล KITTI Batch และเก็บเข้า Workspace ให้เป็นระเบียบ
    # (ถ้าต้องการใช้งานโหมดนี้ ให้เอาคอมเมนต์ # ออก)
    # print("เริ่มประมวลผลชุดรูปภาพ KITTI...")
    # kitti_workspace = os.path.join(RESULT_DIR, "Workspace_KITTI")
    # kitti_processed_dir = os.path.join(kitti_workspace, "2_processed_frames")
    # os.makedirs(kitti_processed_dir, exist_ok=True)
    # test_kitti_batch(KITTI_DIR, kitti_processed_dir)
    # 
    # kitti_video_path = os.path.join(kitti_workspace, "kitti_batch_result.mp4")
    # print("กำลังรวมภาพผลลัพธ์ KITTI กลับเป็นวิดีโอ...")
    # create_video_from_image_sequence(kitti_processed_dir, kitti_video_path, fps=10)
    # print(f"ประมวลผล KITTI เสร็จสิ้น! เช็คผลลัพธ์ได้ที่: {kitti_workspace}")

    # =========================================================================
    # โหมดประมวลผลวิดีโอและบันทึกเป็นไฟล์วิดีโอโดยตรง (รวดเร็วและไม่เปลืองพื้นที่)
    SAVE_INDIVIDUAL_FRAMES = True  # ตั้งเป็น False ถ้าไม่ต้องการเซฟรูปแต่ละเฟรม (ประหยัดพื้นที่)
    input_video = os.path.join(project_root, "data", "videos", "testcase1.mp4")
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    safe_video_name = video_name.replace(" ", "_")
    
    # 1. จัดเตรียมโฟลเดอร์ Workspace ของวิดีโอนี้
    workspace_dir = os.path.join(RESULT_DIR, f"Workspace_{video_name}") # ชื่อโฟลเดอร์มีเว้นวรรคได้ตามต้องการ
    raw_frames_dir = os.path.join(workspace_dir, "1_raw_frames")
    processed_frames_dir = os.path.join(workspace_dir, "2_processed_frames")
    os.makedirs(raw_frames_dir, exist_ok=True)
    os.makedirs(processed_frames_dir, exist_ok=True)
    
    # กำหนดพาธสำหรับเซฟไฟล์วิดีโอผลลัพธ์ไปที่ Workspace 
    final_video_path = os.path.join(workspace_dir, f"{safe_video_name}_result.mp4")
    roi_video_path = os.path.join(workspace_dir, f"{safe_video_name}_roi.mp4")
    tracker_video_path = os.path.join(workspace_dir, f"{safe_video_name}_tracker.mp4")
    
    print(f"เริ่มประมวลผลวิดีโอ: {input_video}")
    # ส่งพาธของโฟลเดอร์เข้าไปในฟังก์ชันเพื่อให้มันเซฟรูประหว่างที่รันวิดีโอไปด้วย (ถ้าเปิด flag)
    test_video(input_video, final_video_path, 
               raw_dir=raw_frames_dir if SAVE_INDIVIDUAL_FRAMES else None, 
               processed_dir=processed_frames_dir if SAVE_INDIVIDUAL_FRAMES else None, 
               roi_video_path=roi_video_path, tracker_video_path=tracker_video_path)
    print(f"เสร็จสมบูรณ์! เช็คไฟล์ทั้งหมด (วิดีโอและรูปแต่ละเฟรม) ได้ที่โฟลเดอร์:\n{workspace_dir}")