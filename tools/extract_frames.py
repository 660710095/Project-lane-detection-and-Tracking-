import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=15):
    """
    สกัดภาพจากวิดีโอเพื่อนำไปใช้ทำ Dataset (LabelMe)
    video_path: ตำแหน่งไฟล์วิดีโอ
    output_dir: โฟลเดอร์สำหรับเก็บรูปภาพ
    frame_interval: ดึงภาพทุกๆ X เฟรม (เช่น 15 เฟรม หมายถึงถ้าวิดีโอ 30fps จะได้ 2 ภาพต่อวินาที)
    """
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ไม่สามารถเปิดวิดีโอได้: {video_path}")
        print("โปรดเช็คว่าใส่ชื่อไฟล์/ตำแหน่งไฟล์ถูกต้องหรือไม่")
        return

    frame_count = 0
    saved_count = 0

    print(f"กำลังเริ่มสกัดภาพจากวิดีโอ: {os.path.basename(video_path)}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break # จบวิดีโอ

        # บันทึกภาพเฉพาะเฟรมที่ตกครบรอบที่กำหนดไว้ (เพื่อไม่ให้ภาพซ้ำกันเกินไป)
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"thai_road_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"สกัดภาพเสร็จสมบูรณ์! ได้รูปภาพมาทั้งหมด {saved_count} ภาพ")
    print(f"เข้าไปดูรูปภาพเพื่อเตรียมทำ LabelMe ได้ที่: {output_dir}")

if __name__ == "__main__":
    # หาตำแหน่งของโฟลเดอร์โปรเจกต์หลัก (ถอยกลับไป 1 ชั้นจากโฟลเดอร์ tools)
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(tools_dir)
    
    # 1. ระบุชื่อไฟล์วิดีโอไทยที่ต้องการสกัดเฟรม (อยู่ใน data/videos/)
    VIDEO_PATH = os.path.join(project_root, "data", "videos", "testcase1.mp4") 
    
    # 2. โฟลเดอร์ที่จะเก็บรูปภาพเพื่อนำไปทำ Dataset ต่อ
    OUTPUT_DIR = os.path.join(project_root, "dataset_for_ai", "images")
    
    extract_frames(VIDEO_PATH, OUTPUT_DIR, frame_interval=30)