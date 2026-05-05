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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. เปลี่ยนตำแหน่งไฟล์วิดีโอตรงนี้ให้ตรงกับคลิปที่คุณต้องการสกัด
    VIDEO_PATH = os.path.join(script_dir, "data", "videos", "Video Project 2.mp4") 
    # 2. โฟลเดอร์ปลายทางที่จะเก็บรูปภาพดิบ
    OUTPUT_DIR = os.path.join(script_dir, "data", "custom_dataset", "raw_images")
    
    extract_frames(VIDEO_PATH, OUTPUT_DIR, frame_interval=15)