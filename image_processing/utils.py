import cv2
import os
def load_image(path):
    # โหลดภาพมาจาก path 
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return img
def load_video(path):
    # โหลดวิดีโอมาจาก path 
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found at path: {path}")
    return cap

def save_video(output_path, frames, fps):
    # บันทึก frames เป็นไฟล์วิดีโอ
    if not frames:
        return
    h,w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def get_kitti_image(kitti_dir):
    # ดึง Path ของภาพจาก KITTI dataset มาทั้งหมด
    img_dir = os.path.join(kitti_dir, f)
    paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])
    return paths