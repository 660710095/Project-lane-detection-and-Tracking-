import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_perspective_matrices(img_shape):
    """
    คำนวณหาเมทริกซ์สำหรับบิดภาพมุมมองปกติ เป็นมุมมองนก (M) 
    และเมทริกซ์สำหรับบิดกลับ (Minv)
    """
    h, w = img_shape[:2]
    
    # 1. พิกัด 4 จุดบนภาพต้นฉบับ (สี่เหลี่ยมคางหมู)
    # เรียงลำดับ: ล่างซ้าย, บนซ้าย, บนขวา, ล่างขวา
    # *คุณสามารถปรับตัวเลข 0.15, 0.45 ฯลฯ ให้เข้ากับมุมกล้องของวิดีโอได้ที่นี่*
    # บีบกรอบให้แคบลง เอาเฉพาะเลนถนนจริงๆ ไม่เอาฟุตบาท
    src = np.float32([
        [w * 0.18, h],           # 1. ล่างซ้าย: ขยับเข้าหาเส้นประด้านซ้าย
        [w * 0.35, h * 0.65],    # 2. บนซ้าย: เล็งให้ตรงปลายเส้นประ
        [w * 0.48, h * 0.65],    # 3. บนขวา: เล็งให้ตรงปลายเส้นทึบสีขาว
        [w * 0.66, h]            # 4. ล่างขวา: หุบเข้ามาเยอะๆ เพื่อเกาะแค่เส้นทึบ ทิ้งฟุตบาทไปเลย!
    ])
    
    # 2. พิกัด 4 จุดปลายทาง (บิดภาพให้กางออกเป็นสี่เหลี่ยมผืนผ้า)
    # เรียงลำดับต้องตรงกับ src: ล่างซ้าย, บนซ้าย, บนขวา, ล่างขวา
    dst = np.float32([
        [w * 0.2, h],            # ล่างซ้าย
        [w * 0.2, 0],            # บนซ้าย
        [w * 0.8, 0],            # บนขวา
        [w * 0.8, h]             # ล่างขวา
    ])
    
    # 3. คำนวณหา Transformation Matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return M, Minv, src  # ส่ง src กลับมาด้วย เผื่อเอาไว้วาดเช็คพิกัด

def warp_image(img, M):
    """ฟังก์ชันสำหรับบิดภาพโดยใช้เมทริกซ์ M"""
    h, w = img.shape[:2]
    # บิดภาพแบบรักษาความละเอียดเท่าเดิม (w, h)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped

# =========================================================================
# โค้ดส่วนล่างนี้คือ "โหมดทดสอบ" จะทำงานก็ต่อเมื่อกดรันไฟล์ perspective.py ตรงๆ
# (จะไม่มีผลตอนที่เรา import ฟังก์ชันไปใช้ใน main.py)
# =========================================================================

if __name__ == '__main__':
    # เพิ่มโฟลเดอร์ปัจจุบันเข้าไปในระบบ เพื่อให้ดึง utils.py มาใช้ได้
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import load_image
    
    # ระบุ Path ไปยังรูปภาพ KITTI ตัวอย่างของคุณ
    # ถ้า Path ไม่ตรง ให้แก้ให้ตรงกับที่มีในเครื่อง


    test_img_path = "data/kitti/training/image_2/um_000000.png"
    
    try:
        img = load_image(test_img_path)
    except FileNotFoundError:
        print(f"หารูปไม่เจอ! ตรวจสอบ Path: {test_img_path}")
        sys.exit()

    # ดึงค่า M และพิกัด 4 จุด
    M, Minv, src_points = get_perspective_matrices(img.shape)
    
    # บิดภาพ
    warped_img = warp_image(img, M)
    
    # วาดกรอบสีแดง (พิกัด 4 จุด) ลงบนภาพต้นฉบับ เพื่อเช็คว่าครอบเลนถนนพอดีไหม
    img_with_pts = img.copy()
    pts = src_points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img_with_pts, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    
    # นำมาพล็อตกราฟโชว์ 2 รูปคู่กัน
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # รูปที่ 1: ภาพต้นฉบับพร้อมกรอบ 4 จุด
    axes[0].imshow(cv2.cvtColor(img_with_pts, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image with Source Points (Red Polygon)')
    
    # รูปที่ 2: ภาพที่ถูกบิดแล้ว (Bird-Eye View)
    axes[1].imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Bird's-Eye View (Warped)")
    
    plt.tight_layout()
    plt.show()