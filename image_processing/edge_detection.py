import cv2
import numpy as np

from perspective import get_perspective_matrices, warp_image
from lane_tracker import sliding_window, draw_lane_area

def to_grayscale(image):
    # แปลงภาพเป็น จาก  BGR --> grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#===========================================================================

def gaussian_blur(image, kernel_size=5):
    # ใช้ Gaussian blur เพื่อลด noise ในภาพ ก่อนทำ edge detection
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

#===========================================================================

def apply_canny(blurred, T_Low=30, T_High=100):
    # ใช้ Canny edge detection เพื่อดึงเส้นขอบของวัตถุในภาพ
    return cv2.Canny(blurred, T_Low, T_High)

#===========================================================================

def region_of_interest(edges, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros_like(edges)
    triangle = np.array([[
        (int(w * 0.05), h),             # ล่างซ้าย
        (int(w * 0.30), int(h * 0.65)), # บนซ้าย
        (int(w * 0.55), int(h * 0.65)), # บนขวา
        (int(w * 0.82), h)              # ล่างขวา
    ]], dtype=np.int32)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(edges, mask)
#===========================================================================

def detect_lines(roi):
    # ใช้ Hough Transform เพื่อดึงเส้นตรงจากภาพที่ผ่านการทำ edge detection 
    lines = cv2.HoughLinesP(roi,
                            rho=1,
                            theta=np.pi/180,
                            threshold=50,
                            minLineLength=40,
                            maxLineGap=50)
    return lines

#===========================================================================

def draw_lines(img, lines):
    # วาดเส้นที่ตรวจจับได้ลงบนภาพต้นฉบับ
    line_img = np.zeros_like(img)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            
    # ซ้อนภาพเส้นสีเขียว ลงบนภาพต้นฉบับ
    return cv2.addWeighted(img, 1, line_img, 1.0, 0.0)

#===========================================================================

def full_pipeline(img):
    gray    = to_grayscale(img)
    blurred = gaussian_blur(gray)
    edges   = apply_canny(blurred)
    roi     = region_of_interest(edges, img.shape)
    lines   = detect_lines(roi)
    result  = draw_lines(img, lines)

    # 2. Perspective Transform (บิดภาพ ROI เป็น Bird's-Eye View)
    M, Minv, src_points = get_perspective_matrices(img.shape)
    binary_warped = warp_image(roi, M) 
    
    # 3. Lane Tracking (สแกนหาเส้นโค้งด้วยกรอบสี่เหลี่ยม)
    left_fit, right_fit, sliding_window_img = sliding_window(binary_warped)
    
    # 4. Draw & Unwarp (ระบายสีเขียวลงบนเลน แล้วบิดกลับไปซ้อนภาพเดิม)
    result = draw_lane_area(img, binary_warped, left_fit, right_fit, Minv)
    return result, edges, roi, binary_warped, sliding_window_img
         
    