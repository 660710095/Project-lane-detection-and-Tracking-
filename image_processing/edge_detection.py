import cv2
import numpy as np

from perspective import get_perspective_matrices, warp_image
from lane_tracker import sliding_window, search_from_prior, draw_lane_area

def to_grayscale(image):
    # แปลงภาพเป็น จาก  BGR --> grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#===========================================================================

#===========================================================================

def region_of_interest(edges, vertices):
    mask = np.zeros_like(edges)
    # สร้างพื้นที่ mask จากพิกัด vertices ที่ได้รับมา
    cv2.fillPoly(mask, [vertices.astype(np.int32)], 255)
    return cv2.bitwise_and(edges, mask)
#===========================================================================

def detect_brightness(img):
    """คำนวณค่าความสว่างเฉลี่ยของภาพเพื่อแยกแยะกลางวัน/กลางคืน"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_v = np.mean(hsv[:,:,2])
    return avg_v < 85 # ถ้าค่าเฉลี่ยความสว่างต่ำกว่า 85 ถือว่าเป็นกลางคืน

def apply_color_and_gradient_threshold(img, is_night=False, s_thresh=(100, 255), l_thresh=(180, 255), sx_thresh=(15, 255)):
    """ปรับปรุงพารามิเตอร์สำหรับถนนไทย:
    - เพิ่ม Yellow Mask (HSV) แบบยืดหยุ่น
    - ใช้ Adaptive Thresholding ช่วยจัดการกับเงาพาดผ่านถนน
    - ใช้ CLAHE ช่วยดึงขอบภาพในสภาพแสงจ้า
    - รองรับโหมดกลางคืน (Night Mode) โดยปรับค่า Threshold ให้ไวขึ้น
    """
    # 1. เตรียมภาพใน Space ต่างๆ
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    l_channel = hls[:,:,1]

    # ปรับพารามิเตอร์ตามสภาพแสง
    if is_night:
        # โหมดกลางคืน: ลดเกณฑ์ความสว่างและสีเหลืองลงเพื่อให้จับแสงสะท้อนได้ง่ายขึ้น
        lower_yellow = np.array([10, 30, 30])
        l_min = 130
        sx_min = 10
    else:
        # โหมดกลางวัน: ค่าปกติที่จูนมาสำหรับแสงจ้า
        lower_yellow = np.array([15, 50, 50])
        l_min = l_thresh[0]
        sx_min = sx_thresh[0]
    
    # 2. Yellow Mask (HSV) - ถนนไทยเส้นเหลืองสำคัญมาก
    upper_yellow = np.array([45, 255, 255])
    yellow_binary = cv2.inRange(hsv, lower_yellow, upper_yellow) // 255

    # 3. White/Bright Mask ด้วย Adaptive Thresholding
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_channel_eq = clahe.apply(l_channel)
    
    l_adaptive = cv2.adaptiveThreshold(l_channel_eq, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, -10)
    
    l_binary = np.zeros_like(l_channel_eq)
    l_binary[(l_channel_eq >= l_min) | (l_adaptive == 1)] = 1

    # 4. Sobel X (Gradient) - เน้นขอบแนวตั้ง
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_min) & (scaled_sobel <= sx_thresh[1])] = 1

    # 5. รวมผลลัพธ์
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(yellow_binary == 1) | (l_binary == 1) | (sx_binary == 1)] = 1
    
    return combined_binary * 255

#===========================================================================

def apply_morphological_closing(binary_img, kernel_size=5):
    """ใช้เทคนิค Closing เพื่อเชื่อมเส้นประที่ขาดให้เป็นเส้นทึบ และลบจุดบอด"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

#===========================================================================

def full_pipeline(img, lane_processor=None):
    # 0. ตรวจสอบความสว่างของภาพ (กลางวัน/กลางคืน)
    is_night = detect_brightness(img)

    # 1. ใช้ Color & Gradient Thresholding (ส่งค่า is_night เข้าไปด้วย)
    combined_binary = apply_color_and_gradient_threshold(img, is_night=is_night)
    combined_binary = apply_morphological_closing(combined_binary, kernel_size=9)

    # 2. Perspective Transform (บิดภาพ ROI เป็น Bird's-Eye View)
    M, Minv, src_points = get_perspective_matrices(img.shape)
    roi = region_of_interest(combined_binary, src_points)
    binary_warped = warp_image(roi, M) 
    
    # 3. Lane Tracking (สแกนหาเส้นโค้ง)
    if lane_processor and lane_processor.detected and lane_processor.current_left_fit is not None:
        left_fit_raw, right_fit_raw, tracker_img, confidences = search_from_prior(
            binary_warped, 
            lane_processor.current_left_fit, 
            lane_processor.current_right_fit
        )
        # Fallback
        if left_fit_raw is None or right_fit_raw is None:
            left_fit_raw, right_fit_raw, tracker_img, confidences = sliding_window(binary_warped)
    else:
        left_fit_raw, right_fit_raw, tracker_img, confidences = sliding_window(binary_warped)
    
    # 4. ทำให้เส้นนิ่งขึ้น โดยส่งค่า Confidence เข้าไปด้วย
    if lane_processor:
        left_fit, right_fit = lane_processor.process_fits(left_fit_raw, right_fit_raw, binary_warped.shape, confidences)
    else:
        left_fit, right_fit = left_fit_raw, right_fit_raw

    # 5. Draw & Unwarp (ระบายสีเขียวลงบนเลน แล้วบิดกลับไปซ้อนภาพเดิม)
    result = draw_lane_area(img, binary_warped, left_fit, right_fit, Minv)
    
    # เพิ่มข้อความสถานะโหมดแสงบนภาพ
    mode_text = "Night Mode" if is_night else "Day Mode"
    color = (255, 255, 0) if is_night else (0, 255, 255)
    cv2.putText(result, f"Mode: {mode_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return result, combined_binary, roi, binary_warped, tracker_img
         
    