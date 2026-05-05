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

def apply_color_and_gradient_threshold(img, s_thresh=(100, 255), l_thresh=(180, 255), sx_thresh=(15, 255)):
    """ปรับปรุงพารามิเตอร์สำหรับถนนไทย:
    - เพิ่ม Yellow Mask (HSV) เพื่อจับเส้นสีเหลืองโดยเฉพาะ
    - ปรับ L-channel ให้ไวขึ้นต่อเส้นที่จาง
    - ใช้ CLAHE ช่วยดึงขอบภาพในสภาพแสงจ้า
    """
    # 1. แปลงเป็น HLS และ HSV
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # 2. Yellow Mask (HSV) - ถนนไทยเส้นเหลืองสำคัญมาก
    lower_yellow = np.array([15, 60, 80]) 
    upper_yellow = np.array([45, 255, 255])
    yellow_binary = cv2.inRange(hsv, lower_yellow, upper_yellow) // 255

    # 3. White/Bright Mask (L-channel) + CLAHE
    # ใช้ CLAHE เพื่อปรับ Contrast เฉพาะจุด ช่วยให้เห็นเส้นขาวกลางแดดจ้า
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_channel_eq = clahe.apply(l_channel)
    l_binary = np.zeros_like(l_channel_eq)
    l_binary[(l_channel_eq >= l_thresh[0]) & (l_channel_eq <= l_thresh[1])] = 1

    # 4. Sobel X (Gradient) - เน้นขอบแนวตั้ง
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # 5. รวมผลลัพธ์ (เหลือง OR ขาว OR ขอบแนวตั้ง)
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
    # 1. ใช้ Color & Gradient Thresholding เพื่อให้ได้ภาพ Binary ที่มีคุณภาพ
    combined_binary = apply_color_and_gradient_threshold(img)
    combined_binary = apply_morphological_closing(combined_binary, kernel_size=9)

    # 2. Perspective Transform (บิดภาพ ROI เป็น Bird's-Eye View)
    M, Minv, src_points = get_perspective_matrices(img.shape)
    roi = region_of_interest(combined_binary, src_points)
    binary_warped = warp_image(roi, M) 
    
    # 3. Lane Tracking (สแกนหาเส้นโค้ง)
    # ใช้เทคนิค Search from Prior ถ้าเฟรมก่อนหน้าหาเส้นเจอและผ่าน Sanity Check
    if lane_processor and lane_processor.detected and lane_processor.current_left_fit is not None:
        left_fit_raw, right_fit_raw, tracker_img = search_from_prior(
            binary_warped, 
            lane_processor.current_left_fit, 
            lane_processor.current_right_fit
        )
        # Fallback: ถ้า search_from_prior ล้มเหลว ให้กลับไปทำ sliding_window ใหม่
        if left_fit_raw is None or right_fit_raw is None:
            left_fit_raw, right_fit_raw, tracker_img = sliding_window(binary_warped)
    else:
        # ถ้าหาไม่เจอ หรือเป็นเฟรมแรก ให้ทำ Sliding Window ใหม่ทั้งหมด
        left_fit_raw, right_fit_raw, tracker_img = sliding_window(binary_warped)
    
    # 4. ทำให้เส้นนิ่งขึ้น (Smoothing) และจัดการกับเฟรมที่หาเส้นไม่เจอ
    if lane_processor:
        left_fit, right_fit = lane_processor.process_fits(left_fit_raw, right_fit_raw, binary_warped.shape)
    else:
        left_fit, right_fit = left_fit_raw, right_fit_raw

    # 5. Draw & Unwarp (ระบายสีเขียวลงบนเลน แล้วบิดกลับไปซ้อนภาพเดิม)
    result = draw_lane_area(img, binary_warped, left_fit, right_fit, Minv)
    return result, combined_binary, roi, binary_warped, tracker_img
         
    