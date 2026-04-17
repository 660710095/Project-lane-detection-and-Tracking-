import cv2
import numpy as np

from perspective import get_perspective_matrices, warp_image
from lane_tracker import sliding_window, search_from_prior, draw_lane_area

def to_grayscale(image):
    # แปลงภาพเป็น จาก  BGR --> grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#===========================================================================

def gaussian_blur(image, kernel_size=5):
    # ใช้ Gaussian blur เพื่อลด noise ในภาพ ก่อนทำ edge detection
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

#===========================================================================

def apply_canny(blurred, T_Low=50, T_High=100):
    # ใช้ Canny edge detection เพื่อดึงเส้นขอบของวัตถุในภาพ
    return cv2.Canny(blurred, T_Low, T_High)

#===========================================================================

def region_of_interest(edges, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros_like(edges)
    # ใช้พิกัดเดียวกับ src ใน perspective.py เพื่อความสอดคล้องกัน
    triangle = np.array([[
        (int(w * 0.20), int(h * 0.95)),  # ล่างซ้าย (ซิงค์กับ perspective.py)
        (int(w * 0.45), int(h * 0.62)),  # บนซ้าย (ซิงค์กับ perspective.py)
        (int(w * 0.55), int(h * 0.62)),  # บนขวา (ซิงค์กับ perspective.py)
        (int(w * 0.85), int(h * 0.95))   # ล่างขวา (ซิงค์กับ perspective.py)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(edges, mask)
#===========================================================================

def apply_color_and_gradient_threshold(img, s_thresh=(100, 255), l_thresh=(220, 255), sx_thresh=(20, 100)):
    """ใช้การกรองสี (HLS) และ Gradient (Sobel) เพื่อแยกเส้นเลนออกจากภาพ
    - S channel (Saturation) เหมาะกับการหาเส้นสีเหลือง/ขาวในสภาพแสงต่างๆ (ปรับให้กว้างขึ้นเล็กน้อย)
    - L channel (Lightness) เหมาะกับการหาเส้นสีขาวสว่างๆ (ปรับให้กว้างขึ้นเล็กน้อย)
    - Sobel X-gradient เหมาะกับการหาเส้นที่ค่อนข้างเป็นแนวตั้ง
    """
    # 1. แปลงเป็น HLS Color Space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]

    # 2. Threshold S-channel (Saturation) เพื่อหาเส้นสี (เหลือง, ขาว)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 100) & (s_channel <= 255)] = 1 # ผ่อนปรนให้จับสีเหลืองจางๆ ของถนนไทยได้

    # 3. Threshold L-channel (Lightness) เพื่อหาเส้นสีขาวสว่าง
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel_eq = clahe.apply(l_channel) 
    
    l_binary = np.zeros_like(l_channel_eq)
    l_binary[(l_channel_eq >= 210) & (l_channel_eq <= 255)] = 1 # ดันค่าเกณฑ์ให้สูงขึ้นได้เพราะภาพชัดขึ้นแล้ว
    # 4. ใช้ Sobel Operator ในแนวแกน X กับภาพ Grayscale
    gray = to_grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 150)] = 1 # กรอง Noise เส้นประที่เบลอ

    # 5. รวมผลลัพธ์ทั้งหมดเข้าด้วยกัน (เส้นสี หรือ เส้นขาวสว่าง หรือ เส้นแนวตั้ง)
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (l_binary == 1) | (sx_binary == 1)] = 1
    return combined_binary * 255

#===========================================================================

def full_pipeline(img, lane_processor=None):
    # 1. ใช้ Color & Gradient Thresholding เพื่อให้ได้ภาพ Binary ที่มีคุณภาพ
    combined_binary = apply_color_and_gradient_threshold(img)
    roi = region_of_interest(combined_binary, img.shape)

    # 2. Perspective Transform (บิดภาพ ROI เป็น Bird's-Eye View)
    M, Minv, src_points = get_perspective_matrices(img.shape)
    binary_warped = warp_image(roi, M) 
    
    # 3. Lane Tracking (สแกนหาเส้นโค้ง)
    # ใช้เทคนิค Search from Prior ถ้าเฟรมก่อนหน้าหาเส้นเจอและผ่าน Sanity Check
    if lane_processor and lane_processor.detected and lane_processor.current_left_fit is not None:
        left_fit_raw, right_fit_raw, tracker_img = search_from_prior(
            binary_warped, 
            lane_processor.current_left_fit, 
            lane_processor.current_right_fit
        )
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
         
    