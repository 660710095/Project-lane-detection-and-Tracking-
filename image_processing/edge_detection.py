import cv2
import numpy as np
def to_grayscale(image):
    # แปลงภาพเป็น จาก  BGR --> grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#===========================================================================

def gaussian_blur(image, kernel_size=5):
    # ใช้ Gaussian blur เพื่อลด noise ในภาพ ก่อนทำ edge detection
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

#===========================================================================

def apply_canny(blurred, T_Low=30, T_High=130):
    # ใช้ Canny edge detection เพื่อดึงเส้นขอบของวัตถุในภาพ
    return cv2.Canny(blurred, T_Low, T_High)

#===========================================================================

def region_of_interest(edged, img_shape):
    
    h, w = img_shape[:2]
    mask = np.zeros_like(edged)
    triangle = np.array([[
                        (0, h),
                        (w, h), 
                        (w//2, int(h*0.55)),
                        (w, h)
    ]],dtype=np.int32)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(edged, mask)

#===========================================================================

def detect_lines(roi):
    # ใช้ Hough Transform เพื่อดึงเส้นตรงจากภาพที่ผ่านการทำ edge detection 
    lines = cv2.HoughLinesP(roi,
                            rho=1,
                            theta=np.pi/180,
                            threshold=40,
                            minLineLength=20,
                            maxLineGap=170)
    return lines

#===========================================================================

def draw_lines(img, lines):
    # วาดเส้นที่ตรวจจับได้ลงบนภาพต้นฉบับ
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return cv2.addWeighted(img, 0.8, img, 1.0, 0.0)

#===========================================================================

def full_pipeline(img):
    gray    = to_grayscale(img)
    blurred = gaussian_blur(gray)
    edges   = apply_canny(blurred)
    roi     = region_of_interest(edges, img.shape)
    lines   = detect_lines(roi)
    result  = draw_lines(img, lines)
    return result, edges, roi  
         
    