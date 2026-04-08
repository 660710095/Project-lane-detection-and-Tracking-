import cv2
import numpy as np

def sliding_window(binary_warped):
    """
    ใช้เทคนิค Sliding Window หาพิกัดของพิกเซลที่เป็นเส้นถนน
    """
    # 1. หาจุดเริ่มต้นของเส้นซ้ายและขวาจากครึ่งล่างของภาพด้วย Histogram
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    
    # สร้างภาพเปล่าไว้สำหรับวาดกรอบสีๆ เพื่อดูการทำงาน (Debug)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 2. ตั้งค่า Sliding Window
    nwindows = 9 # จำนวนหน้าต่างที่จะใช้สแกนจากล่างขึ้นบน
    margin = 100 # ความกว้างของหน้าต่าง (ซ้าย-ขวา จากจุดศูนย์กลาง)
    minpix = 50  # จำนวนพิกเซลขั้นต่ำที่จะให้เลื่อนจุดศูนย์กลางหน้าต่าง

    window_height = int(binary_warped.shape[0] // nwindows)
    
    # หาตำแหน่งของพิกเซลที่ไม่ใช่สีดำทั้งหมดในภาพ
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # อัปเดตตำแหน่ง x ปัจจุบัน
    leftx_current = leftx_base
    rightx_current = rightx_base

    # เก็บ index ของพิกเซลที่อยู่ในกรอบ
    left_lane_inds = []
    right_lane_inds = []

    # 3. เริ่มสแกนทีละหน้าต่าง
    for window in range(nwindows):
        # ขอบเขตแกน Y ของหน้าต่าง
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        # ขอบเขตแกน X ของหน้าต่าง (ซ้ายและขวา)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # ตีกรอบสี่เหลี่ยมเพื่อดูการทำงาน (สีเขียว)
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # หาพิกเซลที่อยู่ในกรอบ
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # ถ้าเจอพิกเซลเยอะกว่า minpix ให้เลื่อนแกน X ของกรอบถัดไปตามค่าเฉลี่ย
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # รวม array ของ index 
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # ดึงพิกัด (x, y) ของพิกเซลที่เป็นเส้น
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 4. คำนวณสมการพาราโบลา (Polynomial Degree 2)
    left_fit, right_fit = None, None
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, out_img

# ===========================================================================

def draw_lane_area(original_img, binary_warped, left_fit, right_fit, Minv):
    """
    วาดพื้นที่สีเขียวทับเลนถนน แล้วบิดมุมมองกลับไปทาบภาพเดิม
    """
    if left_fit is None or right_fit is None:
        return original_img # ถ้าหาเส้นไม่เจอ ให้คืนค่าภาพเดิมกลับไป

    # สร้างแกน Y ล่วงหน้าเพื่อใช้วาดเส้น
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # คำนวณค่า X จากสมการ x = Ay^2 + By + C
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # สร้างภาพเปล่าสีดำ
    color_warp = np.zeros_like(original_img).astype(np.uint8)

    # จัดรูปแบบพิกัดเพื่อวาด Polygon
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # ระบายสีเขียวลงในพื้นที่ระหว่างเส้นซ้าย-ขวา
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # บิดภาพกลับไปยังมุมมองปกติ (ใช้ Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0])) 
    
    # ซ้อนภาพสีเขียวโปร่งแสง ลงบนภาพต้นฉบับ
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result