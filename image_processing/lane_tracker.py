import cv2
import numpy as np

# --- Lane Keep Assist (LKA) Configuration ---
DRIFT_THRESHOLD = 50  # ระยะพิกเซลที่ยอมรับได้ก่อนจะแจ้งเตือน (ปรับค่าความไวได้ที่นี่)

def get_hough_guidance_mask(binary_warped):
    """
    ใช้ Hough Transform ค้นหาแนวเส้นตรงเพื่อสร้าง Mask "ชี้เป้า" 
    ช่วยกรองพิกเซลขยะออกก่อนจะทำ Sliding Window
    """
    mask = np.zeros_like(binary_warped)
    
    # 1. ค้นหาเส้นด้วย Probabilistic Hough Transform
    # rho=1, theta=pi/180, threshold=20, minLineLength=50, maxLineGap=100
    lines = cv2.HoughLinesP(binary_warped, 1, np.pi/180, 20, minLineLength=50, maxLineGap=100)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # กรองเอาเฉพาะเส้นที่มีแนวโน้มเป็นเส้นเลน (ความชันค่อนไปทางแนวตั้ง)
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if abs(slope) > 0.5: # กรองเส้นแนวนอนทิ้ง
                    # วาดเส้นทึบขนาดใหญ่ (thickness=80) เพื่อสร้าง "พื้นที่ปลอดภัย"
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 80)
    
    # ถ้าหาเส้นด้วย Hough ไม่เจอเลย ให้คืนค่าภาพสีขาวทั้งหมด (เพื่อให้ Sliding Window ทำงานปกติ)
    if np.sum(mask) == 0:
        return np.ones_like(binary_warped) * 255
        
    return mask

def sliding_window(binary_warped):
    """
    ใช้เทคนิค Sliding Window หาพิกัดของพิกเซลที่เป็นเส้นถนน
    (ปรับปรุง: วาดเส้น Hough สีเหลืองทับตอนสุดท้ายเพื่อให้เห็นชัดเจน)
    """
    # --- ขั้นตอนที่ 0: ใช้ Hough Transform ชี้เป้า ---
    hough_mask = get_hough_guidance_mask(binary_warped)
    # กรองพิกเซล
    guided_binary = cv2.bitwise_and(binary_warped, hough_mask)

    # --- นำเทคนิค Closing แนวตั้งมาช่วยเชื่อมเส้นประก่อนสแกน ---
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 35))
    guided_binary = cv2.morphologyEx(guided_binary, cv2.MORPH_CLOSE, vertical_kernel)

    # 1. หาจุดเริ่มต้น
    bottom_quarter = guided_binary.shape[0] * 3 // 4
    histogram = np.sum(guided_binary[bottom_quarter:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 2. ตั้งค่า Sliding Window
    nwindows = 10 
    margin = 110 
    minpix = 50   
    window_height = int(guided_binary.shape[0] // nwindows)
    
    nonzero = guided_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    # เตรียมภาพแสดงผล (พิกเซลสีขาว)
    out_img = np.dstack((guided_binary, guided_binary, guided_binary)) * 255

    # 3. เริ่มสแกน
    for window in range(nwindows):
        win_y_low = guided_binary.shape[0] - (window + 1) * window_height
        win_y_high = guided_binary.shape[0] - window * window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

        # ตีกรอบสีเขียว (Sliding Window)
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # --- วาดเส้น Hough สีเหลืองทับตอนท้าย (เพื่อให้เห็นความต่าง) ---
    lines = cv2.HoughLinesP(binary_warped, 1, np.pi/180, 20, minLineLength=50, maxLineGap=100)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if abs(slope) > 0.5:
                    cv2.line(out_img, (x1, y1), (x2, y2), (0, 255, 255), 4) # วาดทับเป็นสีเหลือง

    # รวมพิกัดและคำนวณ Polyfit
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    left_fit, right_fit = None, None
    left_conf, right_conf = 0.0, 0.0
    if len(leftx) > 300:
        left_fit = np.polyfit(lefty, leftx, 3)
        left_conf = min(1.0, len(leftx) / 1500)
    if len(rightx) > 300:
        right_fit = np.polyfit(righty, rightx, 3)
        right_conf = min(1.0, len(rightx) / 1500)

    return left_fit, right_fit, out_img, (left_conf, right_conf)

def search_from_prior(binary_warped, left_fit, right_fit):
    """
    ค้นหาเส้นเลนโดยอ้างอิงจากสมการเส้นโค้งของเฟรมก่อนหน้า (Search from Prior)
    (ปรับปรุง: เพิ่มการวาดเส้น Hough สีเหลืองเปรียบเทียบ)
    """
    # --- นำเทคนิค Closing แนวตั้งมาช่วยเชื่อมเส้นประก่อนสแกน ---
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 35))
    binary_warped = cv2.morphologyEx(binary_warped, cv2.MORPH_CLOSE, vertical_kernel)

    # 1. ตั้งค่าขอบเขตการค้นหา (Margin) จากเส้นเดิม
    margin = 120  

    # 2. หาตำแหน่งพิกเซล
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 3. สร้างขอบเขตการค้นหาจากสมการเดิม
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**3) + left_fit[1]*(nonzeroy**2) + left_fit[2]*nonzeroy + left_fit[3] - margin)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**3) + left_fit[1]*(nonzeroy**2) + left_fit[2]*nonzeroy + left_fit[3] + margin)))
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**3) + right_fit[1]*(nonzeroy**2) + right_fit[2]*nonzeroy + right_fit[3] - margin)) & 
                       (nonzerox < (right_fit[0]*(nonzeroy**3) + right_fit[1]*(nonzeroy**2) + right_fit[2]*nonzeroy + right_fit[3] + margin)))

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds] 
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # 4. คำนวณสมการใหม่
    new_left_fit, new_right_fit = None, None
    left_conf, right_conf = 0.0, 0.0
    if len(leftx) > 300:
        new_left_fit = np.polyfit(lefty, leftx, 3)
        left_conf = min(1.0, len(leftx) / 1500)
    if len(rightx) > 300:
        new_right_fit = np.polyfit(righty, rightx, 3)
        right_conf = min(1.0, len(rightx) / 1500)

    # 5. สร้างภาพ Visualization
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # --- วาดเส้น Hough สีเหลืองเปรียบเทียบ ---
    lines = cv2.HoughLinesP(binary_warped, 1, np.pi/180, 20, minLineLength=50, maxLineGap=100)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if abs(slope) > 0.5:
                    cv2.line(out_img, (x1, y1), (x2, y2), (0, 255, 255), 4)

    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty + left_fit[3]
    right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty + right_fit[3]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return new_left_fit, new_right_fit, result, (left_conf, right_conf)

# ===========================================================================

def draw_lane_area(original_img, binary_warped, left_fit, right_fit, Minv):
    """
    วาดพื้นที่สีเขียวทับเลนถนน แล้วบิดมุมมองกลับไปทาบภาพเดิม
    หากหาเส้นไม่เจอ จะพิมพ์ข้อความเตือนลงบนภาพ
    """
    result_img = np.copy(original_img)
    
    if left_fit is None or right_fit is None:
        # พิมพ์ข้อความเตือนสีแดงกลางจอ
        cv2.putText(result_img, "WARNING: Lane Not Detected", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        return result_img

    # สร้างแกน Y ล่วงหน้าเพื่อใช้วาดเส้น
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    # คำนวณค่า X จากสมการ x = Ay^3 + By^2 + Cy + D
    left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty + left_fit[3]
    right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty + right_fit[3]

    # สร้างภาพเปล่าสีดำ
    color_warp = np.zeros_like(original_img).astype(np.uint8)

    # จัดรูปแบบพิกัดเพื่อวาด Polygon
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # ระบายสีเขียวลงในพื้นที่ระหว่างเส้นซ้าย-ขวา
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # วาดเส้นทึบที่ขอบถนนทั้งสองข้าง เพื่อเน้นขอบโค้งให้คมชัดและดูเกาะถนนมากขึ้น
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=25)  # ขอบซ้าย (สีน้ำเงิน)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25) # ขอบขวา (สีแดง)

    # บิดภาพกลับไปยังมุมมองปกติ (ใช้ Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0])) 
    
    # ซ้อนภาพสีเขียวโปร่งแสง ลงบนภาพต้นฉบับ
    result = cv2.addWeighted(result_img, 1, newwarp, 0.3, 0)

    # --- ระบบแจ้งเตือน Lane Keep Assist (LKA) ---
    # 1. คำนวณหาจุดกึ่งกลางของเลน (ที่ตำแหน่งล่างสุดของภาพ)
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    # 2. หาจุดกึ่งกลางของกล้อง (กึ่งกลางเฟรม)
    camera_center = original_img.shape[1] / 2
    # 3. คำนวณระยะห่าง (Offset)
    offset = lane_center - camera_center

    # 4. กำหนดข้อความและสีตามเงื่อนไข
    if offset > DRIFT_THRESHOLD:
        status_text = "Warning: Drifting LEFT!"
        color = (0, 0, 255) # สีแดง
    elif offset < -DRIFT_THRESHOLD:
        status_text = "Warning: Drifting RIGHT!"
        color = (0, 0, 255) # สีแดง
    else:
        status_text = "Status: Safe"
        color = (0, 255, 0) # สีเขียว

    # 5. วาดข้อความลงบนภาพผลลัพธ์
    cv2.putText(result, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    # แสดงค่า offset เล็กๆ เพื่อการ Debug (ทางเลือก)
    cv2.putText(result, f"Offset: {int(offset)}px", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return result