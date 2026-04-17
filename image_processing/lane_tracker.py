import cv2
import numpy as np

def sliding_window(binary_warped):
    """
    ใช้เทคนิค Sliding Window หาพิกัดของพิกเซลที่เป็นเส้นถนน
    """
    # 1. หาจุดเริ่มต้นของเส้นซ้ายและขวาจาก 1/3 ด้านล่างของภาพด้วย Histogram
    # (ปรับจากครึ่งล่างเป็น 1/3 ด้านล่าง เพื่อไม่ให้ทางโค้งช่วงกลางภาพมาดึงค่าจุดเริ่มต้นผิดไป)
    bottom_third = binary_warped.shape[0] * 2 // 3
    histogram = np.sum(binary_warped[bottom_third:, :], axis=0)
    
    # สร้างภาพเปล่าไว้สำหรับวาดกรอบสีๆ เพื่อดูการทำงาน (Debug)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 2. ตั้งค่า Sliding Window
    nwindows = 25 # เพิ่มจำนวนหน้าต่าง (จาก 15 -> 25) ทำให้หน้าต่างเตี้ยลงและปรับตัวตามขอบโค้งได้ถี่และละเอียดขึ้น
    margin = 100  # ขยายขอบเขตความกว้างหน้าต่าง (จาก 80 -> 100) ป้องกันเส้นหลุดกรอบเวลาเจอโค้งหักศอก
    minpix = 40   # ลดจำนวนพิกเซลขั้นต่ำลงเล็กน้อยให้สัมพันธ์กับหน้าต่างที่เล็กลง

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

def search_from_prior(binary_warped, left_fit, right_fit):
    """
    ค้นหาเส้นเลนโดยอ้างอิงจากสมการเส้นโค้งของเฟรมก่อนหน้า (Search from Prior)
    วิธีนี้จะเร็วกว่าและเสถียรกว่าการทำ Sliding Window ใหม่ทั้งหมด
    """
    # 1. ตั้งค่าขอบเขตการค้นหา (Margin) จากเส้นเดิม
    margin = 100  # ปรับให้สอดคล้องกับ Sliding window ด้านบนเพื่อให้จับช่วงโค้งได้กว้างขึ้น

    # 2. หาตำแหน่งของพิกเซลที่ไม่ใช่สีดำทั้งหมดในภาพ
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 3. สร้างขอบเขตการค้นหาจากสมการเส้นโค้งของเฟรมที่แล้ว
    # พิกเซลที่จะถูกพิจารณาคือพิกเซลที่อยู่ในระยะ margin จากเส้นเดิมเท่านั้น
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # 4. ดึงพิกัด (x, y) ของพิกเซลที่เป็นเส้น
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 5. คำนวณสมการพาราโบลาใหม่
    new_left_fit, new_right_fit = None, None
    if len(leftx) > 0 and len(lefty) > 0:
        new_left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0 and len(righty) > 0:
        new_right_fit = np.polyfit(righty, rightx, 2)

    # 6. สร้างภาพสำหรับแสดงผล (Visualization)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    
    # ระบายสีพิกเซลที่ถูกตรวจจับ (ซ้าย=แดง, ขวา=น้ำเงิน)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # สร้างพื้นที่ค้นหาสีเขียวโปร่งแสง
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return new_left_fit, new_right_fit, result

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
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # วาดเส้นทึบที่ขอบถนนทั้งสองข้าง เพื่อเน้นขอบโค้งให้คมชัดและดูเกาะถนนมากขึ้น
    # ความหนา (thickness=25) เมื่อถูกบิดมุมมองกลับจะทำให้ดูมีมิติ (เส้นเล็กลงเมื่ออยู่ไกล)
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=25)  # ขอบซ้าย (สีน้ำเงิน)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=25) # ขอบขวา (สีแดง)

    # บิดภาพกลับไปยังมุมมองปกติ (ใช้ Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0])) 
    
    # ซ้อนภาพสีเขียวโปร่งแสง ลงบนภาพต้นฉบับ
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result