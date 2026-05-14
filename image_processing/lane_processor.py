import numpy as np

# ค่าคงที่สำหรับ Sanity Check (ปรับจูนสำหรับถนนไทย)
SANITY_MIN_WIDTH_RATIO = 0.20   # ลดค่าขั้นต่ำเพื่อให้รองรับเลนที่ดูแคบลงเวลาขึ้นเนิน/ลงเนิน
SANITY_MAX_WIDTH_RATIO = 0.80   # เพิ่มค่าสูงสุดเพื่อรองรับเลนกว้างพิเศษ (เช่น ทางด่วน) หรือกล้องเลนส์กว้าง (Wide Angle)
SANITY_MAX_CURVATURE_DIFF = 0.025 # เพิ่มความยืดหยุ่นให้สมการเส้นซ้าย-ขวาต่างกันได้มากขึ้นในทางโค้งหักศอก
SANITY_MAX_SLOPE_DIFF = 1.0       # ยอมให้ทิศทางเส้นต่างกันได้มากขึ้นเล็กน้อย

# ค่าคงที่สำหรับเช็คความต่อเนื่องกับเฟรมก่อนหน้า
HISTORY_MAX_CURVATURE_DIFF = 0.01   # ลดลงเพื่อไม่ให้รูปร่างเส้นเปลี่ยนเร็วเกินไป
HISTORY_MAX_SLOPE_DIFF = 0.5        # ลดลงเพื่อไม่ให้เส้นสวิงเปลี่ยนทิศทางเร็วเกินไป
# HISTORY_MAX_POS_DIFF_RATIO ถูกกำหนดแบบ dynamic ในฟังก์ชัน

class LaneProcessor:
    def __init__(self, history_length=15, max_lost_frames=10): # เพิ่มประวัติให้ยาวขึ้นเพื่อให้เส้นนิ่งขึ้น (จาก 6-8 -> 15)
        """
        คลาสสำหรับจัดการสถานะของเส้นเลน (Lane State) เพื่อให้ผลลัพธ์ในวิดีโอนิ่งขึ้น
        """
        self.history_length = history_length
        self.left_fit_history = []
        self.right_fit_history = []
        self.current_left_fit = None
        self.current_right_fit = None
        self.detected = False
        self.lost_frames = 0        # นับจำนวนเฟรมที่หาเส้นไม่เจอต่อเนื่อง
        self.max_lost_frames = max_lost_frames # จำนวนเฟรมสูงสุดที่จะยอมใช้ค่าเก่า ก่อนจะยอมแพ้และหยุดวาด

    def process_fits(self, left_fit, right_fit, img_shape, confidences=(0.0, 0.0)):
        """
        รับค่า fit จากเฟรมปัจจุบัน และค่า Confidence (0.0 - 1.0)
        """
        left_conf, right_conf = confidences
        avg_conf = (left_conf + right_conf) / 2

        # กรณีหาเส้นไม่เจอเลย หรือความมั่นใจต่ำมาก
        if (left_fit is None and right_fit is None) or avg_conf < 0.4:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.detected = False
                self.current_left_fit = None
                self.current_right_fit = None
                self.left_fit_history = []
                self.right_fit_history = []
            return self.current_left_fit, self.current_right_fit

        # เติมเส้นที่ขาดหายไปหนึ่งข้าง (ใช้ความกว้างเลนเฉลี่ยมาประมาณค่า)
        expected_width = int(img_shape[1] * 0.35)
        new_left_fit = np.copy(left_fit) if left_fit is not None else None
        new_right_fit = np.copy(right_fit) if right_fit is not None else None

        if new_left_fit is not None and new_right_fit is None:
            new_right_fit = np.copy(new_left_fit)
            new_right_fit[3] += expected_width
        elif new_right_fit is not None and new_left_fit is None:
            new_left_fit = np.copy(new_right_fit)
            new_left_fit[3] -= expected_width

        # Sanity Check
        if self._sanity_check_ok(new_left_fit, new_right_fit, img_shape):
            self.lost_frames = 0 # เจอเส้นที่ดีแล้ว รีเซ็ตตัวนับ
            self.detected = True
            self.left_fit_history.append(new_left_fit)
            self.right_fit_history.append(new_right_fit)

            if len(self.left_fit_history) > self.history_length:
                self.left_fit_history.pop(0)
                self.right_fit_history.pop(0)
            
            # ปรับน้ำหนักให้เกลี่ยเฉลี่ยมากขึ้น
            n = len(self.left_fit_history)
            weights = np.arange(1, n + 1).astype(np.float64)
            weights /= weights.sum()
            
            avg_left = np.average(self.left_fit_history, axis=0, weights=weights)
            avg_right = np.average(self.right_fit_history, axis=0, weights=weights)

            # Coefficient Clamping (บีบให้การเปลี่ยนแปลงในแต่ละเฟรม "น้อยลง")
            if self.current_left_fit is not None:
                max_changes = [0.00005, 0.0005, 0.05, 30.0] 
                new_left = np.zeros_like(avg_left)
                new_right = np.zeros_like(avg_right)
                for i in range(4):
                    new_left[i] = np.clip(avg_left[i], self.current_left_fit[i] - max_changes[i], self.current_left_fit[i] + max_changes[i])
                    new_right[i] = np.clip(avg_right[i], self.current_right_fit[i] - max_changes[i], self.current_right_fit[i] + max_changes[i])
                self.current_left_fit = new_left
                self.current_right_fit = new_right
            else:
                self.current_left_fit = avg_left
                self.current_right_fit = avg_right
        else:
            # ถ้าไม่ผ่าน Sanity Check ให้นับว่าเป็น Lost Frame
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.detected = False
                self.current_left_fit = None
                self.current_right_fit = None
                self.left_fit_history = []
                self.right_fit_history = []

        return self.current_left_fit, self.current_right_fit

    def _sanity_check_ok(self, left_fit, right_fit, img_shape):
        """
        ตรวจสอบความสมเหตุสมผลของเส้นที่หาได้ (รองรับ Degree 3)
        """
        if left_fit is None or right_fit is None:
            return False

        h = img_shape[0]
        ploty = np.linspace(0, h - 1, h)
        # คำนวณ x ด้วยสมการ Degree 3: Ay^3 + By^2 + Cy + D
        left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty + left_fit[3]
        right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty + right_fit[3]

        # 2. ตรวจสอบความกว้างของเลน (Lane Width) ที่ส่วนล่างของภาพ
        lane_width_bottom = right_fitx[-1] - left_fitx[-1]
        min_width = img_shape[1] * SANITY_MIN_WIDTH_RATIO
        max_width = img_shape[1] * SANITY_MAX_WIDTH_RATIO
        if not (min_width < lane_width_bottom < max_width):
            return False

        # 3. ตรวจสอบความแตกต่างของค่าสัมประสิทธิ์ (เน้นที่ความโค้ง B และความชัน C)
        if abs(left_fit[1] - right_fit[1]) > SANITY_MAX_CURVATURE_DIFF:
            return False
        if abs(left_fit[2] - right_fit[2]) > SANITY_MAX_SLOPE_DIFF:
            return False

        return True