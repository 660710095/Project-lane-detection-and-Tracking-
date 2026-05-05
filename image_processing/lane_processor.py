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
    def __init__(self, history_length=6):
        """
        คลาสสำหรับจัดการสถานะของเส้นเลน (Lane State) เพื่อให้ผลลัพธ์ในวิดีโอนิ่งขึ้น
        - เก็บประวัติของสมการเส้นเลน (polynomial fits)
        - ทำค่าเฉลี่ยเคลื่อนที่ (Moving Average) เพื่อลดการสั่นของเส้น
        - ทำ Sanity Check เพื่อกรองค่าที่ผิดปกติออกไป
        """
        self.history_length = history_length
        # ประวัติของสมการเส้นซ้ายและขวา
        self.left_fit_history = []
        self.right_fit_history = []
        # สมการล่าสุดที่ผ่านการเฉลี่ยแล้ว
        self.current_left_fit = None
        self.current_right_fit = None
        # สถานะว่าตรวจเจอเส้นในเฟรมล่าสุดหรือไม่
        self.detected = False

    def process_fits(self, left_fit, right_fit, img_shape):
        """
        รับค่า fit จากเฟรมปัจจุบัน, ทำ Sanity Check, อัปเดตประวัติ, และคืนค่า fit ที่ผ่านการเฉลี่ยแล้ว
        """

        if left_fit is None and right_fit is None:
            self.detected = False
            return self.current_left_fit, self.current_right_fit

        # 1. บังคับระยะห่างของเลนให้คงที่เสมอ (ลดลงเหลือ 45% ของความกว้างภาพ เพื่อให้กรอบเขียวแคบลงนิดนึง)
        expected_width = int(img_shape[1] * 0.45)
        y_bottom = img_shape[0] - 1

        # คัดลอกค่าออกมาเป็นตัวแปรใหม่ เพื่อป้องกันไม่ให้กระทบกับประวัติเก่า (แก้ปัญหาไม่มีเส้นสีเขียว)
        new_left_fit = np.copy(left_fit) if left_fit is not None else None
        new_right_fit = np.copy(right_fit) if right_fit is not None else None

        # 2. ถ้าเจอทั้งสองเส้น ปล่อยให้สมการเป็นอิสระ (เส้นในและเส้นนอกโค้งมีความโค้งไม่เท่ากัน)
        if new_left_fit is not None and new_right_fit is not None:
            # เพิ่มเกราะป้องกัน: ถ้าเส้นขวาความโค้งกระโดดจากเส้นซ้ายมากไป (มักเกิดกับเส้นประ)
            # ให้บังคับเส้นขวาขนานกับเส้นซ้าย (Parallel Lane Enforcement)
            if abs(new_left_fit[0] - new_right_fit[0]) > 0.002:
                new_right_fit[0] = new_left_fit[0] # ยืมค่าความโค้ง (A) จากเส้นซ้าย
                new_right_fit[1] = new_left_fit[1] # ยืมค่าความชัน (B) จากเส้นซ้าย

        # 3. กรณีเจอแค่เส้นเดียว ก็ใช้วิธีล็อกความกว้างสร้างเส้นฝั่งตรงข้ามขึ้นมาเลย
        elif new_left_fit is not None and new_right_fit is None:
            new_right_fit = np.copy(new_left_fit)
            new_right_fit[2] += expected_width
        elif new_right_fit is not None and new_left_fit is None:
            new_left_fit = np.copy(new_right_fit)
            new_left_fit[2] -= expected_width

        # Sanity Check
        if self._sanity_check_ok(new_left_fit, new_right_fit, img_shape):
            self.detected = True
            self.left_fit_history.append(new_left_fit)
            self.right_fit_history.append(new_right_fit)

            # รักษาความยาวของประวัติ (ลบอันที่เก่าที่สุดออก)
            if len(self.left_fit_history) > self.history_length:
                self.left_fit_history.pop(0)
                self.right_fit_history.pop(0)
            
            # Weighted Moving Average: ลดน้ำหนักของเฟรมใหม่ลงเล็กน้อย ไม่ให้มันดึงเส้นเร็ว/ไวเกินไป
            n = len(self.left_fit_history)
            weights = np.array([1.2**i for i in range(n)], dtype=np.float64)
            weights /= weights.sum()
            new_left = np.average(self.left_fit_history, axis=0, weights=weights)
            new_right = np.average(self.right_fit_history, axis=0, weights=weights)

            # Coefficient Clamping: จำกัดไม่ให้ค่าความโค้ง (A) และความชัน (B) เปลี่ยนกระโดดมากเกินไป
            # ช่วยล็อกให้เส้นนิ่งขึ้น ป้องกันอาการเส้นหลอนหรือสั่นกระตุก
            if self.current_left_fit is not None:
                max_a_change = 0.0015 # ลดลงจาก 0.005 ให้ความโค้งค่อยๆ เปลี่ยนอย่างนุ่มนวล
                max_b_change = 0.1    # ลดลงจาก 0.5 ให้ทิศทางเส้นไม่สวิงซ้ายขวาไวไป
                for fit_new, fit_old in [(new_left, self.current_left_fit), (new_right, self.current_right_fit)]:
                    if fit_old is not None:
                        fit_new[0] = np.clip(fit_new[0], fit_old[0] - max_a_change, fit_old[0] + max_a_change)
                        fit_new[1] = np.clip(fit_new[1], fit_old[1] - max_b_change, fit_old[1] + max_b_change)

            self.current_left_fit = new_left
            self.current_right_fit = new_right

        else:
            self.detected = False

        return self.current_left_fit, self.current_right_fit

    def _sanity_check_ok(self, left_fit, right_fit, img_shape):
        """
        ตรวจสอบความสมเหตุสมผลของเส้นที่หาได้
        1. ตรวจสอบว่าหาเจอทั้งสองเส้นหรือไม่
        2. ตรวจสอบว่าระยะห่างระหว่างเส้นอยู่ในเกณฑ์มาตรฐานหรือไม่ (ประมาณ 300-600 pixels ใน Bird's-eye view)
        3. ตรวจสอบความแตกต่างของค่าสัมประสิทธิ์ (ความโค้ง, ความชัน) ระหว่างเส้นซ้ายและขวา
        4. ตรวจสอบความต่อเนื่องของเส้นเลนจากเฟรมก่อนหน้า (ถ้ามี)
        """
        if left_fit is None or right_fit is None:
            return False

        h = img_shape[0]
        ploty = np.linspace(0, h - 1, h)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # 2. ตรวจสอบความกว้างของเลน (Lane Width) ที่ส่วนล่างของภาพ
        # แก้ปัญหาไม่มีเส้นสีเขียว: ปรับจากพิกเซลคงที่ เป็นสัดส่วน % ของความกว้างหน้าจอ (รองรับวิดีโอความละเอียดสูง)
        lane_width_bottom = right_fitx[-1] - left_fitx[-1]
        min_width = img_shape[1] * SANITY_MIN_WIDTH_RATIO
        max_width = img_shape[1] * SANITY_MAX_WIDTH_RATIO
        if not (min_width < lane_width_bottom < max_width):
            return False

        # 3. ตรวจสอบความแตกต่างของค่าสัมประสิทธิ์ (Curvature and Slope Similarity)
        # ค่า A (left_fit[0], right_fit[0]) บ่งบอกถึงความโค้ง
        # ค่า B (left_fit[1], right_fit[1]) บ่งบอกถึงความชัน
        # ค่า C (left_fit[2], right_fit[2]) บ่งบอกถึงตำแหน่งเริ่มต้น

        # ตรวจสอบความแตกต่างของค่า A (ความโค้ง)
        if abs(left_fit[0] - right_fit[0]) > SANITY_MAX_CURVATURE_DIFF:
            return False

        # ตรวจสอบความแตกต่างของค่า B (ความชัน)
        if abs(left_fit[1] - right_fit[1]) > SANITY_MAX_SLOPE_DIFF:
            return False

        # 4. ตรวจสอบความต่อเนื่องจากเฟรมก่อนหน้า (ถ้ามี)
        if self.detected and self.current_left_fit is not None and self.current_right_fit is not None:
            # ตรวจสอบว่าค่าสัมประสิทธิ์ปัจจุบันไม่ต่างจากค่าเฉลี่ยในประวัติมากเกินไป
            # ปรับค่าจำกัดระยะแกน X (C) ให้ยืดหยุ่นตามความกว้างภาพ
            max_c_diff = img_shape[1] * 0.40 # 40% of image width
            if abs(left_fit[0] - self.current_left_fit[0]) > HISTORY_MAX_CURVATURE_DIFF or \
               abs(left_fit[1] - self.current_left_fit[1]) > HISTORY_MAX_SLOPE_DIFF or \
               abs(left_fit[2] - self.current_left_fit[2]) > max_c_diff:
                return False
            if abs(right_fit[0] - self.current_right_fit[0]) > HISTORY_MAX_CURVATURE_DIFF or \
               abs(right_fit[1] - self.current_right_fit[1]) > HISTORY_MAX_SLOPE_DIFF or \
               abs(right_fit[2] - self.current_right_fit[2]) > max_c_diff:
                return False

        return True