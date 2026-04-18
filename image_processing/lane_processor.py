import numpy as np

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

        # 2. ถ้าระยะห่างเพี้ยนมาก หรือความโค้งต่างกัน ให้อิงเส้นที่แม่นยำกว่าเป็นหลัก (ล็อกเส้นไม่ให้หด/ยืด)
        if new_left_fit is not None and new_right_fit is not None:
            left_x_bottom = new_left_fit[0]*y_bottom**2 + new_left_fit[1]*y_bottom + new_left_fit[2]
            right_x_bottom = new_right_fit[0]*y_bottom**2 + new_right_fit[1]*y_bottom + new_right_fit[2]
            current_width = right_x_bottom - left_x_bottom

            # ** แก้ปัญหาเส้นเบี้ยวตอนเข้าโค้ง **
            # ลบการบังคับให้เส้นขนานกัน (copy curvature) เมื่อเจอทั้งสองเส้นทิ้งไป
            # เพราะในความเป็นจริง ตอนเข้าโค้ง เส้นเลนในและเลนนอกจะมีความโค้งไม่เท่ากัน
            # เราจะปล่อยให้สมการเป็นอิสระ และให้ Sanity Check + History เป็นตัวกรองและทำให้เส้นนิ่งแทน
            pass

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
            
            # คำนวณค่าเฉลี่ยจากประวัติทั้งหมดเพื่อหาค่าปัจจุบัน
            self.current_left_fit = np.mean(self.left_fit_history, axis=0)
            self.current_right_fit = np.mean(self.right_fit_history, axis=0)

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
        min_width = img_shape[1] * 0.30
        max_width = img_shape[1] * 0.60
        if not (min_width < lane_width_bottom < max_width): 
            return False

        # 3. ตรวจสอบความแตกต่างของค่าสัมประสิทธิ์ (Curvature and Slope Similarity)
        # ค่า A (left_fit[0], right_fit[0]) บ่งบอกถึงความโค้ง
        # ค่า B (left_fit[1], right_fit[1]) บ่งบอกถึงความชัน
        # ค่า C (left_fit[2], right_fit[2]) บ่งบอกถึงตำแหน่งเริ่มต้น
        
        # ตรวจสอบความแตกต่างของค่า A (ความโค้ง)
        if abs(left_fit[0] - right_fit[0]) > 0.015: # ขยายความยืดหยุ่นให้มากขึ้นอีก เพื่อให้รองรับโค้งที่ลึกมากๆ
            return False

        # ตรวจสอบความแตกต่างของค่า B (ความชัน)
        if abs(left_fit[1] - right_fit[1]) > 0.8: # เพิ่มความยืดหยุ่น
            return False

        # 4. ตรวจสอบความต่อเนื่องจากเฟรมก่อนหน้า (ถ้ามี)
        if self.detected and self.current_left_fit is not None and self.current_right_fit is not None:
            # ตรวจสอบว่าค่าสัมประสิทธิ์ปัจจุบันไม่ต่างจากค่าเฉลี่ยในประวัติมากเกินไป
            # ปรับค่าจำกัดระยะแกน X (C) ให้ยืดหยุ่นตามความกว้างภาพ
            max_c_diff = img_shape[1] * 0.40
            if abs(left_fit[0] - self.current_left_fit[0]) > 0.01 or \
               abs(left_fit[1] - self.current_left_fit[1]) > 1.0 or \
               abs(left_fit[2] - self.current_left_fit[2]) > max_c_diff:
                return False
            if abs(right_fit[0] - self.current_right_fit[0]) > 0.01 or \
               abs(right_fit[1] - self.current_right_fit[1]) > 1.0 or \
               abs(right_fit[2] - self.current_right_fit[2]) > max_c_diff:
                return False
            
        return True