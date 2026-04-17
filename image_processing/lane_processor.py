import numpy as np

class LaneProcessor:
    def __init__(self, history_length=7):
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

        # Sanity Check
        if self._sanity_check_ok(left_fit, right_fit, img_shape):
            self.detected = True
            # ถ้าผ่าน, เพิ่ม fit ปัจจุบันลงในประวัติ
            self.left_fit_history.append(left_fit)
            self.right_fit_history.append(right_fit)

            # รักษาความยาวของประวัติ (ลบอันที่เก่าที่สุดออก)
            if len(self.left_fit_history) > self.history_length:
                self.left_fit_history.pop(0)
                self.right_fit_history.pop(0)
            
            # คำนวณค่าเฉลี่ยจากประวัติทั้งหมดเพื่อหาค่าปัจจุบัน
            self.current_left_fit = np.mean(self.left_fit_history, axis=0)
            self.current_right_fit = np.mean(self.right_fit_history, axis=0)

        else:
            # ถ้าไม่ผ่าน Sanity Check
            self.detected = False
            # เมื่อ Sanity Check ไม่ผ่าน, เราจะไม่ทำการอัปเดตใดๆ กับ self.current_left_fit
            # และ self.current_right_fit มันจะยังคงเป็นค่าล่าสุดที่ผ่านการตรวจสอบ
            # ซึ่งจะช่วยให้เส้นเลน "ค้าง" อยู่ในตำแหน่งที่ดีล่าสุด แทนที่จะกระตุกหรือหายไป
            # หากยังไม่เคยมีค่าที่ดีเลย (เช่น เฟรมแรกๆ) ค่าก็จะเป็น None อยู่แล้ว
            pass

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
        lane_width_bottom = right_fitx[-1] - left_fitx[-1]
        if not (200 < lane_width_bottom < 850): 
            return False

        # 3. ตรวจสอบความแตกต่างของค่าสัมประสิทธิ์ (Curvature and Slope Similarity)
        # ค่า A (left_fit[0], right_fit[0]) บ่งบอกถึงความโค้ง
        # ค่า B (left_fit[1], right_fit[1]) บ่งบอกถึงความชัน
        # ค่า C (left_fit[2], right_fit[2]) บ่งบอกถึงตำแหน่งเริ่มต้น
        
        # ตรวจสอบความแตกต่างของค่า A (ความโค้ง)
        if abs(left_fit[0] - right_fit[0]) > 0.005: # เพิ่มความยืดหยุ่นให้เส้นโค้งสองฝั่งไม่ต้องเป๊ะมาก
            return False

        # ตรวจสอบความแตกต่างของค่า B (ความชัน)
        if abs(left_fit[1] - right_fit[1]) > 0.5: # เพิ่มความยืดหยุ่นให้ความชันของเส้นซ้าย-ขวา
            return False

        # 4. ตรวจสอบความต่อเนื่องจากเฟรมก่อนหน้า (ถ้ามี)
        if self.detected and self.current_left_fit is not None and self.current_right_fit is not None:
            # ตรวจสอบว่าค่าสัมประสิทธิ์ปัจจุบันไม่ต่างจากค่าเฉลี่ยในประวัติมากเกินไป
            # (ค่า 0.5, 100 อาจต้องปรับจูนตามความเหมาะสม)
            if abs(left_fit[0] - self.current_left_fit[0]) > 0.002 or \
               abs(left_fit[1] - self.current_left_fit[1]) > 0.2 or \
               abs(left_fit[2] - self.current_left_fit[2]) > 200:
                return False
            if abs(right_fit[0] - self.current_right_fit[0]) > 0.002 or \
               abs(right_fit[1] - self.current_right_fit[1]) > 0.2 or \
               abs(right_fit[2] - self.current_right_fit[2]) > 200:
                return False
            
        return True