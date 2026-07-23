import cv2
import numpy as np
from config import DRIFT_THRESHOLD, DRIFT_THRESHOLD_OFF, DRIFT_THRESHOLD_ON, OVERLAY_ALPHA, CAMERA_OFFSET


def draw_lane_overlay(img, left_line, right_line):
    """
    วาด overlay สีเขียว + เส้นเลนซ้าย(น้ำเงิน)/ขวา(แดง)
    + ข้อความ warning และค่า offset

    [แก้ไข] offset คำนวณจาก smoothed line เสมอ
    Warning และ offset ที่แสดงจึงตรงกันทั้ง Original และ Final Result
    """
    overlay = np.zeros_like(img)
    result  = img.copy()

    if left_line is not None and right_line is not None:
        # --- วาด polygon พื้นที่เลน ---
        pts = np.array([[
            (left_line[0],  left_line[1]),
            (left_line[2],  left_line[3]),
            (right_line[2], right_line[3]),
            (right_line[0], right_line[1]),
        ]], dtype=np.int32)
        cv2.fillPoly(overlay, pts, (0, 255, 0))

        # --- วาดเส้นเลน ---
        cv2.line(overlay,
                 (left_line[0],  left_line[1]),
                 (left_line[2],  left_line[3]),
                 (255, 0, 0), 12)
        cv2.line(overlay,
                 (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]),
                 (0, 0, 255), 12)

        result = cv2.addWeighted(result, 1, overlay, OVERLAY_ALPHA, 0)

        # --- offset ชดเชยตำแหน่งกล้อง ---
        lane_center   = (left_line[0] + right_line[0]) / 2
        camera_center = img.shape[1] / 2 + CAMERA_OFFSET
        offset        = lane_center - camera_center

        if offset > DRIFT_THRESHOLD:
            status, color = "Warning: Drifting LEFT!",  (0, 0, 255)
        elif offset < -DRIFT_THRESHOLD:
            status, color = "Warning: Drifting RIGHT!", (0, 0, 255)
        else:
            status, color = "Status: Safe",             (0, 255, 0)

        cv2.putText(result, status,
                    (50, 120),  cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3, cv2.LINE_AA)
        cv2.putText(result, f"Offset: {int(offset)}px",
                    (50, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result, "WARNING: Lane Not Detected",
                    (50, 120),  cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)

    return result


def build_roi_display(roi_left, roi_right, left_vertices, right_vertices):
    """รวม ROI ทั้งสองฝั่งเป็นภาพเดียว พร้อมวาดกรอบ"""
    roi_combined = cv2.bitwise_or(roi_left, roi_right)
    display      = cv2.cvtColor(roi_combined, cv2.COLOR_GRAY2BGR)
    cv2.polylines(display, left_vertices,  isClosed=True,
                  color=(255, 200, 0), thickness=2)
    cv2.polylines(display, right_vertices, isClosed=True,
                  color=(0, 165, 255), thickness=2)
    return display


class WarningState:
    def __init__(self):
        self.is_warning = False

    def update(self, offset):
        if abs(offset) > DRIFT_THRESHOLD_ON:
            self.is_warning = True
        elif abs(offset) < DRIFT_THRESHOLD_OFF:
            self.is_warning = False
        # ระหว่าง 40-60px ให้คงสถานะเดิม (ไม่กะพริบ)
        return self.is_warning