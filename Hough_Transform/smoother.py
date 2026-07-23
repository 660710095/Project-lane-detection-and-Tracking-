import numpy as np
from config import SMOOTH_CONF_MIN, SMOOTH_MAX_JUMP

# ============================================================
# Kalman Filter สำหรับ 1 ตำแหน่ง (x_bottom, y_bottom, x_top, y_top)
# state = [x_bottom, x_top]  (y คงที่ ไม่ต้อง filter)
# ============================================================

class KalmanLane:
    """
    Kalman filter 2-state: [x_bottom, x_top]
    - process noise Q  : ความผันผวนที่คาดว่าเส้นเลนจะขยับต่อเฟรม
    - measurement noise R : ความไม่แน่นอนของ RANSAC output
    """
    def __init__(self, q=50.0, r=30.0):
        self.Q = np.eye(2) * q          # process noise covariance
        self.R = np.eye(2) * r          # measurement noise covariance
        self.x = None                   # state estimate [x_bottom, x_top]
        self.P = np.eye(2) * 500.0      # initial estimation covariance (ไม่แน่ใจมาก)
        self.F = np.eye(2)              # state transition (constant position model)
        self.H = np.eye(2)              # observation matrix

    def reset(self):
        self.x = None
        self.P = np.eye(2) * 500.0

    def predict(self):
        if self.x is None:
            return None
        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement):
        """
        measurement: [x_bottom, x_top]
        คืน smoothed state
        """
        z = np.array(measurement, dtype=np.float64)

        if self.x is None:
            # Initialize ด้วย measurement แรก
            self.x = z.copy()
            self.P = np.eye(2) * 500.0
            return self.x.copy()

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        y = z - self.H @ x_pred
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self.x.copy()

    def get_state(self):
        return self.x.copy() if self.x is not None else None


# ============================================================
# LaneSmoother (ใช้ Kalman แทน EMA)
# ============================================================

class LaneSmoother:
    """
    Kalman-based smoother แทน EMA เดิม
    - ไม่มี lag เหมือน EMA
    - จัดการ missing measurement ด้วย predict step
    - confidence gate ยังคงอยู่
    - sanity check ยังคงอยู่
    """
    def __init__(self,
                 q=50.0,
                 r=30.0,
                 max_jump=SMOOTH_MAX_JUMP,
                 conf_min=SMOOTH_CONF_MIN,
                 max_missing=25,        # reset Kalman หลัง missing เกิน 25 frame (~0.8 วินาที)
                 outlier_thresh=150,    # px — ต่างจาก state เกินนี้ = outlier
                 outlier_max_skip=5):   # skip ต่อเนื่องเกินนี้ = เลนเปลี่ยนจริง
        self.max_jump       = max_jump
        self.conf_min       = conf_min
        self.max_missing    = max_missing
        self.outlier_thresh = outlier_thresh
        self.outlier_max_skip = outlier_max_skip
        self._img_width     = None

        self._kf_left  = KalmanLane(q=q, r=r)
        self._kf_right = KalmanLane(q=q, r=r)

        self._prev_left_y   = None
        self._prev_right_y  = None

        # missing frame counter
        self._missing_left  = 0
        self._missing_right = 0

        # outlier skip counter
        self._skip_left  = 0
        self._skip_right = 0

    # ----------------------------------------------------------
    # internal helpers
    # ----------------------------------------------------------

    def _is_sane(self, line, slope_sign):
        """sanity check x_bottom"""
        if line is None or self._img_width is None:
            return True
        w = self._img_width
        x_bottom = line[0]
        if slope_sign == -1:
            return -w * 0.1 < x_bottom < w * 0.65
        else:
            return w * 0.35 < x_bottom < w * 1.1

    def _clamp_jump(self, new_x, prev_x):
        """จำกัดการกระโดดต่อเฟรม (ใช้กับ measurement ก่อน update Kalman)"""
        if prev_x is None:
            return new_x
        diff = new_x - prev_x
        diff = max(-self.max_jump, min(self.max_jump, diff))
        return prev_x + diff

    def _is_outlier(self, line, kf, skip_counter):
        """
        ตรวจว่า measurement ห่างจาก current state เกิน threshold ไหม
        ถ้าใช่ = outlier → ไม่ update แต่เพิ่ม skip_count
        ถ้า skip ต่อเนื่องเกิน outlier_max_skip → ยอมรับ (เลนเปลี่ยนจริง)
        """
        if line is None:
            return False, skip_counter
        state = kf.get_state()
        if state is None:
            return False, 0  # ยังไม่มี state → ไม่ใช่ outlier
        diff = abs(line[0] - state[0])
        if diff > self.outlier_thresh:
            skip_counter += 1
            if skip_counter <= self.outlier_max_skip:
                return True, skip_counter   # outlier
        else:
            skip_counter = 0
        return False, skip_counter

    def _apply_kalman(self, line, kf, prev_y_attr, missing_attr, skip_attr):
        """
        ส่ง line เข้า Kalman filter พร้อม:
        - outlier rejection
        - missing frame counter + reset
        """
        missing = getattr(self, missing_attr)
        skip    = getattr(self, skip_attr)

        if line is None:
            missing += 1
            setattr(self, missing_attr, missing)

            # reset ถ้า missing นานเกินไป
            if missing > self.max_missing:
                kf.reset()
                setattr(self, prev_y_attr, None)
                return None

            # predict ต่อ
            pred = kf.predict()
            if pred is None:
                return None
            y_b, y_t = getattr(self, prev_y_attr) or (0, 0)
            return (int(pred[0]), y_b, int(pred[1]), y_t)

        # มี measurement — ตรวจ outlier
        is_out, skip = self._is_outlier(line, kf, skip)
        setattr(self, skip_attr, skip)

        if is_out:
            # outlier → predict เฉยๆ ไม่ update
            pred = kf.predict()
            if pred is None:
                return None
            y_b, y_t = getattr(self, prev_y_attr) or (0, 0)
            return (int(pred[0]), y_b, int(pred[1]), y_t)

        # measurement ดี → reset missing counter แล้ว update
        setattr(self, missing_attr, 0)

        x_b, y_b, x_t, y_t = line
        prev_state = kf.get_state()
        if prev_state is not None:
            x_b_c = self._clamp_jump(x_b, prev_state[0])
            x_t_c = self._clamp_jump(x_t, prev_state[1])
        else:
            x_b_c, x_t_c = x_b, x_t

        smoothed = kf.update([x_b_c, x_t_c])
        setattr(self, prev_y_attr, (y_b, y_t))
        return (int(smoothed[0]), y_b, int(smoothed[1]), y_t)

    # ----------------------------------------------------------
    # public
    # ----------------------------------------------------------

    def update(self, left_line, right_line,
               conf_left=1.0, conf_right=1.0,
               img_width=None):

        if img_width is not None:
            self._img_width = img_width

        # Sanity check
        if not self._is_sane(left_line,  slope_sign=-1):
            left_line  = None
        if not self._is_sane(right_line, slope_sign=+1):
            right_line = None

        # Confidence gate
        if conf_left  < self.conf_min:
            left_line  = None
        if conf_right < self.conf_min:
            right_line = None

        # Kalman update / predict
        left_out  = self._apply_kalman(left_line,  self._kf_left,
                                       '_prev_left_y',  '_missing_left',  '_skip_left')
        right_out = self._apply_kalman(right_line, self._kf_right,
                                       '_prev_right_y', '_missing_right', '_skip_right')

        return left_out, right_out

    def reset(self):
        self._kf_left.reset()
        self._kf_right.reset()
        self._prev_left_y   = None
        self._prev_right_y  = None
        self._img_width     = None
        self._missing_left  = 0
        self._missing_right = 0
        self._skip_left     = 0
        self._skip_right    = 0
