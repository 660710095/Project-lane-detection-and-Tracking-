import cv2
import numpy as np
from config import (
    HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
    HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP,
    SLOPE_MIN, SLOPE_MAX,
    RANSAC_ITERATIONS, RANSAC_THRESHOLD,
    LEFT_ROI_TOP_Y, RIGHT_ROI_TOP_Y,
)

# ===========================================================================
# Hough Lines
# ===========================================================================

def get_hough_lines(roi_edges):
    return cv2.HoughLinesP(
        roi_edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )


# ===========================================================================
# RANSAC Lane Fit
# ===========================================================================

def fit_lane_ransac(lines, slope_sign, img_shape):
    """
    slope_sign = -1  เลนซ้าย  (slope ลบ)
    slope_sign = +1  เลนขวา  (slope บวก)
    คืน (x_bottom, y_bottom, x_top, y_top) หรือ None
    """
    if lines is None:
        return None

    h, w = img_shape[:2]
    points = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)

        if slope_sign == -1 and not (-SLOPE_MAX < slope < -SLOPE_MIN):
            continue
        if slope_sign == +1 and not (SLOPE_MIN < slope < SLOPE_MAX):
            continue

        # x-zone filter
        x_mid = (x1 + x2) / 2
        if slope_sign == -1 and not (w * 0.0 < x_mid < w * 0.60):
            continue
        if slope_sign == +1 and not (w * 0.40 < x_mid < w * 1.0):
            continue

        points.extend([(x1, y1), (x2, y2)])

    if len(points) < 4:
        return None

    pts = np.array(points, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]

    best_inliers = []
    best_coeffs  = None

    for _ in range(RANSAC_ITERATIONS):
        idx = np.random.choice(len(pts), 2, replace=False)
        if ys[idx[0]] == ys[idx[1]]:
            continue
        try:
            coeffs = np.polyfit(ys[idx], xs[idx], 1)
        except np.RankWarning:
            continue

        residual = np.abs(xs - np.polyval(coeffs, ys))
        inliers  = np.where(residual < RANSAC_THRESHOLD)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_coeffs  = coeffs

    if best_coeffs is None or len(best_inliers) < 2:
        return None

    try:
        final_coeffs = np.polyfit(ys[best_inliers], xs[best_inliers], 1)
    except np.RankWarning:
        final_coeffs = best_coeffs

    roi_top_y = LEFT_ROI_TOP_Y if slope_sign == -1 else RIGHT_ROI_TOP_Y
    y_bottom  = h
    y_top     = int(h * roi_top_y)
    x_bottom  = int(np.polyval(final_coeffs, y_bottom))
    x_top     = int(np.polyval(final_coeffs, y_top))

    if slope_sign == -1 and not (-w * 0.1 < x_bottom < w * 0.60):
        return None
    if slope_sign == +1 and not (w * 0.40 < x_bottom < w * 1.1):
        return None

    return (x_bottom, y_bottom, x_top, y_top)


# ===========================================================================
# Confidence Score
# ===========================================================================

def get_confidence(lines, slope_sign):
    if lines is None or len(lines) == 0:
        return 0.0
    valid = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope_sign == -1 and (-SLOPE_MAX < slope < -SLOPE_MIN):
            valid += 1
        elif slope_sign == +1 and (SLOPE_MIN < slope < SLOPE_MAX):
            valid += 1
    return valid / len(lines)
