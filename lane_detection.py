import cv2
import numpy as np

from preprocessor import preprocess_union, detect_edges
from preprocessor import region_of_interest_left, region_of_interest_right
from lane_fitter   import get_hough_lines, fit_lane_ransac, get_confidence
from smoother      import LaneSmoother
from visualizer    import draw_lane_overlay, build_roi_display

# ===========================================================================
# Global smoother
# ===========================================================================
_smoother = LaneSmoother()


def reset():
    _smoother.reset()


# ===========================================================================
# Full Pipeline
# ===========================================================================

def full_pipeline(img):
    """
    Pipeline หลัก — 2 ROI + union color filter

    คืน: (result, blur, edges, roi_display, smoothed_offset)
    """
    h, w = img.shape[:2]

    # 1. Color filter: white + yellow รวมกัน (union)
    blur = preprocess_union(img)

    # 2. Canny
    edges = detect_edges(blur)

    # 3. ROI แยกซ้าย/ขวา
    roi_left,  left_vertices  = region_of_interest_left(edges,  img.shape)
    roi_right, right_vertices = region_of_interest_right(edges, img.shape)

    # 4. Hough แยกซ้าย/ขวา
    lines_left  = get_hough_lines(roi_left)
    lines_right = get_hough_lines(roi_right)

    # 5. RANSAC
    left_line_raw  = fit_lane_ransac(lines_left,  slope_sign=-1, img_shape=img.shape)
    right_line_raw = fit_lane_ransac(lines_right, slope_sign=+1, img_shape=img.shape)

    # 6. Confidence
    conf_left  = get_confidence(lines_left,  -1)
    conf_right = get_confidence(lines_right, +1)

    # 7. Kalman smoothing
    left_line, right_line = _smoother.update(
        left_line_raw, right_line_raw,
        conf_left=conf_left,
        conf_right=conf_right,
        img_width=w,
    )

    # 8. Offset (ชดเชยตำแหน่งกล้อง)
    smoothed_offset = None
    if left_line is not None and right_line is not None:
        from config import CAMERA_OFFSET
        lane_center   = (left_line[0] + right_line[0]) / 2
        camera_center = w / 2 + CAMERA_OFFSET
        smoothed_offset = int(lane_center - camera_center)

    # 9. Visualize
    result = draw_lane_overlay(img, left_line, right_line)

    # ROI display รวมทั้งสอง
    roi_display = build_roi_display(
        roi_left, roi_right, left_vertices, right_vertices
    )

    return result, blur, edges, roi_display, smoothed_offset