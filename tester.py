import cv2
import numpy as np
import os
import shutil

from lane_detection import full_pipeline, reset
from preprocessor import _white_mask, _yellow_mask
from config import DRIFT_THRESHOLD

# ===========================================================================
# helpers
# ===========================================================================

def _label(img, text, sub=""):
    """วาด header bar พร้อม title และ subtitle"""
    out = img.copy()
    bar_h = 48 if sub else 36
    cv2.rectangle(out, (0, 0), (img.shape[1], bar_h), (15, 15, 30), -1)
    cv2.putText(out, text, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    return out


def _to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# ===========================================================================
# Full Comparison Grid (2 rows × 4 cols)
# ===========================================================================
# Row 0: Original | White Mask | Yellow Mask | Union Mask
# Row 1: Canny Edges | Dual ROI | Hough Lines | Final Result
# ===========================================================================

def _make_full_grid(frame, result, edges, roi_display,
                    lines_left, lines_right,
                    left_vertices, right_vertices,
                    smoothed_offset,
                    scale=0.38):
    """สร้าง 2×4 comparison grid"""

    def rs(img):
        return cv2.resize(img, (0, 0), fx=scale, fy=scale)

    h, w = frame.shape[:2]

    # ── Cell 0: Original + status overlay ─────────────────────────────────
    orig = frame.copy()
    if smoothed_offset is not None:
        if smoothed_offset > DRIFT_THRESHOLD:
            status, sc = "Drifting LEFT!", (0, 50, 255)
        elif smoothed_offset < -DRIFT_THRESHOLD:
            status, sc = "Drifting RIGHT!", (0, 50, 255)
        else:
            status, sc = "Safe", (0, 210, 60)
        cv2.putText(orig, status, (18, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, sc, 2, cv2.LINE_AA)
        cv2.putText(orig, f"Offset: {smoothed_offset}px", (18, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(orig, "Lane Not Detected", (18, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 50, 255), 2, cv2.LINE_AA)

    # ── Cell 1: White Mask ─────────────────────────────────────────────────
    white_bgr = _to_bgr(_white_mask(frame))

    # ── Cell 2: Yellow Mask ────────────────────────────────────────────────
    yellow_bgr = _to_bgr(_yellow_mask(frame))

    # ── Cell 3: Union Mask ─────────────────────────────────────────────────
    union_mask = cv2.bitwise_or(_white_mask(frame), _yellow_mask(frame))
    union_bgr  = _to_bgr(union_mask)

    # ── Cell 4: Canny Edges ────────────────────────────────────────────────
    edges_bgr = _to_bgr(edges)

    # ── Cell 5: Dual ROI + polygon boundaries ─────────────────────────────
    roi_vis = roi_display.copy()   # already BGR with colored borders

    # ── Cell 6: Hough Lines on original ───────────────────────────────────
    hough_vis = frame.copy()
    if lines_left is not None:
        for ln in lines_left:
            x1, y1, x2, y2 = ln[0]
            cv2.line(hough_vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
    if lines_right is not None:
        for ln in lines_right:
            x1, y1, x2, y2 = ln[0]
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.polylines(hough_vis, left_vertices,  True, (255, 200, 0), 2)
    cv2.polylines(hough_vis, right_vertices, True, (0, 165, 255), 2)
    n_l = len(lines_left)  if lines_left  is not None else 0
    n_r = len(lines_right) if lines_right is not None else 0
    cv2.putText(hough_vis, f"L:{n_l}  R:{n_r} lines",
                (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 80), 2, cv2.LINE_AA)

    # ── Cell 7: Final Result ───────────────────────────────────────────────
    final = result.copy()

    # ── labels (title, subtitle) ───────────────────────────────────────────
    cells = [
        (orig,       "0 - Original",        "Input Frame"),
        (white_bgr,  "1a - White Mask",     "HLS: L>=120, S<=80"),
        (yellow_bgr, "1b - Yellow Mask",    "HLS: H=10-45"),
        (union_bgr,  "1c - Union Filter",   "White + Yellow Union, Blur 5x5"),
        (edges_bgr,  "2 - Canny Edges",     f"Thresh: {30}|{100}"),
        (roi_vis,    "3 - Dual ROI",        "Left x=10-50%  Right x=50-100%"),
        (hough_vis,  "4 - Hough Lines",     "RANSAC Lane Fitting"),
        (final,      "5 - Kalman + Result", "Smoothed + Lane Overlay"),
    ]

    panels = [_label(rs(img), t, s) for img, t, s in cells]

    # ── assemble 2×4 ──────────────────────────────────────────────────────
    row0 = cv2.hconcat(panels[0:4])
    row1 = cv2.hconcat(panels[4:8])

    # ensure same width (shouldn't differ, but guard)
    if row0.shape[1] != row1.shape[1]:
        target_w = max(row0.shape[1], row1.shape[1])
        row0 = cv2.resize(row0, (target_w, row0.shape[0]))
        row1 = cv2.resize(row1, (target_w, row1.shape[0]))

    grid = cv2.vconcat([row0, row1])

    # ── bottom title bar ──────────────────────────────────────────────────
    bar = np.full((36, grid.shape[1], 3), (15, 15, 30), dtype=np.uint8)
    title = ("Lane Detection Pipeline  |  "
             "Union Color Filter  -  Dual ROI  -  Hough  -  RANSAC  -  Kalman Smoothing")
    cv2.putText(bar, title, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1, cv2.LINE_AA)

    return cv2.vconcat([grid, bar])


# ===========================================================================
# helper ที่ pipeline ต้องการ (expose lines + vertices)
# ===========================================================================

def _full_pipeline_verbose(img):
    """
    เหมือน full_pipeline แต่คืน lines + vertices ด้วย
    เพื่อให้ _make_full_grid วาด Hough cell ได้
    """
    from preprocessor import preprocess_union, detect_edges
    from preprocessor import region_of_interest_left, region_of_interest_right
    from lane_fitter   import get_hough_lines, fit_lane_ransac, get_confidence
    from visualizer    import draw_lane_overlay, build_roi_display
    from lane_detection import _smoother

    h, w = img.shape[:2]

    blur  = preprocess_union(img)
    edges = detect_edges(blur)

    roi_left,  left_verts  = region_of_interest_left(edges,  img.shape)
    roi_right, right_verts = region_of_interest_right(edges, img.shape)

    lines_left  = get_hough_lines(roi_left)
    lines_right = get_hough_lines(roi_right)

    left_raw  = fit_lane_ransac(lines_left,  slope_sign=-1, img_shape=img.shape)
    right_raw = fit_lane_ransac(lines_right, slope_sign=+1, img_shape=img.shape)

    conf_left  = get_confidence(lines_left,  -1)
    conf_right = get_confidence(lines_right, +1)

    left_line, right_line = _smoother.update(
        left_raw, right_raw,
        conf_left=conf_left, conf_right=conf_right, img_width=w,
    )

    smoothed_offset = None
    if left_line is not None and right_line is not None:
        from config import CAMERA_OFFSET
        lane_center   = (left_line[0] + right_line[0]) / 2
        camera_center = w / 2 + CAMERA_OFFSET
        smoothed_offset = int(lane_center - camera_center)

    result      = draw_lane_overlay(img, left_line, right_line)
    roi_display = build_roi_display(roi_left, roi_right, left_verts, right_verts)

    return (result, blur, edges, roi_display, smoothed_offset,
            lines_left, lines_right, left_verts, right_verts)


# ===========================================================================
# โหมด 1: ทดสอบภาพเดี่ยว
# ===========================================================================

def test_single_image(img_path, save_path=None):
    import matplotlib.pyplot as plt

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์: {img_path}")

    reset()
    (result, blur, edges, roi_display, smoothed_offset,
     lines_left, lines_right, left_verts, right_verts) = _full_pipeline_verbose(img)

    grid = _make_full_grid(img, result, edges, roi_display,
                           lines_left, lines_right,
                           left_verts, right_verts,
                           smoothed_offset, scale=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, grid)
        print(f"บันทึกภาพที่ {save_path}")

    cv2.imshow("Lane Detection Pipeline", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===========================================================================
# โหมด 2: ทดสอบวิดีโอ — บันทึกเฉพาะ Final Result
# ===========================================================================

def test_video(video_path, output_path="output.mp4", show_window=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่พบวิดีโอ: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (w, h))
    reset()

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        (result, _, edges, roi_display, smoothed_offset,
         lines_left, lines_right, left_verts, right_verts) = _full_pipeline_verbose(frame)

        # บันทึกเฉพาะ result (ขนาดเท่า input)
        writer.write(result)

        if show_window:
            grid = _make_full_grid(frame, result, edges, roi_display,
                                   lines_left, lines_right,
                                   left_verts, right_verts,
                                   smoothed_offset, scale=0.38)
            cv2.imshow("Lane Detection Pipeline  (Q = quit)", grid)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("หยุดโดยผู้ใช้")
                break

        count += 1
        if count % 30 == 0:
            print(f"  ประมวลผล {count}/{total} เฟรม")

    cap.release()
    writer.release()
    if show_window:
        cv2.destroyAllWindows()
    print(f"บันทึกวิดีโอที่ {output_path}  ({count} เฟรม, {w}x{h})")


# ===========================================================================
# โหมด 3: แยก frame จากวิดีโอ
# ===========================================================================

def extract_frames(video_path, output_dir, interval=30):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่พบวิดีโอ: {video_path}")

    reset()
    count = saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            (result, _, edges, roi_display, smoothed_offset,
             lines_left, lines_right, left_verts, right_verts) = _full_pipeline_verbose(frame)

            grid = _make_full_grid(frame, result, edges, roi_display,
                                   lines_left, lines_right,
                                   left_verts, right_verts,
                                   smoothed_offset, scale=0.5)

            path = os.path.join(output_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(path, grid)
            saved += 1

        count += 1

    cap.release()
    print(f"บันทึก {saved} เฟรม ที่ {output_dir}")


# ===========================================================================
# เรียกใช้งาน
# ===========================================================================

if __name__ == "__main__":

    test_video(
        video_path  = "testcase2.mp4",
        output_path = "result/output.mp4",
        show_window = True,
    )

    extract_frames(
        video_path = "testcase2.mp4",
        output_dir = "result/frames",
        interval   = 30,
    )