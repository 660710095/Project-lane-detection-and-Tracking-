import cv2
import numpy as np
from config import (
    BLUR_KERNEL_SIZE,
    CANNY_THRESHOLD1, CANNY_THRESHOLD2,
    LEFT_ROI_TOP_Y,  LEFT_ROI_TOP_LEFT_X,  LEFT_ROI_TOP_RIGHT_X,
    LEFT_ROI_BOT_LEFT_X,  LEFT_ROI_BOT_RIGHT_X,
    RIGHT_ROI_TOP_Y, RIGHT_ROI_TOP_LEFT_X, RIGHT_ROI_TOP_RIGHT_X,
    RIGHT_ROI_BOT_LEFT_X, RIGHT_ROI_BOT_RIGHT_X,
)

# ===========================================================================
# Color Filter (HLS — ดีกว่า HSV ในแสงจ้า)
# ===========================================================================

def _white_mask(img):
    hls  = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls,
                       np.array([0,  120,  0]),   # L>=120 (ลดจาก 150)
                       np.array([180, 255, 80]))   # S<=80  (เพิ่มจาก 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def _yellow_mask(img):
    hls  = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls,
                       np.array([10,  80, 40]),   # H=10 L>=80 S>=40 (หลวมขึ้น)
                       np.array([45, 255, 255]))   # H=45 (ขยายจาก 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def _to_blur(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask)
    gray   = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)


def preprocess_white(img):
    """กรองเฉพาะสีขาว"""
    if img is None:
        return None
    return _to_blur(img, _white_mask(img))


def preprocess_yellow(img):
    """กรองเฉพาะสีเหลือง"""
    if img is None:
        return None
    return _to_blur(img, _yellow_mask(img))


def preprocess_union(img):
    """
    [แก้ไข ขั้นที่ 2] Union filter — จับทั้งขาวและเหลือง
    ใช้กับทั้ง 2 ROI แทนการผูกสีกับฝั่ง
    ทำให้ detect ได้แม้เส้นเลนสลับสี
    """
    if img is None:
        return None
    combined = cv2.bitwise_or(_white_mask(img), _yellow_mask(img))
    return _to_blur(img, combined)


# ===========================================================================
# Canny
# ===========================================================================

def detect_edges(blur):
    if blur is None:
        return None
    return cv2.Canny(blur, CANNY_THRESHOLD1, CANNY_THRESHOLD2)


# ===========================================================================
# ROI Mask — 2 ROI (ซ้าย/ขวา)
# ===========================================================================

def region_of_interest_left(edges, img_shape):
    """ROI ซ้าย — trapezoid ฝั่งซ้าย"""
    h, w = img_shape[:2]
    vertices = np.array([[
        (int(w * LEFT_ROI_BOT_LEFT_X),  h),
        (int(w * LEFT_ROI_TOP_LEFT_X),  int(h * LEFT_ROI_TOP_Y)),
        (int(w * LEFT_ROI_TOP_RIGHT_X), int(h * LEFT_ROI_TOP_Y)),
        (int(w * LEFT_ROI_BOT_RIGHT_X), h),
    ]], dtype=np.int32)

    if edges is None:
        blank = np.zeros((h, w), dtype=np.uint8)
        return blank, vertices

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask), vertices


def region_of_interest_right(edges, img_shape):
    """ROI ขวา — trapezoid ฝั่งขวา"""
    h, w = img_shape[:2]
    vertices = np.array([[
        (int(w * RIGHT_ROI_BOT_LEFT_X),  h),
        (int(w * RIGHT_ROI_TOP_LEFT_X),  int(h * RIGHT_ROI_TOP_Y)),
        (int(w * RIGHT_ROI_TOP_RIGHT_X), int(h * RIGHT_ROI_TOP_Y)),
        (int(w * RIGHT_ROI_BOT_RIGHT_X), h),
    ]], dtype=np.int32)

    if edges is None:
        blank = np.zeros((h, w), dtype=np.uint8)
        return blank, vertices

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask), vertices
