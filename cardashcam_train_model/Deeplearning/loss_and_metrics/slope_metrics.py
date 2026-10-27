# -*- coding: utf-8 -*-
"""
metric วัดความถูกต้องของเส้นเลนจาก "ความชัน" (มุมของเส้น) แทนการนับพิกเซลทับกันแบบ IoU

ย้ายมาจากเซลล์ในโน้ตบุ๊ก train_unet_resnet34_imagenet.ipynb เพื่อให้ทั้งโน้ตบุ๊กเทรน
และ evaluate_metrics_comparison.ipynb ใช้โค้ดชุดเดียวกัน (notebook import กันเองไม่ได้)

โมดูลนี้ตั้งใจ **ไม่พึ่ง torch / segmentation_models_pytorch** จะได้ทดสอบได้เร็วโดยไม่ต้องโหลด CUDA

self-check:  python slope_metrics.py [dataset_root]
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

DEFAULT_ROOT = "/home/jovyan/shared/donut/CarDashCam/CULane"

# ค่าคงที่ทั้งหมดเลือกจากการวัดข้อมูลจริง ไม่ใช่เดา
SLOPE_MIN_AREA = 200     # median area ของเส้นจริง ~3,800 px กรอง noise เล็กออก
SLOPE_MIN_ROWS = 12      # ต้องมีจุดพอ fit เส้นตรง
SLOPE_MATCH_GATE = 150.0 # px กันจับคู่เส้นคนละฝั่งถนน
SLOPE_MAX_RMS = 10.0     # px เกินนี้ถือว่า "ความชันนิยามไม่ได้" (โค้งจัด/เลนติดกัน)
SLOPE_Y_REF_FRAC = 0.9   # จับคู่ด้วยตำแหน่ง x ที่ 90% ความสูง (ใกล้ตัวรถ แยกเลนชัดสุด)


# ======================================================================
# ส่วนหลัก: ดึงเส้นเลนและวัดมุม
# ======================================================================

def extract_lanes(binmask, min_area=SLOPE_MIN_AREA, min_rows=SLOPE_MIN_ROWS,
                  y_ref_frac=SLOPE_Y_REF_FRAC):
    """
    แยกเส้นเลนแต่ละเส้นออกจาก mask ไบนารี แล้วหาความชันของแต่ละเส้น

    คืน list ของ dict เรียงจากซ้ายไปขวา แต่ละตัวมี
      a     : ความชัน dx/dy ของเส้น (x = a*y + b)
      b     : จุดตัด
      ang   : มุมองศาในช่วง [0,180) โดย 90 = เส้นตั้งฉาก
      rms   : ค่าความคลาดของ fit (px) ยิ่งน้อยยิ่งเป็นเส้นตรง
      xref  : ตำแหน่ง x ที่ y = y_ref_frac*H ใช้สำหรับจับคู่เส้น
      area, nrows, y_top, y_bot

    mask ว่างเปล่าจะคืน list ว่าง (ไม่ error)

    ทำไมไม่ใช้ PCA: เส้นเลนใน mask เป็นแถบหนา ถ้าเอา PCA กับกลุ่มพิกเซลตรงๆ
    ความหนาและความโค้งจะดึงแกนหลักเพี้ยน (ทดลองแล้วได้มุม 5 องศาทั้งที่เป็นเส้นแนวตั้ง)
    การยุบเป็น centerline ก่อนแก้ปัญหานี้
    """
    binmask = np.asarray(binmask)
    if binmask.ndim != 2:
        binmask = binmask.reshape(binmask.shape[-2], binmask.shape[-1])
    binmask = (binmask > 0).astype(np.uint8)
    h = binmask.shape[0]
    y_ref = y_ref_frac * h

    n, lab, stats, _ = cv2.connectedComponentsWithStats(binmask, connectivity=8)
    lanes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(lab == i)
        order = np.argsort(ys, kind="stable")
        ys, xs = ys[order], xs[order]

        # centerline: จุดกลางแนวนอนของแต่ละแถว
        uy, start = np.unique(ys, return_index=True)
        if len(uy) < min_rows:
            continue
        counts = np.diff(np.append(start, len(xs)))
        cx = np.add.reduceat(xs.astype(np.float64), start) / counts

        # fit x = a*y + b (ใช้ dx/dy เพราะเส้นเลนเกือบตั้ง ถ้าใช้ dy/dx จะหารศูนย์)
        A = np.stack([uy.astype(np.float64), np.ones(len(uy))], axis=1)
        coef, *_ = np.linalg.lstsq(A, cx, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        rms = float(np.sqrt(np.mean((cx - A @ coef) ** 2)))
        ang = float(np.degrees(np.arctan2(1.0, a)) % 180.0)

        lanes.append(dict(a=a, b=b, ang=ang, rms=rms, xref=float(a * y_ref + b),
                          area=int(stats[i, cv2.CC_STAT_AREA]), nrows=int(len(uy)),
                          y_top=float(uy.min()), y_bot=float(uy.max())))

    lanes.sort(key=lambda d: d["xref"])
    return lanes


def angle_error(ang1, ang2):
    """
    ผลต่างมุมของเส้นสองเส้น พับเข้าช่วง [0, 90]
    เส้นตรงไม่มีทิศทาง (undirected) ดังนั้น 1 องศา กับ 179 องศา ต่างกันแค่ 2 องศา ไม่ใช่ 178
    """
    d = abs(float(ang1) - float(ang2)) % 180.0
    return float(min(d, 180.0 - d))


def match_lanes(gt_lanes, pred_lanes, gate=SLOPE_MATCH_GATE):
    """
    จับคู่เส้น GT กับเส้นที่ทำนาย ด้วยตำแหน่ง x ใกล้ล่างภาพ
    ใช้ Hungarian algorithm หาการจับคู่ที่ระยะรวมน้อยสุด แล้วตัดคู่ที่ไกลเกิน gate ทิ้ง
    คืน list ของ (index_gt, index_pred)
    """
    if not gt_lanes or not pred_lanes:
        return []
    g = np.array([d["xref"] for d in gt_lanes], dtype=np.float64)
    p = np.array([d["xref"] for d in pred_lanes], dtype=np.float64)
    cost = np.abs(g[:, None] - p[None, :])
    ri, ci = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if cost[i, j] <= gate]


def slope_metrics_for_pair(gt_mask, pred_mask, **kw):
    """
    วัดความชันของ 1 ภาพ
    คืน dict: n_gt, n_pred, n_matched, pairs (error ของแต่ละคู่), gt_lanes, pred_lanes, match
    """
    g = extract_lanes(gt_mask, **kw)
    p = extract_lanes(pred_mask, **kw)
    return slope_metrics_for_lanes(g, p)


def slope_metrics_for_lanes(g, p):
    """
    เหมือน slope_metrics_for_pair แต่รับ "เส้นที่ดึงมาแล้ว" ทั้งสองฝั่ง
    ใช้ตอนที่ฝั่งทำนายไม่ได้มาจาก mask เช่น baseline ที่เดามุมจากตำแหน่ง
    """
    pairs = match_lanes(g, p)
    recs = [dict(err=angle_error(g[i]["ang"], p[j]["ang"]),
                 gt_ang=g[i]["ang"], pred_ang=p[j]["ang"],
                 gt_rms=g[i]["rms"], pred_rms=p[j]["rms"])
            for i, j in pairs]
    return dict(n_gt=len(g), n_pred=len(p), n_matched=len(pairs),
                pairs=recs, gt_lanes=g, pred_lanes=p, match=pairs)


# ======================================================================
# IoU บน numpy (ของในโน้ตบุ๊กเป็น torch แบบ batch คนละ signature)
# ======================================================================

def mask_iou(a, b, eps=1e-6):
    """IoU ของ mask สองภาพ (numpy, ต่อภาพเดียว)"""
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return (inter + eps) / (union + eps)


# ======================================================================
# baseline: เดามุมจากตำแหน่งเพียงอย่างเดียว (ไม่เรียนรู้อะไร)
# ======================================================================

class AnglePositionPrior:
    """
    baseline สำหรับเทียบว่าโมเดล "เรียนรู้จริงไหม"

    เลนในภาพ dashcam มีเรขาคณิตที่เดาได้อยู่แล้ว เส้นซ้ายสุดเอียงทางหนึ่ง เส้นขวาสุดเอียงอีกทาง
    baseline นี้จึงเดามุมจากตำแหน่ง x ของเส้นเท่านั้น ไม่ดูภาพเลย
    ถ้าโมเดลทำได้ไม่ดีกว่านี้เท่าไหร่ แปลว่ามันไม่ได้เรียนรู้อะไรมากกว่าการจำตำแหน่ง

    วิธีใช้:
        prior = AnglePositionPrior().fit(masks_จาก_train_split)
        pred_lanes = prior.as_pred_lanes(gt_lanes)
    """

    def __init__(self, bins=10, width=800):
        self.bins = int(bins)
        self.width = float(width)
        self.edges = np.linspace(0.0, self.width, self.bins + 1)
        self.table = [None] * self.bins

    def _bin(self, xref):
        return int(min(self.bins - 1, max(0, int(np.digitize(xref, self.edges)) - 1)))

    def fit(self, masks, **kw):
        """สร้างตาราง ตำแหน่ง -> มุมกลาง จาก GT mask (ควรใช้ train split ไม่ใช่ test)"""
        buckets = [[] for _ in range(self.bins)]
        for m in masks:
            for d in extract_lanes(m, **kw):
                buckets[self._bin(d["xref"])].append(d["ang"])
        self.table = [float(np.median(v)) if v else None for v in buckets]
        return self

    def predict(self, xref):
        v = self.table[self._bin(xref)]
        return 90.0 if v is None else v   # ไม่มีข้อมูลในช่วงนี้ -> เดาว่าเส้นตั้งฉาก

    def as_pred_lanes(self, gt_lanes):
        """คืนเส้นที่ตำแหน่งเดิมทุกอย่าง แต่เปลี่ยน 'มุม' เป็นค่าจาก prior"""
        return [dict(d, ang=self.predict(d["xref"])) for d in gt_lanes]

    def as_random_lanes(self, gt_lanes, rng):
        """พื้นล่างสุดของสเกล: ตำแหน่งเดิมแต่สุ่มมุม"""
        return [dict(d, ang=rng.uniform(0.0, 180.0)) for d in gt_lanes]


# ======================================================================
# ฟังก์ชันบิดเบือน mask สำหรับตรวจว่า metric ไวต่ออะไร / ตาบอดอะไร
# ======================================================================

def shift_mask(m, dx, dy=0):
    """เลื่อน mask ทั้งภาพ — ทดสอบความไวต่อ 'ตำแหน่ง' โดยทิศทางของเส้นไม่เปลี่ยน"""
    m = (np.asarray(m) > 0).astype(np.uint8)
    h, w = m.shape
    out = np.zeros_like(m)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out > 0


def dilate_mask(m, k=9):
    """ขยายเส้นให้หนาขึ้น — เลียนแบบโมเดลที่ทำนายเส้นอ้วนเกิน"""
    m = (np.asarray(m) > 0).astype(np.uint8)
    return cv2.dilate(m, np.ones((k, k), np.uint8)) > 0


def erode_mask(m, k=5):
    """หรอเส้นให้ผอมลง — เลียนแบบโมเดลที่ทำนายเส้นบางเกิน"""
    m = (np.asarray(m) > 0).astype(np.uint8)
    return cv2.erode(m, np.ones((k, k), np.uint8)) > 0


def rotate_lanes(m, deg, min_area=SLOPE_MIN_AREA):
    """
    หมุน "แต่ละเลน" รอบจุด centroid ของตัวเอง

    นี่คือการทดสอบตัวตัดสินของไฟล์เปรียบเทียบ เพราะมันเปลี่ยนเฉพาะ **ทิศทาง**
    โดยตำแหน่งของเลนยังอยู่ที่เดิม (ถ้าหมุนภาพทั้งภาพจะเปลี่ยนทั้งมุมและตำแหน่งพร้อมกัน
    แยกตัวแปรไม่ได้)

    ค่าที่คาด: metric ความชันต้องรายงาน ~deg ส่วน IoU ควรตกเพียงเล็กน้อย
    ใช้ warpAffine + INTER_NEAREST เพื่อไม่ให้เกิดรูโหว่จากการปัดพิกัด
    """
    m = (np.asarray(m) > 0).astype(np.uint8)
    h, w = m.shape
    out = np.zeros_like(m)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (lab == i).astype(np.uint8)
        M = cv2.getRotationMatrix2D((float(cent[i][0]), float(cent[i][1])), float(deg), 1.0)
        out |= cv2.warpAffine(comp, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out > 0


def drop_one_lane(m, rng=None, min_area=SLOPE_MIN_AREA):
    """ลบเลนทิ้งหนึ่งเส้น — ทดสอบว่า metric จับ 'ตรวจไม่เจอ' ได้ไหม"""
    m = (np.asarray(m) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(keep) <= 1:
        return m > 0            # มีเส้นเดียว ลบแล้วไม่เหลืออะไรให้วัด
    victim = rng.choice(keep) if rng is not None else keep[0]
    return (m > 0) & (lab != victim)


# ======================================================================
# self-check:  python slope_metrics.py [dataset_root]
# ======================================================================

def _draw_line_mask(target_deg, h=448, w=800, radius=6):
    """วาดเส้นตรงที่รู้มุมจริงลงบน mask เปล่า (ใช้ตรวจว่า extract_lanes คืนมุมถูก)"""
    a = 1.0 / np.tan(np.radians(target_deg))
    m = np.zeros((h, w), np.uint8)
    ys = np.arange(int(0.15 * h), int(0.95 * h))
    xs = a * ys + (w / 2 - a * (0.55 * h))
    for y, x in zip(ys, xs):
        if 0 <= x < w:
            cv2.circle(m, (int(round(x)), int(y)), radius, 1, -1)
    return m


def _selftest_synthetic(tol_deg=1.5):
    ok = True
    print("1) มุมสังเคราะห์ที่รู้คำตอบ")
    for target in (30.0, 45.0, 90.0, 135.0, 150.0):
        lanes = extract_lanes(_draw_line_mask(target))
        if len(lanes) != 1:
            print(f"   {target:5.1f}° -> เจอ {len(lanes)} เส้น (ควรได้ 1)  ✗")
            ok = False
            continue
        d = angle_error(lanes[0]["ang"], target)
        flag = "✓" if d <= tol_deg else "✗"
        ok = ok and d <= tol_deg
        print(f"   {target:5.1f}° -> ได้ {lanes[0]['ang']:6.2f}°  (คลาด {d:.2f}°, "
              f"rms={lanes[0]['rms']:.2f}px) {flag}")

    print("\n2) การพับมุม (เส้นไม่มีทิศทาง)")
    for a1, a2, exp in ((1.0, 179.0, 2.0), (10.0, 170.0, 20.0), (0.0, 90.0, 90.0),
                        (89.0, 91.0, 2.0)):
        got = angle_error(a1, a2)
        flag = "✓" if abs(got - exp) < 1e-6 else "✗"
        ok = ok and abs(got - exp) < 1e-6
        print(f"   angle_error({a1:5.1f}, {a2:5.1f}) = {got:5.1f}° (ควรได้ {exp:.1f}°) {flag}")

    print("\n3) เคสขอบ")
    checks = [
        ("mask ว่างเปล่า", extract_lanes(np.zeros((448, 800), np.uint8)), []),
        ("จุดเล็กจิ๋ว area < min", extract_lanes(
            np.pad(np.ones((4, 4), np.uint8), ((10, 434), (10, 786)))), []),
        ("match_lanes ฝั่งว่าง", match_lanes([], [dict(xref=1.0)]), []),
        ("match_lanes ไกลเกิน gate", match_lanes([dict(xref=0.0)], [dict(xref=500.0)]), []),
    ]
    for name, got, exp in checks:
        flag = "✓" if got == exp else "✗"
        ok = ok and got == exp
        print(f"   {name:26s} -> {got} (ควรได้ {exp}) {flag}")

    g = [dict(xref=100.0, ang=80.0, rms=1.0), dict(xref=600.0, ang=100.0, rms=1.0)]
    p = [dict(xref=105.0, ang=82.0, rms=1.0), dict(xref=610.0, ang=99.0, rms=1.0),
         dict(xref=400.0, ang=90.0, rms=1.0)]
    mm = match_lanes(g, p)
    flag = "✓" if len(mm) == 2 else "✗"
    ok = ok and len(mm) == 2
    print(f"   GT 2 เส้น vs PRED 3 เส้น    -> จับคู่ได้ {len(mm)} คู่ (ควรได้ 2) {flag}")
    return ok


def _selftest_real(root, n=25, rot_deg=5.0):
    """null test + ตรวจฟังก์ชันบิดเบือน บน mask จริง"""
    import csv
    import os
    test_csv = os.path.join(root, "test_list.csv")
    mask_dir = os.path.join(root, "masks")
    if not os.path.exists(test_csv):
        print(f"\n(ข้าม null test บนข้อมูลจริง: ไม่พบ {test_csv})")
        return True

    rows = list(csv.DictReader(open(test_csv)))
    step = max(1, len(rows) // n)
    picks = rows[::step][:n]

    null_err, rot_err, shift_err = [], [], []
    null_iou, rot_iou, shift_iou = [], [], []
    n_gt = n_null_match = 0
    for r in picks:
        p = os.path.join(mask_dir, r["filename"].replace(".jpg", ".png"))
        gt = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = gt > 127
        g = extract_lanes(gt)
        n_gt += len(g)

        res = slope_metrics_for_pair(gt, gt)
        n_null_match += res["n_matched"]
        null_err += [d["err"] for d in res["pairs"]]
        null_iou.append(mask_iou(gt, gt))

        rot = rotate_lanes(gt, rot_deg)
        rot_err += [d["err"] for d in slope_metrics_for_pair(gt, rot)["pairs"]]
        rot_iou.append(mask_iou(gt, rot))

        sh = shift_mask(gt, 10)
        shift_err += [d["err"] for d in slope_metrics_for_pair(gt, sh)["pairs"]]
        shift_iou.append(mask_iou(gt, sh))

    ok = True
    print(f"\n4) null test บน mask จริง {len(picks)} ภาพ (เส้น GT {n_gt} เส้น)")
    ne = float(np.max(null_err)) if null_err else float("nan")
    ni = float(np.mean(null_iou))
    print(f"   เฉลย vs เฉลย  -> มุมคลาดสูงสุด {ne:.4f}° | IoU {ni:.4f} | "
          f"จับคู่ได้ {n_null_match}/{n_gt}")
    good = ne < 1e-9 and abs(ni - 1.0) < 1e-6 and n_null_match == n_gt
    ok = ok and good
    print(f"   {'✓ ผ่าน (ต้องได้ 0° / IoU 1 / จับคู่ครบ)' if good else '✗ ไม่ผ่าน — metric มีบั๊ก'}")

    print(f"\n5) ตรวจฟังก์ชันบิดเบือน")
    rm = float(np.median(rot_err)) if rot_err else float("nan")
    good_rot = abs(rm - rot_deg) < 1.0
    ok = ok and good_rot
    print(f"   หมุนเลน {rot_deg:.0f}°  -> มุมคลาด median {rm:.2f}° (ควรได้ ~{rot_deg:.0f}°) "
          f"| IoU {np.mean(rot_iou):.3f}  {'✓' if good_rot else '✗'}")
    sm = float(np.median(shift_err)) if shift_err else float("nan")
    good_sh = sm < 0.5
    ok = ok and good_sh
    print(f"   เลื่อนขวา 10px -> มุมคลาด median {sm:.2f}° (ควรได้ ~0°) "
          f"| IoU {np.mean(shift_iou):.3f}  {'✓' if good_sh else '✗'}")
    return ok


if __name__ == "__main__":
    import sys
    print("=" * 64)
    print("self-check ของ slope_metrics.py")
    print("=" * 64)
    passed = _selftest_synthetic()
    passed = _selftest_real(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT) and passed
    print("\n" + ("✅ ผ่านทั้งหมด" if passed else "❌ มีข้อที่ไม่ผ่าน"))
    sys.exit(0 if passed else 1)
