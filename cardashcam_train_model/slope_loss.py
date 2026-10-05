"""
slope loss ที่หา gradient ได้ — เอา "มุมของเส้นเลน" เข้าไปใน loss จริงๆ

เดิมทั้งโปรเจกต์เทรนด้วย BCE + Dice แล้วใช้ความชันเป็นแค่ตัววัดกับเกณฑ์เลือก epoch
ไฟล์นี้เติมช่องว่างนั้น

ทำไมต้องแยกไฟล์: CLAUDE.md กำหนดว่า slope_metrics.py เป็น single source of truth
และต้องไม่มี torch — ไฟล์นี้จึง import ของจากตรงนั้นมาใช้ ไม่ก๊อปสูตรมาเขียนใหม่

วิธี (แบบเดียวกับที่ DETR ทำกับ Hungarian): **จับคู่โดยไม่เอา gradient แล้วคิด loss แบบมี gradient**
  1. no_grad: threshold ที่ 0.5 แล้วเรียก slope_metrics_for_pair ของเดิม ได้คู่เส้น GT-pred
  2. มี gradient: สำหรับคู่ที่จับได้ คิด centerline แบบนุ่มจาก prob ดิบ โดยจำกัดอยู่ใน
     component ของเส้นนั้น แล้ว fit x = a*y + b ด้วย torch แล้วแปลงเป็นมุม
  3. loss = mean(|มุมต่าง|) / 10

ข้อจำกัดเชิงกลไก ที่ต้องเขียนกำกับทุกครั้งที่รายงานผล:
  gradient ไหลเฉพาะ**ในเส้นที่จับคู่ได้แล้ว** loss ตัวนี้จึงสอนให้ "เล็งตรงขึ้น"
  ไม่ได้สอนให้ "หาเส้นเจอ" — ช่วงต้นของการเทรนที่ยังไม่มีเส้นให้จับ ค่าจะเป็น 0

self-check: python slope_loss.py
"""
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch

from slope_metrics import (extract_lanes, slope_metrics_for_pair,
                           SLOPE_MIN_AREA, SLOPE_MIN_ROWS)

EPS = 1e-6
SCALE = 10.0        # หารองศาด้วยค่านี้ ให้ lambda อยู่ระดับ 1


def lane_components(binmask, min_area=SLOPE_MIN_AREA, min_rows=SLOPE_MIN_ROWS):
    """
    คืน list ของ (component mask, lane dict) เรียงแบบเดียวกับ extract_lanes (ตาม xref)

    เรียก extract_lanes กับ **แต่ละ component แยกกัน** ไม่ก๊อปสูตรมาเขียนใหม่
    ผลจึงตรงกับ extract_lanes(ทั้ง mask) เสมอ เพราะ component แยกกันอยู่แล้วโดยนิยาม
    """
    binmask = (np.asarray(binmask) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binmask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        m = (lab == i)
        ls = extract_lanes(m, min_area=min_area, min_rows=min_rows)
        if len(ls) == 1:
            out.append((m, ls[0]))
    out.sort(key=lambda t: t[1]["xref"])
    return out


def soft_angle(prob, comp_mask, xs=None):
    """
    มุมของเส้น จาก prob ที่ถ่วงด้วย component mask — หา gradient ได้

    ใช้สูตรเดียวกับ extract_lanes: centerline ต่อแถว -> lstsq x = a*y + b
    -> ang = degrees(arctan2(1, a)) % 180
    ถ้าป้อน prob ที่เป็น 0/1 ค่าที่ได้ต้องเท่ากับ extract_lanes เป๊ะ (ดู self-check)
    """
    H, W = prob.shape
    if xs is None:
        xs = torch.arange(W, dtype=prob.dtype, device=prob.device)
    w = prob * comp_mask
    rows = w.sum(1)
    valid = rows > EPS
    if int(valid.sum()) < SLOPE_MIN_ROWS:
        return None
    ys = torch.arange(H, dtype=prob.dtype, device=prob.device)[valid]
    xbar = (w * xs).sum(1)[valid] / rows[valid]
    A = torch.stack([ys, torch.ones_like(ys)], dim=1)
    sol = torch.linalg.lstsq(A, xbar.unsqueeze(1)).solution
    a = sol[0, 0]
    ang = torch.rad2deg(torch.atan2(torch.ones_like(a), a)) % 180.0
    return ang


def angle_diff(ang_a, ang_b):
    """ผลต่างมุมพับเข้าช่วง [0,90] — เหมือน slope_metrics.angle_error แต่เป็น torch"""
    d = (ang_a - ang_b).abs() % 180.0
    return torch.minimum(d, 180.0 - d)


def slope_loss(logits, targets, threshold=0.5):
    """
    logits  (B,1,H,W) ดิบจากโมเดล · targets (B,1,H,W) ค่า 0/1
    คืน (loss สเกลาร์, จำนวนคู่ที่ใช้) — ไม่มีคู่ไหนจับได้เลยจะคืน loss 0 ที่ยังต่อ graph อยู่
    """
    prob = torch.sigmoid(logits.float())[:, 0]
    tgt = targets[:, 0]
    terms = []
    for b in range(prob.shape[0]):
        p_np = (prob[b] > threshold).detach().cpu().numpy()
        t_np = (tgt[b] > 0.5).detach().cpu().numpy()
        with torch.no_grad():
            r = slope_metrics_for_pair(t_np, p_np)
        if not r["match"]:
            continue
        comps = lane_components(p_np)
        # ถ้าจำนวนไม่ตรงกับที่ slope_metrics_for_pair เห็น แปลว่า assumption พัง — ข้ามภาพนี้
        if len(comps) != len(r["pred_lanes"]):
            continue
        xs = torch.arange(prob.shape[2], dtype=prob.dtype, device=prob.device)
        for gi, pi in r["match"]:
            comp = torch.from_numpy(comps[pi][0]).to(prob.device, prob.dtype)
            ang_p = soft_angle(prob[b], comp, xs)
            if ang_p is None:
                continue
            ang_g = torch.tensor(r["gt_lanes"][gi]["ang"], dtype=prob.dtype,
                                 device=prob.device)
            terms.append(angle_diff(ang_p, ang_g))
    if not terms:
        return prob.sum() * 0.0, 0
    return torch.stack(terms).mean() / SCALE, len(terms)


def _self_check():
    """มุมจาก soft_angle บน mask ไบนารี ต้องเท่า extract_lanes และ gradient ต้องไหล"""
    rng = np.random.default_rng(0)
    H, W = 448, 800
    worst = 0.0
    n_checked = 0
    for k in range(20):
        m = np.zeros((H, W), np.uint8)
        for lane in range(2):
            x0 = 200 + lane * 300 + int(rng.integers(-40, 40))
            slope = float(rng.uniform(-0.6, 0.6))
            for y in range(150, H):
                x = int(x0 + slope * (y - 150))
                m[y, max(x - 8, 0):x + 8] = 1
        lanes = extract_lanes(m)
        comps = lane_components(m)
        assert len(comps) == len(lanes), f"component กับ lane ไม่ตรงกัน {len(comps)} vs {len(lanes)}"
        p = torch.from_numpy(m.astype(np.float32))
        for (cm, ld), ref in zip(comps, lanes):
            ang = soft_angle(p, torch.from_numpy(cm).float())
            d = abs(float(ang) - ref["ang"])
            worst = max(worst, min(d, 180 - d))
            n_checked += 1
    print(f"   มุมจาก soft_angle เทียบ extract_lanes: คลาดมากสุด {worst:.4f}° "
          f"({n_checked} เส้น)  {'✓' if worst < 0.5 else '✗'}")

    # gradient ไหลจริงไหม
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    with torch.no_grad():
        logits += torch.from_numpy(m.astype(np.float32)).mul(6).sub(3)[None, None]
    logits = logits.detach().requires_grad_(True)
    tgt = torch.from_numpy(m.astype(np.float32))[None, None]
    # ขยับเฉลยให้ต่างจากที่ทำนาย ไม่งั้น loss เป็น 0 พอดีแล้ววัด gradient ไม่ได้
    tgt_shift = torch.zeros_like(tgt)
    tgt_shift[..., 20:] = tgt[..., :-20]
    l, npair = slope_loss(logits, tgt_shift)
    l.backward()
    g = logits.grad
    print(f"   slope_loss = {float(l):.6f} จาก {npair} คู่ · "
          f"gradient ไม่เป็นศูนย์ทั้งก้อน: {'✓' if g is not None and float(g.abs().sum()) > 0 else '✗'}")
    print(f"   |grad| รวม {float(g.abs().sum()):.4f}")
    assert worst < 0.5, "สูตรมุมไม่ตรงกับ extract_lanes — ห้ามใช้จนกว่าจะแก้"
    assert npair > 0 and float(g.abs().sum()) > 0
    print("\n✅ ผ่านทั้งหมด")


if __name__ == "__main__":
    print("ตรวจ slope_loss.py")
    _self_check()
