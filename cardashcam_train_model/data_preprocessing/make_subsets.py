"""
สร้าง subset ของชุดเทรนไทย สำหรับวัดเส้นโค้งปริมาณข้อมูล

ตอบคำถามให้ทีม: label เพิ่มแล้วคุ้มไหม ถ้าเส้นโค้งยังชันที่ 749 ภาพ = คุ้ม ถ้าแบนแล้ว = ไม่คุ้ม

สุ่มแบบ **group ตาม clip_id** ไม่ใช่รายเฟรม เพราะเฟรมจากคลิปเดียวกันเกือบเหมือนกัน
ถ้าตัดรายเฟรม subset เล็กจะได้เห็นฉากครบเท่าชุดเต็ม แล้วดูดีเกินจริง

เลือก clip แบบ deterministic (เรียงตามขนาดจากมากไปน้อย แล้วตามชื่อ) รันซ้ำได้ชุดเดิม
ไม่ใช้ random seed เลย

แต่ละ subset สร้างเป็นโฟลเดอร์รากของตัวเอง โดย images/ masks/ เป็น symlink ไปของจริง
val/test ใช้ชุดเดิมทั้งหมด — เปลี่ยนแค่ชุดเทรน

ใช้: python3 make_thai_subsets.py
"""
import collections
import csv
import os

THAI_ROOT = "thai_road_lane"
SUBSET_DIR = "data_subsets"      # ของที่สคริปต์นี้สร้าง แยกจากข้อมูลต้นฉบับ
TARGETS = [250, 500]          # 749 คือชุดเต็ม ใช้ผลที่รันไปแล้ว


def pick_clips(rows, target):
    """
    เลือก clip ให้ได้จำนวนเฟรมใกล้ target ที่สุด โดยรักษาสัดส่วนหมวดของชุดเต็ม

    clip เป็นหน่วยที่แบ่งไม่ได้ จำนวนจริงจึงไม่ตรง target เป๊ะ — ยอมรับได้
    สำคัญคือทุกหมวดต้องมีอย่างน้อย 1 clip ไม่งั้น subset เล็กจะไม่มีกลางคืนเลย
    """
    bycat = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        bycat[r["category"]][r["clip_id"]].append(r)

    total = len(rows)
    picked = []
    for cat in sorted(bycat):
        clips = bycat[cat]
        share = sum(len(v) for v in clips.values()) / total
        want = target * share
        # เรียงใหญ่ไปเล็ก ตัดสินเสมอด้วยชื่อ clip เพื่อให้ผลคงที่ทุกครั้ง
        order = sorted(clips, key=lambda c: (-len(clips[c]), c))
        # clip แรกต้องหยิบเสมอ (กันหมวดหายทั้งหมวด) แต่ต้องหยิบตัวที่ขนาดใกล้ want ที่สุด
        # ไม่ใช่ตัวใหญ่สุด ไม่งั้นหมวดที่มี clip เดียวใหญ่ๆ จะกินสัดส่วนเกิน
        first = min(order, key=lambda c: (abs(len(clips[c]) - want), c))
        chosen, got = [first], len(clips[first])
        for c in order:
            if c in chosen:
                continue
            n = len(clips[c])
            if got + n <= want:
                chosen.append(c)
                got += n
        # ถ้าเติมอีก clip แล้วเข้าใกล้ want กว่าเดิม ก็เติม
        for c in order:
            if c in chosen:
                continue
            n = len(clips[c])
            if abs(got + n - want) < abs(got - want):
                chosen.append(c)
                got += n
        picked += [r for c in chosen for r in clips[c]]
    return picked


def main():
    rows = list(csv.DictReader(open(f"{THAI_ROOT}/train_list.csv")))
    field = list(rows[0])
    full = collections.Counter(r["category"] for r in rows)
    print(f"ชุดเต็ม {len(rows)} ภาพ {dict(sorted(full.items()))} "
          f"{len(set(r['clip_id'] for r in rows))} clip\n")

    for target in TARGETS:
        sub = pick_clips(rows, target)
        # เรียงตามลำดับเดิมในไฟล์ เพื่อให้ dataset กับ CSV ตรงกันแบบเดียวกับชุดเต็ม
        order = {r["filename"]: i for i, r in enumerate(rows)}
        sub.sort(key=lambda r: order[r["filename"]])

        root = os.path.join(SUBSET_DIR, f"thai_{target}")
        os.makedirs(root, exist_ok=True)
        for d in ("images", "masks"):
            link = os.path.join(root, d)
            if not os.path.islink(link) and not os.path.exists(link):
                os.symlink(os.path.abspath(f"{THAI_ROOT}/{d}"), link)
        with open(f"{root}/train_list.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=field)
            w.writeheader()
            w.writerows(sub)
        for s in ("val", "test"):
            src, dst = f"{THAI_ROOT}/{s}_list.csv", f"{root}/{s}_list.csv"
            open(dst, "w").write(open(src).read())

        cnt = collections.Counter(r["category"] for r in sub)
        clips = collections.Counter()
        for r in sub:
            clips[r["category"]] = clips[r["category"]]
        nclip = {c: len(set(r["clip_id"] for r in sub if r["category"] == c))
                 for c in sorted(cnt)}
        print(f"เป้า {target} -> ได้ {len(sub)} ภาพ  ({root}/)")
        for c in sorted(full):
            n = cnt.get(c, 0)
            print(f"   {c:7s} {n:4d} ภาพ ({n/max(len(sub),1):5.1%}, ชุดเต็ม {full[c]/len(rows):5.1%})"
                  f"  {nclip.get(c, 0)} clip")
        if any(cnt.get(c, 0) == 0 for c in full):
            print("   ⚠️  มีหมวดที่หายไปทั้งหมวด — ผลของ subset นี้ตีความยาก")
        print()


if __name__ == "__main__":
    main()
