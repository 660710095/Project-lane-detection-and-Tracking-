# Lane Detection and Tracking for Thai Roads 🚗💨

โครงสร้างโปรเจกต์ระบบตรวจจับและติดตามเส้นเลนถนน โดยเน้นการประยุกต์ใช้กับสภาพถนนในประเทศไทย พัฒนาด้วยเทคนิค Computer Vision และ Deep Learning (U-Net)

## 🌟 จุดเด่นของโปรเจกต์ (Project Highlights)
- **Thai Road Optimized:** ปรับจูนพารามิเตอร์เพื่อรับมือกับแสงจ้า, เส้นสีเหลือง, และเส้นเลนที่จางในไทย
- **Dual-Pipeline Approach:** 
    1. **Traditional CV:** ใช้ Color Thresholding (HLS/HSV) และ Perspective Transform เพื่อการประมวลผลที่รวดเร็ว
    2. **Deep Learning:** ใช้ U-Net Model สำหรับการทำ Lane Segmentation ที่แม่นยำในสภาวะซับซ้อน
- **Robust Tracking:** มีระบบ Lane Processor ช่วยให้เส้นเลนนิ่งและเสถียรด้วยเทคนิค Moving Average และ Sanity Check

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
Project-lane-detection-and-Tracking/
├── image_processing/       # 🛠️ ส่วนประมวลผลภาพ (Traditional CV)
│   ├── edge_detection.py   # การหาขอบภาพและกรองสี (HLS/HSV/Sobel)
│   ├── perspective.py      # การทำ Bird's-Eye View (ปรับจูนมุมกล้องไทย)
│   ├── lane_tracker.py     # Sliding Window และการค้นหาเส้นเลน
│   ├── lane_processor.py   # ระบบจัดการความเสถียรของเส้นเลน (Smoothing)
│   └── main.py             # ไฟล์หลักสำหรับรันระบบ CV
├── machine_learning/       # 🧠 ส่วนสมองกล (Deep Learning)
│   ├── kitti_ref/          # ไฟล์อ้างอิงและ Template จากชุดข้อมูล KITTI
│   ├── unet_model.py       # โครงสร้างสมองกล U-Net
│   ├── train.py            # สคริปต์สำหรับฝึกสอน AI
│   ├── test_model.py       # สคริปต์ทดสอบ AI กับภาพและวิดีโอ
│   └── unet_lane_model.pth # Weight ของโมเดลที่เทรนสำเร็จแล้ว
├── tools/                  # 🔧 เครื่องมือเสริม
│   └── extract_frames.py   # สกัดภาพจากวิดีโอไทยเพื่อทำ Dataset
├── data/                   # 📹 ที่เก็บวิดีโอและข้อมูลดิบ
├── dataset_for_ai/         # 📚 ชุดข้อมูลสำหรับเทรน AI (Images & Masks)
└── web_app/                # 🌐 ส่วนการนำเสนอผ่านเว็บ
```

---

## 🚀 เริ่มต้นใช้งาน (Getting Started)

### 1. การติดตั้ง (Installation)
```bash
pip install -r requirements.txt
```

### 2. การรันระบบตรวจจับเลน (CV Version)
```bash
python image_processing/main.py
```

### 3. การทดสอบ AI (Deep Learning Version)
```bash
python machine_learning/test_model.py
```

---

## 📈 แผนการพัฒนาถัดไป (Future Roadmap)
- [ ] สกัดเฟรมจากวิดีโอถนนไทยเพิ่มเติมเพื่อสร้าง **Thai Lane Dataset**
- [ ] ทำ Data Labeling (Masking) สำหรับรูปภาพถนนไทย
- [ ] ทำ **Transfer Learning** จากโมเดลเดิมเพื่อให้ฉลาดกับถนนไทยมากขึ้น
- [ ] พัฒนา Web Interface ให้รองรับการอัปโหลดวิดีโอเพื่อประมวลผลออนไลน์

---
**พัฒนาโดย:** Manorin  
**อาจารย์ที่ปรึกษา:** [ชื่ออาจารย์]  
*ส่วนหนึ่งของโครงการ Senior Project*
