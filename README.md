# 🛣️ Road Lane Detection and Tracking 
**ระบบตรวจจับและติดตามช่องจราจรจากวิดีโอ (Image Processing vs Machine Learning)**

![Python](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green.svg?logo=opencv&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Model-orange.svg)
![Web App](https://img.shields.io/badge/Web_App-Interface-purple.svg)

โปรเจคนี้เป็นการศึกษาและเปรียบเทียบประสิทธิภาพในการตรวจจับและติดตามเส้นแบ่งช่องจราจรจากวิดีโอกล้องหน้ารถ โดยใช้วิธีการประมวลผลภาพพื้นฐาน (OpenCV) เปรียบเทียบกับการใช้โมเดลการเรียนรู้ของเครื่อง (Machine Learning) พร้อมระบบเว็บแอปพลิเคชันสำหรับการแสดงผลการทำงาน

---
## Dataset
โหลด KITTI Road Dataset ที่ https://www.cvlibs.net/datasets/kitti/eval_road.php
- โหลด **base kit** และ **development kit**
- แตกไฟล์วางไว้ที่ `data/kitti/training/`


## 📂 โครงสร้างโปรเจค (Project Structure)

```text
Project-lane-detection-and-Tracking-/
│
├── data/                       ← โฟลเดอร์เก็บชุดข้อมูลต้นฉบับ (เช่น ภาพ/วิดีโอจาก KITTI หรือกล้องหน้ารถ)
│
├── image_processing/           ← [มโนรินทร์] โฟลเดอร์หลักสำหรับการประมวลผลภาพ (OpenCV)
│   ├── result/                 ← เก็บไฟล์ผลลัพธ์วิดีโอที่รันทดสอบ 
│   ├── edge_detection.py       ← Pipeline หลักสำหรับกรองสีและดึงเส้นขอบจราจร
│   ├── lane_processor.py       ← [ใหม่] ระบบจัดการสถานะและทำค่าเฉลี่ย (Moving Average) เพื่อลดการสั่นของเส้น
│   ├── lane_tracker.py         ← อัลกอริทึมสแกนหาพิกเซลเส้นถนนจากล่างขึ้นบน (Sliding Window)
│   ├── main.py                 ← ไฟล์สั่งรันโปรแกรมหลักของฝั่ง Image Processing
│   ├── perspective.py          ← บิดมุมมองภาพจากมุมมองปกติให้กลายเป็นมุมมองนก (Bird's-Eye View)
│   └── utils.py                ← ฟังก์ชันช่วยเหลือเบ็ดเตล็ด (เช่น โหลด/เซฟ รูปภาพและวิดีโอ)
│
├── machine_learning/           ← [มโนรินทร์] โฟลเดอร์สำหรับการสอน AI (Semantic Segmentation)
│   ├── evaluate.py             ← โค้ดวัดผลและเปรียบเทียบความแม่นยำของโมเดล
│   ├── model.py                ← โครงสร้างและสถาปัตยกรรมของโมเดล
│   ├── predict.py              ← โค้ดสำหรับทำนายเส้นจราจรจากวิดีโอ/ภาพ ด้วยโมเดลที่เทรนแล้ว
│   ├── train.py                ← โค้ดสำหรับดึงข้อมูลและสั่งรันฝึกสอน (Train) AI โมเดล
│   └── README.md               ← คู่มืออธิบายการใช้งานเฉพาะส่วนของ Machine Learning
│
├── web_app/                    ← [ปั๊ม] ระบบเว็บแอปพลิเคชันสำหรับแสดงผลและใช้งาน
│   ├── backend.py              ← จัดการระบบหลังบ้าน (API, การประมวลผล, การเชื่อมต่อโมเดล)
│   └── frontend.py             ← จัดการหน้าเว็บและส่วนติดต่อผู้ใช้งาน (UI)
│
└── results/                    ← เก็บผลลัพธ์รวมของโปรเจกต์ โมเดล และกราฟ (ถูก Ignore ไม่นำขึ้น GitHub)
```
#### วิธีการทำงานร่วมกัน (Git Workflow)
สำหรับผู้ร่วมพัฒนา กรุณาสร้าง Branch ของตัวเองก่อนเริ่มเขียนโค้ดเพื่อป้องกันโค้ดทับซ้อนกัน
## การติดตั้ง
```bash
pip install opencv-python numpy matplotlib
```

## วิธี Clone และเริ่มทำงาน
```bash
# 1. Clone repo
git clone https://github.com/660710095/Project-lane-detection-and-Tracking-.git
cd Project-lane-detection-and-Tracking-

# 2. สร้าง branch ของตัวเอง
git checkout -b machine-learning

# 3. ติดตั้ง library
pip install opencv-python numpy matplotlib
```

## ความรับผิดชอบ
| ส่วน | ผู้รับผิดชอบ |
|------|-------------|
| Image Processing (OpenCV) | มโนรินทร์ นันทะนิ 660710095 |
| Machine Learning | มโนรินทร์ นันทะนิ 660710095 |
| Web Application | ปั้ม |

วิธีการทำงานร่วมกัน (Git Workflow)
สำหรับผู้ร่วมพัฒนา กรุณาสร้าง Branch ของตัวเองก่อนเริ่มเขียนโค้ดเพื่อป้องกันโค้ดทับซ้อนกัน
```bash
# การสร้างและสลับไป Branch ของตัวเอง
git checkout -b image-processing  # สำหรับมโนรินทร์
git checkout -b machine-learning  # สำหรับมโนรินทร์
git checkout -b web_app  # สำหรับปั้ม

# การอัปเดตโค้ดขึ้น GitHub
git add .
git commit -m "อัปเดต: [ใส่รายละเอียดสิ่งที่ทำ]"
git push origin <ชื่อ-branch-ของตัวเอง>
```
