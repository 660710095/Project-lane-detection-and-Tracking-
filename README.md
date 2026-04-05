# Project-lane-detection-and-Tracking-
Road Lane Detection and Tracking using OpenCV and Python
ระบบตรวจจับและติดตามช่องจราจรจากวิดีโอ ด้วย OpenCV และ Machine Learning

## โครงสร้างโปรเจค
Project/
├── image_processing/   ← OpenCV (มโนรินทร์)
│   ├── utils.py        ← ฟังก์ชันโหลดรูป/วิดีโอ
│   ├── edge_detection.py ← pipeline ตรวจจับเส้น
│   └── main.py         ← รันโปรแกรมหลัก
├── machine_learning/   ← ML Model (ปั้ม)
│   ├── model.py        ← โครงสร้างโมเดล
│   ├── train.py        ← เทรนโมเดล
│   ├── predict.py      ← ทำนายเส้นจราจร
│   └── evaluate.py     ← วัดผลความแม่นยำ
├── web_app/            ← เว็บแอป (ปั้ม)
├── results/            ← ผลลัพธ์ (ไม่ขึ้น GitHub)
└── data/               ← dataset (ไม่ขึ้น GitHub)

## การติดตั้ง
```bash
pip install opencv-python numpy matplotlib
```

## Dataset
โหลด KITTI Road Dataset ที่ https://www.cvlibs.net/datasets/kitti/eval_road.php
- โหลด **base kit** และ **development kit**
- แตกไฟล์วางไว้ที่ `data/kitti/training/`

## โครงสร้าง Data

data/
└── kitti/
└── training/
├── image_2/      ← รูปภาพ
└── gt_image_2/   ← ground truth

## วิธี Clone และเริ่มทำงาน
```bash
# 1. Clone repo
git clone https://github.com/660710095/Project-lane-detection-and-Tracking-.git
cd Project-lane-detection-and-Tracking-

# 2. สร้าง branch ของตัวเอง (เพื่อน)
git checkout -b machine-learning

# 3. ติดตั้ง library
pip install opencv-python numpy matplotlib
```

## ความรับผิดชอบ
| ส่วน | ผู้รับผิดชอบ |
|------|-------------|
| Image Processing (OpenCV) | มโนรินทร์ นันทะนิ 660710095 |
| Machine Learning | สองคน |
| Web Application | ชื่อ ปั้ม |

copy ใส่ README.md หลักแล้ว Ctrl+S ได้เลย แล้ว push ขึ้น GitHub:
powershellgit add .
git commit -m "update README"
git push origin image-processing
git checkout main
git merge image-processing
git push origin main
git checkout image-processing

# 🛣️ Road Lane Detection and Tracking 
**ระบบตรวจจับและติดตามช่องจราจรจากวิดีโอ ด้วยเทคนิค Image Processing และ Machine Learning**

![Python](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green.svg?logo=opencv&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Model-orange.svg)

โปรเจคนี้เป็นการศึกษาและเปรียบเทียบประสิทธิภาพในการตรวจจับและติดตามเส้นแบ่งช่องจราจรจากวิดีโอกล้องหน้ารถ โดยใช้วิธีการประมวลผลภาพพื้นฐาน (OpenCV) เปรียบเทียบกับการใช้โมเดลการเรียนรู้ของเครื่อง (Machine Learning) พร้อมระบบเว็บแอปพลิเคชันสำหรับการแสดงผล

---

## 📂 โครงสร้างโปรเจค (Project Structure)

```text
Project/
├── image_processing/   ← สคริปต์สำหรับการประมวลผลภาพ (OpenCV)
│   ├── utils.py        ← ฟังก์ชันโหลดรูปภาพและวิดีโอ
│   ├── edge_detection.py ← Pipeline สำหรับตรวจจับเส้นจราจร
│   └── main.py         ← ไฟล์รันโปรแกรมหลัก (Image Processing)
├── machine_learning/   ← สคริปต์สำหรับโมเดล Machine Learning
│   ├── model.py        ← โครงสร้างและสถาปัตยกรรมโมเดล
│   ├── train.py        ← โค้ดสำหรับเทรนโมเดล
│   ├── predict.py      ← โค้ดสำหรับทำนายเส้นจราจร
│   └── evaluate.py     ← โค้ดวัดผลและเปรียบเทียบความแม่นยำ
├── web_app/            ← ระบบเว็บแอปพลิเคชันสำหรับใช้งาน
├── results/            ← เก็บผลลัพธ์วิดีโอและกราฟ (ถูก Ignore ไม่นำขึ้น GitHub)
└── data/               ← เก็บ Dataset สำหรับเทรนและทดสอบ (ถูก Ignore ไม่นำขึ้น GitHub)
