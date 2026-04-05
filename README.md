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
Project/
├── image_processing/   ← [มโนรินทร์] สคริปต์สำหรับการประมวลผลภาพ (OpenCV)
│   ├── utils.py        ← ฟังก์ชันช่วยเหลือ เช่น โหลดรูปภาพและวิดีโอ
│   ├── edge_detection.py ← Pipeline หลักสำหรับตรวจจับเส้นจราจร
│   └── main.py         ← ไฟล์รันโปรแกรมหลัก (ฝั่ง Image Processing)
│
├── machine_learning/   ← [ปั้ม] สคริปต์สำหรับโมเดล Machine Learning
│   ├── model.py        ← โครงสร้างและสถาปัตยกรรมของโมเดล
│   ├── train.py        ← โค้ดสำหรับดึงข้อมูลมาเทรนโมเดล
│   ├── predict.py      ← โค้ดสำหรับทำนายเส้นจราจรจากวิดีโอ/ภาพ
│   └── evaluate.py     ← โค้ดวัดผลและเปรียบเทียบความแม่นยำ
│
├── web_app/            ← [ปั้ม] ระบบเว็บแอปพลิเคชันสำหรับแสดงผลและใช้งาน
│
├── results/            ← เก็บผลลัพธ์วิดีโอและกราฟ (ถูก Ignore ไม่นำขึ้น GitHub)
└── data/               ← เก็บ Dataset สำหรับเทรนและทดสอบ (ถูก Ignore ไม่นำขึ้น GitHub)
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
วิธีการทำงานร่วมกัน (Git Workflow)
สำหรับผู้ร่วมพัฒนา กรุณาสร้าง Branch ของตัวเองก่อนเริ่มเขียนโค้ดเพื่อป้องกันโค้ดทับซ้อนกัน

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

วิธีการทำงานร่วมกัน (Git Workflow)
สำหรับผู้ร่วมพัฒนา กรุณาสร้าง Branch ของตัวเองก่อนเริ่มเขียนโค้ดเพื่อป้องกันโค้ดทับซ้อนกัน
```bash
# การสร้างและสลับไป Branch ของตัวเอง
git checkout -b image-processing  # สำหรับมโนรินทร์
git checkout -b machine-learning  # สำหรับปั้ม

# การอัปเดตโค้ดขึ้น GitHub
git add .
git commit -m "อัปเดต: [ใส่รายละเอียดสิ่งที่ทำ]"
git push origin <ชื่อ-branch-ของตัวเอง>
```
