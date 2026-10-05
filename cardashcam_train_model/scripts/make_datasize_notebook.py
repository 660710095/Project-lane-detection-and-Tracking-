import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# การทดลอง Fine-tune ตามขนาดข้อมูล: 250 รูป, 500 รูป และ ทั้งหมด (749 รูป)

สมุดนี้ใช้ทำการ Fine-tune โมเดล U-Net (ResNet50 SSL) ด้วยภาพถนนไทย 3 ขนาดข้อมูล:
1. **ชุด 250 รูป (`data_subsets/thai_250`):** 245 ภาพจริง (4 clips)
2. **ชุด 500 รูป (`data_subsets/thai_500`):** 505 ภาพจริง (19 clips)
3. **ชุดทั้งหมด (`thai_road_lane`):** 749 ภาพจริง (36 clips)

และเปรียบเทียบผลลัพธ์กับ **Zero-shot (0 รูป)** บนชุดข้อสอบเดียวกัน (Test 191 รูป, Val 97 รูป)"""),

    nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ให้ path อ้างอิงจาก root ของ workspace เสมอ
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.append('.')

import finetune_thai as F
print(f'🚀 ใช้ Device: {F.DEVICE}')
"""),

    nbf.v4.new_markdown_cell("""## การตั้งค่า (Configuration)
- `INIT_PTH`: Checkpoint ตั้งต้นจาก CULane (`best_model_resnet50_ssl_bs16_ep30.pth`)
- `REHEARSE_FRAC`: 0.5 (ใช้ภาพ CULane 50% ต่อ batch เพื่อป้องกันการลืมถนนกลางคืน)
- `EPOCHS`: กำหนดรอบการเทรน (ตั้ง 1 สำหรับการรันสาธิต หรือ 30 สำหรับเทรนเต็มรูปแบบ)"""),

    nbf.v4.new_code_cell("""INIT_PTH = 'models/best_model_resnet50_ssl_bs16_ep30.pth'
REHEARSE_FRAC = 0.5
SEED = 1
EPOCHS = 1  # ตั้ง 1 เพื่อรันสาธิตให้เห็นผลทันที (ปรับเป็น 30 สำหรับเทรนเต็ม)
"""),

    nbf.v4.new_markdown_cell("""## 1. รัน Fine-tune: ชุด 250 รูป (245 ภาพจริง)"""),

    nbf.v4.new_code_cell("""print('=' * 70)
print('🎯 เริ่ม Fine-tune: ชุด 250 รูป (data_subsets/thai_250)')
print('=' * 70)

F.THAI_ROOT = 'data_subsets/thai_250'
res_250 = F.run(mode='finetune', init=INIT_PTH, tag='nb_ft_250',
                epochs=EPOCHS, rehearse_frac=REHEARSE_FRAC, seed=SEED)
"""),

    nbf.v4.new_markdown_cell("""## 2. รัน Fine-tune: ชุด 500 รูป (505 ภาพจริง)"""),

    nbf.v4.new_code_cell("""print('=' * 70)
print('🎯 เริ่ม Fine-tune: ชุด 500 รูป (data_subsets/thai_500)')
print('=' * 70)

F.THAI_ROOT = 'data_subsets/thai_500'
res_500 = F.run(mode='finetune', init=INIT_PTH, tag='nb_ft_500',
                epochs=EPOCHS, rehearse_frac=REHEARSE_FRAC, seed=SEED)
"""),

    nbf.v4.new_markdown_cell("""## 3. รัน Fine-tune: ชุดทั้งหมด (749 ภาพจริง)"""),

    nbf.v4.new_code_cell("""print('=' * 70)
print('🎯 เริ่ม Fine-tune: ชุดทั้งหมด 749 รูป (thai_road_lane)')
print('=' * 70)

F.THAI_ROOT = 'thai_road_lane'
res_all = F.run(mode='finetune', init=INIT_PTH, tag='nb_ft_all',
                epochs=EPOCHS, rehearse_frac=REHEARSE_FRAC, seed=SEED)
"""),

    nbf.v4.new_markdown_cell("""## 4. สรุปผลการทดลองเปรียบเทียบ Benchmark (เต็ม 30 Epochs)
เปรียบเทียบโมเดล Zero-shot vs 250 รูป vs 500 รูป vs ทั้งหมด 749 รูป บนชุดทดสอบไทย (191 ภาพ)"""),

    nbf.v4.new_code_cell("""# โหลดสรุปผลลัพธ์ที่เป็นเกณฑ์ทางการของโปรเจกต์ (30 epochs)
summary_df = pd.read_csv('results/thai/datasize_summary.csv')
f1_crit = summary_df[summary_df['criterion'] == 'f1'].copy()

zero_df = pd.read_csv('results/thai/zeroshot_unet_resnet50_ssl_bs16_ep30.csv')
zero_all = zero_df[zero_df['scope'] == 'ALL'].iloc[0]

rows = [
    {
        'โมเดล': 'Zero-shot (CULane เดิม)',
        'ภาพเทรนไทย': 0,
        'IoU รวม': round(zero_all['iou'], 3),
        'F1 รวม': round(zero_all['f1'], 3),
        'มุมคลาด (องศา)': round(zero_all['median_err'], 2),
        'Lane Recall': round(zero_all['lane_recall'], 3)
    },
    {
        'โมเดล': 'Fine-tune 250 รูป (ft_n245)',
        'ภาพเทรนไทย': 245,
        'IoU รวม': round(f1_crit[f1_crit['n_thai_train'] == 245]['thai_iou'].values[0], 3),
        'F1 รวม': round(f1_crit[f1_crit['n_thai_train'] == 245]['thai_f1'].values[0], 3),
        'มุมคลาด (องศา)': round(f1_crit[f1_crit['n_thai_train'] == 245]['thai_median_err'].values[0], 2),
        'Lane Recall': round(f1_crit[f1_crit['n_thai_train'] == 245]['thai_recall'].values[0], 3)
    },
    {
        'โมเดล': 'Fine-tune 500 รูป (ft_n505)',
        'ภาพเทรนไทย': 505,
        'IoU รวม': round(f1_crit[f1_crit['n_thai_train'] == 505]['thai_iou'].values[0], 3),
        'F1 รวม': round(f1_crit[f1_crit['n_thai_train'] == 505]['thai_f1'].values[0], 3),
        'มุมคลาด (องศา)': round(f1_crit[f1_crit['n_thai_train'] == 505]['thai_median_err'].values[0], 2),
        'Lane Recall': round(f1_crit[f1_crit['n_thai_train'] == 505]['thai_recall'].values[0], 3)
    },
    {
        'โมเดล': 'Fine-tune ทั้งหมด (ft_rh_s1)',
        'ภาพเทรนไทย': 749,
        'IoU รวม': round(f1_crit[f1_crit['n_thai_train'] == 749]['thai_iou'].values[0], 3),
        'F1 รวม': round(f1_crit[f1_crit['n_thai_train'] == 749]['thai_f1'].values[0], 3),
        'มุมคลาด (องศา)': round(f1_crit[f1_crit['n_thai_train'] == 749]['thai_median_err'].values[0], 2),
        'Lane Recall': round(f1_crit[f1_crit['n_thai_train'] == 749]['thai_recall'].values[0], 3)
    }
]

df_res = pd.DataFrame(rows)
display(df_res)
"""),

    nbf.v4.new_markdown_cell("""## 5. กราฟเปรียบเทียบแนวโน้มรายหมวด (Normal vs Curve vs Night)"""),

    nbf.v4.new_code_cell("""cat_data = {
    'สภาพถนน': ['Normal (ถนนตรงกลางวัน)', 'Curve (ทางโค้ง)', 'Night (กลางคืน)'],
    'Zero-shot (0 รูป)': [0.375, 0.183, 0.336],
    '250 รูป': [0.553, 0.147, 0.202],
    '500 รูป': [0.550, 0.152, 0.220],
    'ทั้งหมด (749 รูป)': [0.544, 0.196, 0.239]
}
cat_df = pd.DataFrame(cat_data)
display(cat_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# กราฟ 1: IoU ภาพรวม
x = [0, 245, 505, 749]
iou_all = [zero_all['iou'],
           f1_crit[f1_crit['n_thai_train'] == 245]['thai_iou'].values[0],
           f1_crit[f1_crit['n_thai_train'] == 505]['thai_iou'].values[0],
           f1_crit[f1_crit['n_thai_train'] == 749]['thai_iou'].values[0]]

ax1.plot(x, iou_all, 'o-', color='#1f77b4', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Thai Training Images', fontsize=11)
ax1.set_ylabel('IoU (Test Set)', fontsize=11)
ax1.set_title('Overall IoU vs Data Size', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# กราฟ 2: แยกรายหมวด
ax2.plot(x, [0.375, 0.553, 0.550, 0.544], 'o-', label='Normal', linewidth=2)
ax2.plot(x, [0.183, 0.147, 0.152, 0.196], 's--', label='Curve', linewidth=2)
ax2.plot(x, [0.336, 0.202, 0.220, 0.239], '^-.', label='Night', linewidth=2)
ax2.set_xlabel('Number of Thai Training Images', fontsize=11)
ax2.set_ylabel('IoU by Category', fontsize=11)
ax2.set_title('Category IoU vs Data Size', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()
""")
]

with open('notebooks/finetune_datasize.ipynb', 'w') as f:
    nbf.write(nb, f)
print('✅ Created notebooks/finetune_datasize.ipynb cleanly')
