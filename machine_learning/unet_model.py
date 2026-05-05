import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    บล็อกพื้นฐานของ U-Net: ประกอบด้วย Convolution 2 ชั้นติดกัน
    ตามด้วย Batch Normalization (เพื่อให้เทรนง่ายขึ้น) และ ReLU (Activation Function)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # Conv ชั้นที่ 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Conv ชั้นที่ 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # ================= ส่วนขาลง (Encoder) =================
        # ภาพสีมี 3 ช่อง (RGB) -> ขยายความลึกเป็น 64
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) # ย่อขนาดภาพลงครึ่งนึง
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # ================= ส่วนก้นกระทะ (Bottleneck) =================
        self.bottleneck = DoubleConv(512, 1024)

        # ================= ส่วนขาขึ้น (Decoder) =================
        # ขยายขนาดภาพกลับ (Up-convolution)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)

        # ================= ชั้นส่งออก (Output Layer) =================
        # ยุบความลึกจาก 64 กลับมาเหลือ 1 ช่อง (ภาพขาวดำ 0-1)
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # วิ่งผ่านขาลง พร้อมเก็บข้อมูลแต่ละชั้นไว้ทำ Skip Connection
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        
        # วิ่งผ่านก้นกระทะ
        x5 = self.bottleneck(self.pool4(x4))
        
        # วิ่งผ่านขาขึ้น พร้อมนำข้อมูลจากขาลงมาต่อกัน (Concatenate)
        x = self.upconv4(x5)
        x = torch.cat([x, x4], dim=1) # นี่คือ Skip Connection โยง x4 มาแปะ
        x = self.up4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        
        return self.outconv(x)

# ================= ส่วนทดสอบการทำงานของโมเดล =================
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    dummy_input = torch.randn(1, 3, 256, 512) # จำลองภาพ 1 รูป ขนาด 512x256 RGB
    output = model(dummy_input)
    
    print("สร้างสมองกล U-Net สำเร็จ!")
    print(f"ขนาดภาพขาเข้า (Input Shape): {dummy_input.shape}")
    print(f"ขนาดภาพขาออก (Output Shape): {output.shape}")
    print("ถ้า Output ออกมาเป็น (1, 1, 256, 512) แปลว่าโมเดลพร้อมใช้งานแล้ว!")