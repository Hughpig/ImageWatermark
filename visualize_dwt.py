import cv2
import numpy as np
import os
from core.transform import apply_dwt

def visualize_dwt_bands():
    # 1. 加载图像并取 B 通道（和你现在的系统对齐）
    img_path = "data/original/lena.png"
    if not os.path.exists(img_path):
        print(f"错误：找不到图像 {img_path}")
        return
        
    img = cv2.imread(img_path)
    B, G, R = cv2.split(img)
    print(f">>> 已读取图像 {img_path}，提取 B 通道进行 DWT 分解...")

    # 2. 执行 DWT 分解
    # coeffs 返回的是 (LL, (LH, HL, HH))
    coeffs = apply_dwt(B, wavelet='haar')
    LL, (LH, HL, HH) = coeffs

    # 3. 图像归一化处理（频域系数通常包含负数或极大值，直接显示是黑的）
    def normalize_band(band):
        # 将系数映射到 0-255 方便视觉观察
        return cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    LL_vis = normalize_band(LL)
    LH_vis = normalize_band(LH)
    HL_vis = normalize_band(HL)
    HH_vis = normalize_band(HH)

    # 4. 拼接成经典的四宫格布局
    # 左上: LL (近似) | 右上: LH (横向细节)
    # 左下: HL (纵向细节) | 右下: HH (对角线细节)
    top_row = np.hstack((LL_vis, LH_vis))
    bottom_row = np.hstack((HL_vis, HH_vis))
    dwt_collage = np.vstack((top_row, bottom_row))

    # 5. 保存结果
    output_path = "debug_dwt_visualization.png"
    cv2.imwrite(output_path, dwt_collage)
    cv2.imwrite("original_img.png", B ) # 同时保存原图以供对比
    
    print("-" * 30)
    print(f"✅ DWT 分解完成！")
    print(f"LL 尺寸: {LL.shape} (这是你嵌入水印的安全区)")
    print(f"LH 尺寸: {LH.shape}")
    print(f"结果已保存至: {output_path}")
    print("-" * 30)
    print("💡 视觉指南：")
    print("1. 左上角 (LL) 是原图的『缩影』，包含了大部分能量。")
    print("2. 另外三个子带主要记录边缘和噪声，值越接近 128 (中性) 说明细节越少。")

if __name__ == "__main__":
    visualize_dwt_bands()
