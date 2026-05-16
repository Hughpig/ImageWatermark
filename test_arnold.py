import cv2
import numpy as np
import os
from core.protect import arnold_transform, inv_arnold_transform

def test_arnold_lossless():
    # 1. 准备高画质数据：读取灰度图，不进行二值化
    wm_path = "data/original/logo.png"
    if not os.path.exists(wm_path):
        # 兼容性处理
        test_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    else:
        # 直接读取原始灰度数据，保留 0-255 的画质
        test_img = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.resize(test_img, (640, 640)) # 确保是正方形，适合 Arnold 变换
    
    print(">>> 原始高画质图像已就绪 (0-255 灰度)")

    # 2. 执行 Arnold 加密
    # 算法内部处理的是坐标，所以 0-255 的值会被完美原样搬运
    key = 1
    encrypted = arnold_transform(test_img, key)
    cv2.imwrite("debug_arnold_high_quality_encrypted.png", encrypted)
    print(f">>> 已完成高画质加密 (Key={key})，结果已保存至: debug_arnold_high_quality_encrypted.png")

    # 3. 执行 Arnold 逆变换解密
    decrypted = inv_arnold_transform(encrypted, key)
    cv2.imwrite("debug_arnold_high_quality_decrypted.png", decrypted)
    print(f">>> 已完成高画质解密 (Key={key})，结果已保存至: debug_arnold_high_quality_decrypted.png")

    # 4. 严格一致性验证
    # 只要像素值完全相同，就说明画质 100% 保留
    if np.array_equal(test_img, decrypted):
        print("\n✨ 完美！Arnold 算法对高画质图像实现了 100% 无损位置置乱。")
    else:
        max_diff = np.max(np.abs(test_img.astype(int) - decrypted.astype(int)))
        print(f"\n⚠️ 注意：发现数值偏移。最大偏差: {max_diff}")

if __name__ == "__main__":
    test_arnold_lossless()