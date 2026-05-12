import argparse
import numpy as np
import cv2
import os

# 导入你编写的三大核心模块
from core.protect import arnold_transform, inv_arnold_transform
from core.transform import apply_dwt, apply_idwt
from core.watermark import embed_watermark, extract_watermark

def main():
    parser = argparse.ArgumentParser(description="图片盲水印系统 V1.0")
    parser.add_argument("--action", choices=["embed", "extract"], required=True, help="操作模式: embed(嵌入) 或 extract(提取)")
    parser.add_argument("--host", type=str, required=True, help="载体大图的路径")
    parser.add_argument("--watermark", type=str, help="水印小图的路径 (仅嵌入时需要)")
    parser.add_argument("--output", type=str, default="output.png", help="输出结果的保存路径")
    
    # 系统安全与算法参数
    parser.add_argument("--key", type=int, default=10, help="Arnold 置乱密钥(迭代次数)")
    parser.add_argument("--delta", type=float, default=20.0, help="QIM 量化步长 (越大越抗压缩)")
    parser.add_argument("--wm_size", type=int, default=64, help="水印图像的尺寸 N (默认 64x64)")

    args = parser.parse_args()

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.action == "embed":
        print(">>> 启动水印嵌入流水线...")
        if not args.watermark:
            print("错误：嵌入模式必须提供 --watermark 参数")
            return
            
        # 1. 读取图像 (由于 DWT 基础版处理二维矩阵，这里统一使用灰度模式)
        host_img = cv2.imread(args.host, cv2.IMREAD_GRAYSCALE)
        wm_img = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)
        
        if host_img is None or wm_img is None:
            print("错误：无法读取载体或水印图像，请检查路径。")
            return

        # 2. 预处理水印：强制缩放为正方形并二值化 (0 和 1)
        wm_img = cv2.resize(wm_img, (args.wm_size, args.wm_size))
        _, wm_bin = cv2.threshold(wm_img, 127, 1, cv2.THRESH_BINARY)
        
        # 3. [Protect 层] Arnold 置乱
        print(f"[*] 正在进行 Arnold 置乱加密 (Key={args.key})...")
        wm_encrypted = arnold_transform(wm_bin, args.key)
        wm_bits = wm_encrypted.flatten() # 展平为一维比特流
        
        # 4. [Transform 层] DWT 分解
        print("[*] 正在对载体图像进行 DWT 分解...")
        coeffs = apply_dwt(host_img, wavelet='haar')
        LL, details = coeffs[0], coeffs[1]
        
        # 5. [Watermark 层] QIM 嵌入
        print(f"[*] 正在使用 QIM 算法将比特流嵌入 LL 子带 (Delta={args.delta})...")
        LL_stego = embed_watermark(LL, wm_bits, args.delta)
        
        # 6. [Transform 层] IDWT 重构
        print("[*] 正在执行 IDWT 重构含密图像...")
        stego_img = apply_idwt((LL_stego, details), wavelet='haar')
        
        cv2.imwrite(args.output, stego_img)
        print(f">>> 嵌入完成！含密图像已保存至: {args.output}")

    elif args.action == "extract":
        print(">>> 启动盲水印提取流水线...")
        
        # 1. 读取含密图像
        stego_img = cv2.imread(args.host, cv2.IMREAD_GRAYSCALE)
        if stego_img is None:
            print("错误：无法读取含密图像。")
            return
            
        # 2. [Transform 层] DWT 分解拿到被污染的 LL
        print("[*] 正在对含密图像进行 DWT 分解...")
        coeffs = apply_dwt(stego_img, wavelet='haar')
        LL_stego = coeffs[0]
        
        # 3. [Watermark 层] QIM 盲提取
        total_bits = args.wm_size * args.wm_size
        extracted_bits = extract_watermark(LL_stego, wm_len=total_bits, delta=args.delta)
        print(f"[*] 正在盲提取一维比特流 (Delta={args.delta}, 长度={total_bits})...")
        
        # 4. 重新塑形为二维矩阵
        wm_encrypted_extracted = extracted_bits.reshape((args.wm_size, args.wm_size))
        
        # 5. [Protect 层] Arnold 逆变换
        print(f"[*] 正在进行 Arnold 逆变换恢复视觉图像 (Key={args.key})...")
        wm_decrypted = inv_arnold_transform(wm_encrypted_extracted, args.key)
        
        # 将 0/1 矩阵恢复为 0/255 像素用于保存
        final_wm_img = wm_decrypted * 255
        
        cv2.imwrite(args.output, final_wm_img)
        print(f">>> 提取完成！提取出的水印已保存至: {args.output}")

if __name__ == "__main__":
    main()