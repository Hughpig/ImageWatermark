import argparse
import numpy as np
import cv2
import os

# 导入你编写的三大核心模块
from core.protect import arnold_transform, inv_arnold_transform
from core.transform import apply_dwt, apply_idwt
from core.watermark import embed_watermark, extract_watermark

def save_debug_img(filename, img, debug_mode):
    """辅助函数：仅在 debug 模式下保存中间图像"""
    if debug_mode:
        cv2.imwrite(filename, img)
        print(f"  [Debug] 已保存中间图像: {filename}")

def main():
    parser = argparse.ArgumentParser(description="图片盲水印系统 V2.0 (彩色 YCbCr + Debug)")
    parser.add_argument("--action", choices=["embed", "extract"], required=True, help="操作模式: embed 或 extract")
    parser.add_argument("--host", type=str, required=True, help="载体大图的路径 (彩色图)")
    parser.add_argument("--watermark", type=str, help="水印小图的路径 (仅嵌入时需要)")
    parser.add_argument("--output", type=str, default="output.png", help="输出结果的保存路径")
    
    # 系统安全与算法参数
    parser.add_argument("--key", type=int, default=10, help="Arnold 置乱密钥")
    parser.add_argument("--delta", type=float, default=20.0, help="QIM 量化步长")
    parser.add_argument("--wm_size", type=int, default=64, help="水印图像的尺寸 N (默认 64)")
    
    # Debug 开关
    parser.add_argument("--debug", action="store_true", help="开启 Debug 模式，输出中间变量并保存过程图")

    args = parser.parse_args()

    # 确保输出目录和 debug 目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if args.debug and not os.path.exists("debug_output"):
        os.makedirs("debug_output")

    if args.action == "embed":
        print(">>> 启动彩色水印嵌入流水线...")
        if not args.watermark:
            print("错误：嵌入模式必须提供 --watermark 参数")
            return
            
        # 1. 读取图像 (彩色模式)
        host_img = cv2.imread(args.host)
        wm_img = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE) # 水印选择单通道黑白
        
        if host_img is None or wm_img is None:
            print("错误：无法读取载体或水印图像。")
            return

        # 2. 预处理水印
        wm_img = cv2.resize(wm_img, (args.wm_size, args.wm_size))
        _, wm_bin = cv2.threshold(wm_img, 127, 1, cv2.THRESH_BINARY)
        save_debug_img("debug_output/01_wm_binary.png", wm_bin * 255, args.debug)
        
        # 3. [Protect 层] Arnold 置乱
        print(f"[*] 正在进行 Arnold 置乱加密 (Key={args.key})...")
        wm_encrypted = arnold_transform(wm_bin, args.key)
        wm_bits = wm_encrypted.flatten()
        save_debug_img("debug_output/02_wm_arnold.png", wm_encrypted * 255, args.debug)
        
        if args.debug:
            print(f"  [Debug] 水印尺寸: {wm_img.shape}, 展平后比特数: {len(wm_bits)}")
            print(f"  [Debug] 前 10 个水印比特: {wm_bits[:10]}")

        # 4. 色彩空间转换 (BGR -> YCbCr)
        img_yuv = cv2.cvtColor(host_img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(img_yuv)
        
        # 5. [Transform 层] 仅对 Y (亮度) 分量进行 DWT
        print("[*] 正在对载体 Y 分量进行 DWT 分解...")
        coeffs = apply_dwt(Y, wavelet='haar')
        LL, details = coeffs[0], coeffs[1]
        
        if args.debug:
            print(f"  [Debug] 原图 Y 分量尺寸: {Y.shape}")
            print(f"  [Debug] DWT LL 子带尺寸: {LL.shape}, 容量: {LL.size} 像素")
            print(f"  [Debug] 嵌入前 LL 前 5 个系数: {LL.flatten()[:5]}")
        
        # 6. [Watermark 层] QIM 嵌入
        print(f"[*] 正在将比特流嵌入 LL 子带 (Delta={args.delta})...")
        LL_stego = embed_watermark(LL, wm_bits, args.delta)
        
        if args.debug:
            print(f"  [Debug] 嵌入后 LL 前 5 个系数: {LL_stego.flatten()[:5]}")
            # 可视化 LL 频带的变化 (归一化到 0-255)
            LL_vis = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            LL_stego_vis = cv2.normalize(LL_stego, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            save_debug_img("debug_output/03_LL_original.png", LL_vis, args.debug)
            save_debug_img("debug_output/04_LL_stego.png", LL_stego_vis, args.debug)

        # 7. [Transform 层] IDWT 重构 Y 分量
        print("[*] 正在执行 IDWT 重构...")
        Y_stego = apply_idwt((LL_stego, details), wavelet='haar')
        
        # 8. 合并回彩色图像 (Y_stego + 原来的 Cr, Cb)
        stego_yuv = cv2.merge([Y_stego, Cr, Cb])
        stego_img = cv2.cvtColor(stego_yuv, cv2.COLOR_YCrCb2BGR)
        
        cv2.imwrite(args.output, stego_img)
        print(f">>> 嵌入完成！彩色含密图像已保存至: {args.output}")


    elif args.action == "extract":
        print(">>> 启动盲水印提取程序...")
        
        # 1. 读取彩色含密图像
        stego_img = cv2.imread(args.host)
        if stego_img is None:
            print("错误：无法读取含密图像。")
            return
            
        # 2. 提取 Y 分量
        img_yuv = cv2.cvtColor(stego_img, cv2.COLOR_BGR2YCrCb)
        Y_stego, _, _ = cv2.split(img_yuv)
        
        # 3. [Transform 层] DWT 分解拿到 LL
        print("[*] 正在对 Y 分量进行 DWT 分解...")
        coeffs = apply_dwt(Y_stego, wavelet='haar')
        LL_stego = coeffs[0]
        
        if args.debug:
            print(f"  [Debug] 提取到的 LL 子带尺寸: {LL_stego.shape}")
            print(f"  [Debug] 准备盲提取的 LL 前 5 个系数: {LL_stego.flatten()[:5]}")
        
        # 4. [Watermark 层] QIM 盲提取
        total_bits = args.wm_size * args.wm_size
        print(f"[*] 正在盲提取比特流 (Delta={args.delta}, 长度={total_bits})...")
        extracted_bits = extract_watermark(LL_stego, wm_len=total_bits, delta=args.delta)
        
        if args.debug:
            print(f"  [Debug] 提取出的前 10 个比特: {extracted_bits[:10]}")
        
        # 5. 重新塑形并保存中间态
        wm_encrypted_extracted = extracted_bits.reshape((args.wm_size, args.wm_size))
        save_debug_img("debug_output/05_extracted_arnold.png", wm_encrypted_extracted * 255, args.debug)
        
        # 6. [Protect 层] Arnold 逆变换
        print(f"[*] 正在进行 Arnold 逆变换恢复视觉图像 (Key={args.key})...")
        wm_decrypted = inv_arnold_transform(wm_encrypted_extracted, args.key)
        final_wm_img = wm_decrypted * 255
        
        cv2.imwrite(args.output, final_wm_img)
        print(f">>> 提取完成！提取出的水印已保存至: {args.output}")

if __name__ == "__main__":
    main()