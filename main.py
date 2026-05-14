import argparse
import numpy as np
import cv2
import os

# 导入三大核心模块 (增加 DCT 相关的导入)
from core.protect import arnold_transform, inv_arnold_transform
from core.transform import apply_dwt, apply_idwt, apply_dct, apply_idct, apply_fft, apply_ifft
from core.watermark import embed_watermark, extract_watermark

def save_debug_img(filename, img, debug_mode):
    """辅助函数：仅在 debug 模式下保存中间图像"""
    if debug_mode:
        cv2.imwrite(filename, img)
        print(f"  [Debug] 已保存中间图像: {filename}")

def main():
    parser = argparse.ArgumentParser(description="图片盲水印系统 (灰度图 DWT/DCT 适配多模态路由)")
    parser.add_argument("--action", choices=["embed", "extract"], required=True, help="操作模式")
    parser.add_argument("--method", choices=["dwt", "dct", "fft"], default="dwt", help="变换算法")
    parser.add_argument("--host", type=str, required=True, help="载体大图的路径")
    parser.add_argument("--watermark", type=str, help="水印小图的路径")
    parser.add_argument("--output", type=str, default="output.png", help="输出路径")
    
    # 核心参数默认值
    parser.add_argument("--key", type=int, default=12, help="Arnold 置乱密钥")
    parser.add_argument("--delta", type=float, default=25.0, help="QIM 量化步长")
    parser.add_argument("--wm_size", type=int, default=64, help="水印尺寸 N (默认 64)")
    parser.add_argument("--debug", action="store_true", help="开启 Debug 模式")

    args = parser.parse_args()

    # 确保目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if args.debug and not os.path.exists("debug_output"):
        os.makedirs("debug_output")

    if args.action == "embed":
        print(f">>> 启动稳定版 {args.method.upper()} 彩色水印嵌入 (B 通道)...")
        if not args.watermark:
            print("错误：嵌入模式必须提供 --watermark")
            return
            
        # 1. 读取图像：宿主图读取为彩色，水印图读取为灰度
        host_img = cv2.imread(args.host) # 彩色读取
        wm_img = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE) 
        
        if host_img is None or wm_img is None:
            print("错误：无法读取图像。")
            return

        # 2. 拆分通道，仅对 Blue (B) 通道进行操作
        B, G, R = cv2.split(host_img)

        # [DCT 安全检查] OpenCV DCT 要求图像尺寸必须为偶数
        if args.method == 'dct':
            h, w = B.shape
            if h % 2 != 0 or w % 2 != 0:
                B = B[:h-(h%2), :w-(w%2)]
                G = G[:h-(h%2), :w-(w%2)]
                R = R[:h-(h%2), :w-(w%2)]
                print(f"  [提示] DCT 要求偶数尺寸，图像已自动裁剪为: {B.shape}")

        # 3. 水印二值化
        wm_img = cv2.resize(wm_img, (args.wm_size, args.wm_size))
        _, wm_bin = cv2.threshold(wm_img, 127, 1, cv2.THRESH_BINARY)
        save_debug_img("debug_output/01_wm_binary.png", wm_bin * 255, args.debug)
        
        # 4. [Protect 层] Arnold
        print(f"[*] 正在进行 Arnold 加密 (Key={args.key})...")
        wm_encrypted = arnold_transform(wm_bin, args.key)
        wm_bits = wm_encrypted.flatten()
        save_debug_img("debug_output/02_wm_arnold.png", wm_encrypted * 255, args.debug)

        # 5. [Transform 层] 频域分解 (针对 B 通道)
        print(f"[*] 正在对载体 B 通道进行 {args.method.upper()} 分解...")
        if args.method == 'dwt':
            coeffs = apply_dwt(B, wavelet='haar')
        elif args.method == 'dct':
            coeffs = apply_dct(B)
        elif args.method == 'fft':
            coeffs = apply_fft(B)
        
        if args.debug:
            print(f"  [Debug] B 通道尺寸: {B.shape}")
            if args.method == 'dwt':
                print(f"  [Debug] {args.method.upper()} LL 子带尺寸: {coeffs[0].shape}")
            else:
                print(f"  [Debug] {args.method.upper()} 系数矩阵尺寸: {coeffs.shape}")
        
        # 6. [Watermark 层] QIM 嵌入 (传入完整 coeffs 和 method)
        print(f"[*] 正在将比特流嵌入频域 (Delta={args.delta})...")
        coeffs_stego = embed_watermark(coeffs, wm_bits, method=args.method, delta=args.delta)
        
        # 7. [Transform 层] 频域重构
        print(f"[*] 正在执行 I{args.method.upper()} 重构...")
        if args.method == 'dwt':
            B_stego = apply_idwt(coeffs_stego, wavelet='haar')
        elif args.method == 'dct':
            B_stego = apply_idct(coeffs_stego)
        elif args.method == 'fft':
            B_stego = apply_ifft(coeffs_stego)
        
        # 8. 合并通道回彩色图像 (B_stego + 原始 G 和 R)
        stego_img = cv2.merge([B_stego, G, R])
        
        cv2.imwrite(args.output, stego_img)
        print(f">>> 嵌入完成！彩色含密图像: {args.output}")


    elif args.action == "extract":
        print(f">>> 启动稳定版 {args.method.upper()} 彩色盲提取 (B 通道)...")
        
        # 1. 读取彩色含密图像
        stego_img = cv2.imread(args.host)
        if stego_img is None:
            print("错误：无法读取含密图像。")
            return

        # 2. 提取 B 通道
        B_stego, _, _ = cv2.split(stego_img)

        # [DCT 安全检查]
        if args.method == 'dct':
            h, w = B_stego.shape
            if h % 2 != 0 or w % 2 != 0:
                B_stego = B_stego[:h-(h%2), :w-(w%2)]
            
        # 3. [Transform 层] 频域分解 (针对 B 通道)
        print(f"[*] 正在进行 {args.method.upper()} 分解...")
        if args.method == 'dwt':
            coeffs = apply_dwt(B_stego, wavelet='haar')
        elif args.method == 'dct':
            coeffs = apply_dct(B_stego)
        elif args.method == 'fft':
            coeffs = apply_fft(B_stego)
        
        # 4. [Watermark 层] QIM 提取 (传入完整 coeffs 和 method)
        total_bits = args.wm_size * args.wm_size
        print(f"[*] 正在盲提取比特流 (Delta={args.delta}, 长度={total_bits})...")
        extracted_bits = extract_watermark(coeffs, wm_len=total_bits, method=args.method, delta=args.delta)
        
        # 4. 重新塑形
        wm_encrypted_extracted = extracted_bits.reshape((args.wm_size, args.wm_size))
        save_debug_img("debug_output/05_extracted_arnold.png", wm_encrypted_extracted * 255, args.debug)
        
        # 5. [Protect 层] 逆 Arnold
        print(f"[*] 正在逆 Arnold 恢复视觉图像 (Key={args.key})...")
        wm_decrypted = inv_arnold_transform(wm_encrypted_extracted, args.key)
        final_wm_img = wm_decrypted * 255
        _, final_wm_img = cv2.threshold(final_wm_img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(args.output, final_wm_img)
        print(f">>> 提取完成！提取的水印: {args.output}")

if __name__ == "__main__":
    main()