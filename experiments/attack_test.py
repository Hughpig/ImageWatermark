import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# ==================== 评价指标 ====================
def calculate_psnr(img1_path, img2_path):
    """计算图像质量 PSNR"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None: return 0.0
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_nc(wm_orig_path, wm_ext_path):
    """计算水印相似度 NC"""
    wm_orig = cv2.imread(wm_orig_path, cv2.IMREAD_GRAYSCALE)
    wm_ext = cv2.imread(wm_ext_path, cv2.IMREAD_GRAYSCALE)
    if wm_orig is None or wm_ext is None: return 0.0
    
    # 统一尺寸并二值化 (映射到 0 和 1)
    wm_ext = cv2.resize(wm_ext, (wm_orig.shape[1], wm_orig.shape[0]))
    w1 = (wm_orig > 127).astype(np.float32).flatten()
    w2 = (wm_ext > 127).astype(np.float32).flatten()
    
    # NC 公式
    num = np.sum(w1 * w2)
    den = np.sqrt(np.sum(w1**2)) * np.sqrt(np.sum(w2**2))
    return num / den if den != 0 else 0.0

# ==================== 攻击模拟器 ====================
def attack_jpeg(img_path, output_path, quality=50):
    img = cv2.imread(img_path)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

def attack_gaussian_noise(img_path, output_path, var=0.005):
    """PPT里的高斯噪声(方差0.005)"""
    img = cv2.imread(img_path)
    img_float = img.astype(np.float32) / 255.0
    sigma = math.sqrt(var)
    noise = np.random.normal(0, sigma, img_float.shape)
    noisy_img = np.clip(img_float + noise, 0, 1.0) * 255.0
    cv2.imwrite(output_path, noisy_img.astype(np.uint8))

def attack_salt_pepper(img_path, output_path, density=0.05):
    """PPT里的椒盐噪声(密度0.05)"""
    img = cv2.imread(img_path)
    noisy_img = np.copy(img)
    num_salt = np.ceil(density * img.size * 0.5)
    num_pepper = np.ceil(density * img.size * 0.5)
    
    # 添加盐噪声 (白点)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[tuple(coords)] = 255
    # 添加椒噪声 (黑点)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[tuple(coords)] = 0
    cv2.imwrite(output_path, noisy_img)

# ==================== 自动化流水线 ====================
def run_test_suite():
    # 修复 1：去掉所有的 ../，统一使用相对于根目录的路径
    host = "data/original/lena.png" 
    wm = "data/original/logo.png"         
    
    methods = {'dwt': 20, 'dct': 25, 'fft': 250000}
    
    for method, delta in methods.items():
        print(f"\n{'='*40}")
        print(f"🚀 开始测试模态: {method.upper()} (Delta={delta})")
        
        stego_path = f"data/output/stego_{method}.png"
        
        # 修复 2：将 python ../main.py 改为 python main.py
        cmd_embed = f"python main.py --action embed --method {method} --host {host} --watermark {wm} --output {stego_path} --delta {delta}"
        os.system(cmd_embed)
        
        psnr_val = calculate_psnr(host, stego_path)
        print(f"✅ 无攻击嵌入完成 | PSNR: {psnr_val:.2f} dB")
        
        # 修复 3：加入英文简写标识 (tag)，避免 Windows 命令行中文乱码
        attacks = [
            ("无攻击", lambda inp, out: cv2.imwrite(out, cv2.imread(inp)), None, "none"),
            ("JPEG压缩 (Q=50)", attack_jpeg, 50, "jpeg"),
            ("高斯噪声 (Var=0.005)", attack_gaussian_noise, 0.005, "gauss"),
            ("椒盐噪声 (密度=0.05)", attack_salt_pepper, 0.05, "sp")
        ]
        
        for atk_name, atk_func, atk_param, atk_tag in attacks:
            attacked_path = f"data/output/attacked_{method}_{atk_tag}.png"
            
            # 修复 4：使用纯英文 tag 作为文件名，彻底解决乱码
            ext_wm_path = f"data/output/ext_{method}_{atk_tag}.png"
            
            # 施加攻击
            if atk_param is not None:
                atk_func(stego_path, attacked_path, atk_param)
            else:
                atk_func(stego_path, attacked_path)
                
            # 执行盲提取
            cmd_ext = f"python main.py --action extract --method {method} --host {attacked_path} --output {ext_wm_path} --delta {delta}"
            os.system(cmd_ext + " > NUL 2>&1") # Windows 下屏蔽输出使用 > NUL 2>&1
            
            # 计算 NC
            nc_val = calculate_nc(wm, ext_wm_path)
            print(f"  [{atk_name}] 提取 NC: {nc_val:.3f}")

def generate_visual_grid():
    print("\n" + "="*40)
    print("📸 正在生成全模态提取视觉对比图...")
    
    methods = ['dwt', 'dct', 'fft']
    attacks = ['none', 'jpeg', 'gauss', 'sp']
    attack_names = ['No Attack', 'JPEG (Q=50)', 'Gaussian', 'Salt & Pepper']
    
    # 创建 3行 x 4列 的画板
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle('Multi-Modal Blind Watermarking Extraction Results', fontsize=20, fontweight='bold', y=0.95)
    
    for i, method in enumerate(methods):
        for j, atk_tag in enumerate(attacks):
            img_path = f"data/output/ext_{method}_{atk_tag}.png"
            ax = axes[i, j]
            
            # 读取灰度 Logo
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                ax.imshow(img, cmap='gray')
            else:
                ax.text(0.5, 0.5, 'Extraction\nFailed', ha='center', va='center', color='red')
            
            # 设置标题，关闭坐标轴
            ax.set_title(f"{method.upper()} - {attack_names[j]}", fontsize=14)
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # 留出大标题空间
    
    # 保存大图
    report_path = "data/output/visual_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"✅ 视觉对比图已生成，完美契合PPT: {report_path}")

if __name__ == "__main__":
    # 确保输出目录存在
    if not os.path.exists("../data/output"):
        os.makedirs("../data/output")
    run_test_suite()
    generate_visual_grid()