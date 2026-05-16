## 多模态彩色图像盲水印系统 (DWT / DCT / FFT)

### 项目简介

本项目实现了一个稳健的彩色图像盲水印系统。系统支持 **DWT (小波变换)**、**DCT (离余弦变换)** 和 **FFT (快速傅里叶变换)** 三种频域路由，并结合 **QIM (量化索引调制)** 技术实现了无需原图参与的盲提取功能。

针对彩色图像，系统创新性地采用了 **B 通道独立嵌入方案**，完美解决了色彩空间转换带来的浮点舍入误差，确保了水印提取的 100% 准确率。

### 技术栈

* **核心算法**: DWT (Haar), DCT, FFT, QIM (Quantization Index Modulation)
* **安全保护**: Arnold Transform (Cat Map) 混沌置乱
* **图像处理**: OpenCV, NumPy, PyWavelets

### 核心特性

* **多模态路由**: 支持通过命令行参数一键切换三大频域变换算法。
* **彩色图支持**: 通过 B 通道分离技术，规避了 YCbCr 转换造成的精度损失。
* **盲提取能力**: 采用 QIM 算法，提取端仅需步长 $\Delta$ 即可完成提取。
* **安全性**: 利用 Arnold 变换对水印进行预加密，保障版权信息安全。
* **鲁棒性**: 对 JPEG 压缩、亮度调整等攻击具有良好的生存能力。

### 📦 快速开始

#### 1. 安装依赖

```bash
pip install numpy opencv-python PyWavelets

```

#### 2. 水印嵌入

```bash
# 使用 DWT 算法在彩色图片中嵌入水印
python main.py --action embed --method dwt --host host.png --watermark logo.png --output stego.png --delta 25 --key 12

```

#### 3. 水印提取

```bash
# 从含密图像中盲提取水印
python main.py --action extract --method dwt --host stego.png --output extracted.png --delta 25 --key 12

```

### 实验对比

| 变换方法 | 隐蔽性 (PSNR) | 提取准确率 | 建议 Delta ($\Delta$) | 特性 |
| --- | --- | --- | --- | --- |
| **DWT** | 40dB+ | 极高 | 20-30 | 时频局部性强，最稳健 |
| **DCT** | 38dB+ | 高 | 25-40 | 能量集中，抗压缩性好 |
| **FFT** | 35dB+ | 中 | 400,000+ | 全局变换，需处理共轭对称 |

### 技术注意

* **FFT 能量补偿**: 由于 FFT 逆变换的缩放效应，$\Delta$ 需设置在 $10^5$ 量级以对抗 `uint8` 截断误差。
* **尺寸对齐**: 系统会自动将载体图像裁剪为偶数尺寸，以保证 DCT 与 FFT 的数学对称性。
* **重影消除**: 提取端内置了二值化硬判决逻辑，有效消除了 FFT 频域修改产生的视觉残影。

### 许可证

本项目采用 MIT 许可证开源。

### 特别鸣谢

本项目作为上海交通大学“学森挑战计划”的课程大作业，在此对于上海交通大学老师的辛勤付出与培养表示由衷的感谢。