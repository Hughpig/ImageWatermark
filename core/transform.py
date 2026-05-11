# 变换层：实现 DWT, DCT 或 FFT

import pywt
import numpy as np

def apply_dwt(image:np.ndarray,wavelet:str='haar')->tuple:
    # 使用 PyWavelets 进行离散小波变换
    # 转成float32计算
    float_image = image.astype(np.float32)
    coeffs = pywt.dwt2(float_image, wavelet)
    return coeffs

def apply_idwt(coeffs:tuple,wavelet:str='haar')->np.ndarray:
    # 使用 PyWavelets 进行离散小波逆变换
    reconstructed_image = pywt.idwt2(coeffs, wavelet)
    safe_img=np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    return safe_img

def apply_dct(image:np.ndarray)->np.ndarray:
    # 使用 NumPy 进行离散余弦变换
    return np.fft.fft2(image)

def apply_fft(image:np.ndarray)->np.ndarray:
    # 使用 NumPy 进行快速傅里叶变换
    return np.fft.fft2(image)
