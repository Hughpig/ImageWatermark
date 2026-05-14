# 变换层：实现 DWT, DCT 或 FFT

import pywt
import numpy as np
import cv2

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
    # 使用 OpenCV 进行离散余弦变换
    float_image = image.astype(np.float32)
    coeffs = cv2.dct(float_image)
    return coeffs

def apply_idct(coeffs:np.ndarray)->np.ndarray:
    # 使用 OpenCV 进行离散余弦逆变换
    reconstructed_image = cv2.idct(coeffs)
    safe_img=np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    return safe_img

def apply_fft(image:np.ndarray)->np.ndarray:
    # 对图像进行进行快速傅里叶变换
    coeffs = np.fft.fft2(image)
    shifted = np.fft.fftshift(coeffs)
    return shifted

def apply_ifft(shifted_coeffs:np.ndarray)->np.ndarray:
    # 对图像进行进行快速傅里叶逆变换
    coeffs = np.fft.ifftshift(shifted_coeffs)
    reconstructed_image = np.fft.ifft2(coeffs)
    real_img = np.real(reconstructed_image)
    safe_img=np.clip(real_img, 0, 255).astype(np.uint8)
    return safe_img