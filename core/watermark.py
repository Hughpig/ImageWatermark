import numpy as np

# 嵌入层：实现 QIM 量化索引调制嵌入与提取

def embed_watermark(image:np.ndarray, watermark:np.ndarray,delta:float=10.0)->np.ndarray:
    # 利用QIM在LL子带嵌入一维水印
    image_copy=image.copy()
    # 展平矩阵
    flat_image=image_copy.flatten()
    wm_len=len(watermark)
    if wm_len>len(flat_image):
        raise ValueError("Watermark is too large to embed in the image.")
    
    target_coeffs=flat_image[:wm_len]
    # QIM嵌入

    even_quantized=np.round(target_coeffs/(2*delta))*2*delta
    offset=np.where(target_coeffs>=even_quantized,delta,-delta)

    # 嵌入水印
    modified_coeffs=even_quantized+offset*watermark
    flat_image[:wm_len]=modified_coeffs
    # 重塑回原图像形状
    return flat_image.reshape(image_copy.shape)

def extract_watermark(watermarked_image:np.ndarray, watermark:np.ndarray, delta:float=10.0)->np.ndarray:
    # 利用QIM在LL子带提取一维水印
    watermarked_image_copy=watermarked_image.copy()
    # 展平矩阵
    flat_watermarked_image=watermarked_image_copy.flatten()
    wm_len=len(watermark)
    if wm_len>len(flat_watermarked_image):
        raise ValueError("Watermark is too large to extract from the image.")

    target_coeffs=flat_watermarked_image[:wm_len]
    # QIM提取

    even_quantized=np.round(target_coeffs/(2*delta))*2*delta
    offset=np.where(target_coeffs>=even_quantized,delta,-delta)

    # 提取水印
    extracted_watermark=(target_coeffs-even_quantized)/offset
    return extracted_watermark
