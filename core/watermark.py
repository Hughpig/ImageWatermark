import numpy as np

# ==================== QIM 核心基础算法 ====================

def _qim_core_embed(target_coeffs: np.ndarray, wm_bits: np.ndarray, delta: float) -> np.ndarray:
    """内部函数：执行底层的 QIM 矢量化量化逻辑"""
    even_quantized = np.round(target_coeffs / (2 * delta)) * (2 * delta)
    offset = np.where(target_coeffs >= even_quantized, delta, -delta)
    return even_quantized + wm_bits * offset

def _qim_core_extract(target_coeffs: np.ndarray, delta: float) -> np.ndarray:
    """内部函数：执行底层的 QIM 盲提取判定逻辑"""
    quantized_idx = np.round(target_coeffs / delta)
    return np.mod(quantized_idx, 2).astype(np.uint8)

# ==================== 统一路由接口 ====================

def embed_watermark(coeffs, wm_bits: np.ndarray, method: str = 'dwt', delta: float = 10.0):
    """
    统一的水印嵌入路由函数
    :param coeffs: transform.py 返回的系数 (DWT 是元组，DCT/FFT 是矩阵)
    :param wm_bits: 一维水印比特流
    :param method: 'dwt', 'dct', 或 'fft'
    """
    wm_len = len(wm_bits)
    wm_size = int(np.sqrt(wm_len)) # 假设水印是正方形，算出边长
    
    if method == 'dwt':
        # DWT: 稳定的中心区域锁定，避免 1 像素偏移
        LL, details = coeffs[0], coeffs[1]
        LL_mod = LL.copy()
        
        # 稳定的中心点计算逻辑
        h, w = LL.shape
        start_r = (h - wm_size) // 2
        start_c = (w - wm_size) // 2
        
        # 抠出精确的 wm_size x wm_size 区域
        target_block = LL_mod[start_r : start_r + wm_size, start_c : start_c + wm_size]
        flat_target = target_block.flatten()
        
        # QIM 嵌入
        flat_target = _qim_core_embed(flat_target, wm_bits, delta)
        
        # 原路放回中心位置
        LL_mod[start_r : start_r + wm_size, start_c : start_c + wm_size] = flat_target.reshape(wm_size, wm_size)
        
        return (LL_mod, details)

    elif method == 'dct':
        # DCT: 避开左上角低频，从中频 (16, 16) 开始挖一块塞入
        coeffs_mod = coeffs.copy()
        start_x, start_y = 16, 16
        target_block = coeffs_mod[start_x:start_x+wm_size, start_y:start_y+wm_size]
        
        flat_target = target_block.flatten()
        flat_target = _qim_core_embed(flat_target, wm_bits, delta)
        
        coeffs_mod[start_x:start_x+wm_size, start_y:start_y+wm_size] = flat_target.reshape(wm_size, wm_size)
        return coeffs_mod

    elif method == 'fft':
        # FFT: 在幅度谱上修改，且必须严格遵循中心对称！
        coeffs_mod = coeffs.copy()
        mag = np.abs(coeffs_mod)     # 幅度谱
        phase = np.angle(coeffs_mod) # 相位谱 (保持不动)
        
        cx, cy = mag.shape[0] // 2, mag.shape[1] // 2
        offset_x, offset_y = 60, 60 # 远离中心，避开低频区域
        
        # 选中心偏右上方的中频区域
        target_mag = mag[cx-offset_x-wm_size : cx-offset_x, cy+offset_y : cy+offset_y+wm_size]
        flat_target = target_mag.flatten()
        flat_target = _qim_core_embed(flat_target, wm_bits, delta)
        new_target_mag = flat_target.reshape(wm_size, wm_size)
        
        # 更新右上角
        mag[cx-offset_x-wm_size : cx-offset_x, cy+offset_y : cy+offset_y+wm_size] = new_target_mag
        
        # 【极其关键的修复：+1 对齐】
        # FFT 的中心对称点是 (cx, cy)。坐标 (cx-a) 的绝对对称点是 (cx+a)。
        # 但在 Python 切片 [start:end] 中，end 是不包含的边界。
        # 严格对称的左下角矩阵切片必须在四个边界参数上全部 +1
        mag[cx+offset_x+1 : cx+offset_x+wm_size+1, cy-offset_y-wm_size+1 : cy-offset_y+1] = np.rot90(new_target_mag, 2)
        
        # 重组复数矩阵返回
        return mag * np.exp(1j * phase)
    else:
        raise ValueError("不支持的算法")

def extract_watermark(coeffs_stego, wm_len: int, method: str = 'dwt', delta: float = 10.0) -> np.ndarray:
    """统一的水印盲提取路由函数"""
    wm_size = int(np.sqrt(wm_len))
    
    if method == 'dwt':
        LL_stego = coeffs_stego[0]
        
        # 使用与嵌入端完全一致的起始点定位逻辑
        h, w = LL_stego.shape
        start_r = (h - wm_size) // 2
        start_c = (w - wm_size) // 2
        
        target_block = LL_stego[start_r : start_r + wm_size, start_c : start_c + wm_size]
        
        return _qim_core_extract(target_block.flatten(), delta)

    elif method == 'dct':
        start_x, start_y = 16, 16
        target_block = coeffs_stego[start_x:start_x+wm_size, start_y:start_y+wm_size]
        return _qim_core_extract(target_block.flatten(), delta)

    elif method == 'fft':
        mag = np.abs(coeffs_stego)
        cx, cy = mag.shape[0] // 2, mag.shape[1] // 2
        offset_x, offset_y = 60, 60 # 远离中心，避开低频区域
        
        # 只需要去右上角读数据即可
        target_mag = mag[cx-offset_x-wm_size : cx-offset_x, cy+offset_y : cy+offset_y+wm_size]
        return _qim_core_extract(target_mag.flatten(), delta)