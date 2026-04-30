# Arnold Transform Protection

import numpy as np
import cv2

def arnold_transform(image:np.ndarray,iter_num:int=1)->np.ndarray:
    """
    Arnold 变换（正向）
    :param image: 待置乱的二维图像（需要是正方形 N*N）
    :param iter_num: 迭代次数，作为密钥
    :return: 置乱后的图像
    """
    h,w=image.shape[:2]
    if h!=w:
        raise ValueError("Arnold Transform requires a square image (Height == Width).")
    
    N=h
    out_img=np.zeros_like(image)
    temp_img = image.copy()
    
    for it in range(iter_num):
        for x in range(N):
            for y in range(N):
                # Arnold 变换公式（取a=b=1）
                new_x=(x+y)%N
                new_y=(x+2*y)%N
                out_img[new_x,new_y]=temp_img[x,y]
        temp_img = out_img.copy()

    return out_img

def inv_arnold_transform(image:np.ndarray,iter_num:int=1)->np.ndarray:
    """
    Arnold 变换（逆向）
    param image: 待恢复的图片
    param iter_num: 迭代次数（必须与加密时使用的次数一致）
    return: 恢复后的图像
    """
    h,w=image.shape[:2]
    if h!=w:
        raise ValueError("Arnold Transform requires a square image (Height == Width).")
    
    N=h
    out_img=np.zeros_like(image)
    temp_img = image.copy()
    
    for it in range(iter_num):
        for x in range(N):
            for y in range(N):
                # 逆 Arnold 变换公式
                new_x=(2*x-y)%N
                new_y=(-x+y)%N
                out_img[new_x,new_y]=temp_img[x, y]
        temp_img = out_img.copy()

    return out_img