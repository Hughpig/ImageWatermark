import argparse
import numpy as np
import cv2
from core.protect import arnold_transform, inv_arnold_transform

def main():
    parser=argparse.ArgumentParser(description="图片盲水印系统")
    # TODO: 添加命令行参数解析
    parser.add_argument("--input",type=str,required=True,help="输入原图路径")
    parser.add_argument("--key",type=int,default=1,help="密钥(迭代次数)")
    args=parser.parse_args()
    print("图片盲水印系统初始化完成！")

if __name__ == "__main__":
    main()
