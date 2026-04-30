import argparse
import numpy as np
import cv2
import os
from core.protect import arnold_transform, inv_arnold_transform

def main():
    parser=argparse.ArgumentParser(description="图片盲水印系统")
    # TODO: 添加命令行参数解析
    parser.add_argument("--input",type=str,required=True,help="输入原图路径")
    parser.add_argument("--key",type=int,default=1,help="密钥(迭代次数)")
    args=parser.parse_args()
    print("图片盲水印系统初始化完成！")

    if not os.path.exists(args.input):
        print(f"错误：找不到文件 {args.input}")
        return

    # img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(args.input)
    if img is None:
        print(f"错误：无法读取图片 {args.input}")
        return
    
    h, w = img.shape[:2]
    dim = min(h, w)
    img = cv2.resize(img, (dim, dim))
    print(f"图片已预处理为正方形: {dim}x{dim}")

    print(f"正在进行 Arnold 变换，迭代次数: {args.key}...")
    encoded = arnold_transform(img, args.key)

    print("正在进行逆向恢复...")
    decoded = inv_arnold_transform(encoded, args.key)
    combined_res = np.hstack((img, encoded, decoded))
    
    cv2.imshow("Test: Original | Encoded | Decoded", combined_res)
    print("测试完成！按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
