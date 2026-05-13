import cv2

# 读取你生成的含水印图片
img = cv2.imread("data/output/stego.png")

# 设定压缩质量 (0-100)，数字越小压得越狠
# 建议测试三个档位：90 (轻微), 50 (中等), 10 (重度)
quality = 10

# 保存为压缩后的图
cv2.imwrite("data/output/stego_compressed.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
print(f"压缩完成，当前质量等级: {quality}")