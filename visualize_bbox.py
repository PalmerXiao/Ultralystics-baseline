import cv2
import numpy as np

# 读取图像
image_path = r"E:\PycharmProject\Datasets\CV\AI-TOD\train\images\000d42241.png"
label_path = r"E:\PycharmProject\Datasets\CV\AI-TOD\train\labels\000d42241.txt"

# 读取图像
image = cv2.imread(image_path)
if image is None:
    print("无法读取图像文件")
    exit()

# 读取标注文件
with open(label_path, 'r') as f:
    lines = f.readlines()

# 为不同的目标使用不同的颜色
colors = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 0, 255),    # 红色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 粉色
]

# 处理每一行标注
for i, line in enumerate(lines):
    # 解析坐标和类别
    coords = line.strip().split()
    if len(coords) >= 5:  # 确保有足够的数据
        x1, y1, x2, y2 = map(float, coords[:4])
        class_name = coords[4]
        
        # 选择颜色（循环使用颜色列表）
        color = colors[i % len(colors)]
        
        # 在图像上绘制边界框
        thickness = 2
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # 添加类别标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, class_name, (int(x1), int(y1)-10), font, 0.5, color, thickness)

# 保存标注后的图像
output_path = 'annotated_image.jpg'
cv2.imwrite(output_path, image)
print(f"标注后的图像已保存至: {output_path}")
print(f"总共标注了 {len(lines)} 个目标") 