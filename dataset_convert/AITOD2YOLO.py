# import cv2
# import numpy as np
import os
import shutil
from tqdm import tqdm

# # 读取图像
# image_path = r"E:\PycharmProject\Datasets\CV\AI-TOD\train\images\00b0fa633.png"
# label_path = r"E:\PycharmProject\Datasets\CV\AI-TOD\train\labels\00b0fa633.txt"

# # 读取图像
# image = cv2.imread(image_path)
# if image is None:
#     print("无法读取图像文件")
#     exit()

# # 读取标注文件
# with open(label_path, 'r') as f:
#     lines = f.readlines()

# # 定义绘制参数
# color = (0, 255, 0)  # 绿色
# thickness = 2
# font = cv2.FONT_HERSHEY_SIMPLEX

# # 遍历每一行标注并绘制
# for line in lines:
#     # 解析坐标和类别
#     coords = line.strip().split()
#     x1, y1, x2, y2 = map(float, coords[:4])
#     class_name = coords[4]

#     # 在图像上绘制边界框
#     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
#     # 添加类别标签
#     cv2.putText(image, class_name, (int(x1), int(y1)-10), font, 0.5, color, thickness)

# # 保存标注后的图像
# output_path = 'annotated_image.jpg'
# cv2.imwrite(output_path, image)
# print(f"标注后的图像已保存至: {output_path}")

def convert_to_yolo_format(bbox, img_width=800, img_height=800):
    # 从左上角和右下角坐标转换为YOLO格式（中心点坐标和宽高）
    x1, y1, x2, y2 = map(float, bbox[:4])
    
    # 计算中心点坐标和宽高
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # 归一化坐标
    x_center = x_center / img_width
    y_center = y_center / img_height
    width = width / img_width
    height = height / img_height
    
    return x_center, y_center, width, height

# 类别映射字典
class_mapping = {
    'airplane': 0,
    'bridge': 1,
    'storage-tank': 2,
    'ship': 3,
    'swimming-pool': 4,
    'vehicle': 5,
    'person': 6,
    'wind-mill': 7
}

# 定义源路径和目标路径（图像）
source_paths = {
    'train': r"E:\PycharmProject\Datasets\CV\AI-TOD\train\images",
    'test': r"E:\PycharmProject\Datasets\CV\AI-TOD\test\images",
    'val': r"E:\PycharmProject\Datasets\CV\AI-TOD\val\images"
}

# 定义标签的源路径和目标路径
label_source_paths = {
    'train': r"E:\PycharmProject\Datasets\CV\AI-TOD\train\labels",
    'val': r"E:\PycharmProject\Datasets\CV\AI-TOD\val\labels"
}

# 创建新的目标文件夹结构
base_path = r"E:\PycharmProject\Datasets\CV\AI-TOD"
new_images_path = os.path.join(base_path, "images")
new_labels_path = os.path.join(base_path, "labels")

# 创建主文件夹
for path in [new_images_path, new_labels_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# 处理图像文件
print("\n处理图像文件...")
for folder_name, source_path in source_paths.items():
    target_path = os.path.join(new_images_path, folder_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    if os.path.exists(source_path):
        files = os.listdir(source_path)
        print(f"\n正在移动图像从 {source_path} 到 {target_path}")
        
        for file in tqdm(files, desc=f"移动{folder_name}图像"):
            source_file = os.path.join(source_path, file)
            target_file = os.path.join(target_path, file)
            
            if os.path.exists(target_file):
                os.remove(target_file)
                
            shutil.move(source_file, target_file)
        print(f"已移动 {len(files)} 个图像文件到 {target_path}")
    else:
        print(f"源路径不存在: {source_path}")

# 处理标签文件
print("\n处理标签文件...")
for folder_name, source_path in label_source_paths.items():
    target_path = os.path.join(new_labels_path, folder_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    if os.path.exists(source_path):
        files = os.listdir(source_path)
        print(f"\n正在转换并移动标签从 {source_path} 到 {target_path}")
        
        for file in tqdm(files, desc=f"处理{folder_name}标签"):
            source_file = os.path.join(source_path, file)
            target_file = os.path.join(target_path, file)
            
            # 读取原始标注并转换格式
            with open(source_file, 'r') as f:
                lines = f.readlines()
            
            # 转换为YOLO格式
            yolo_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保有足够的数据
                    bbox = parts[:4]
                    class_name = parts[4]
                    
                    # 转换坐标
                    x_center, y_center, width, height = convert_to_yolo_format(bbox)
                    
                    # 获取类别索引
                    class_idx = class_mapping.get(class_name, -1)
                    if class_idx != -1:
                        # YOLO格式：<class> <x_center> <y_center> <width> <height>
                        yolo_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        yolo_lines.append(yolo_line)
            
            # 写入新格式的标注文件
            with open(target_file, 'w') as f:
                f.writelines(yolo_lines)
                
        print(f"已转换并移动 {len(files)} 个标签文件到 {target_path}")
    else:
        print(f"源路径不存在: {source_path}")

print("\n所有文件处理完成！")

