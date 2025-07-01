import os
import shutil
import random
from tqdm import tqdm
import glob
import numpy as np

# --- 配置 ---
dataset_root = r'E:\PycharmProject\Datasets\CV\UAVSwarm-dataset'
# --- End 配置 ---

# 定义源文件夹路径
source_train = os.path.join(dataset_root, "train")
source_test = os.path.join(dataset_root, "test")

# 定义目标文件夹路径
target_images_base = os.path.join(dataset_root, "images")
target_labels_base = os.path.join(dataset_root, "labels")

# 创建目标文件夹结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_images_base, split), exist_ok=True)
    os.makedirs(os.path.join(target_labels_base, split), exist_ok=True)

print("目标文件夹结构已创建或已存在。")


# 收集所有图像和标注
def collect_data(root_dir):
    data = []

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    for folder in tqdm(subfolders, desc=f"扫描 {os.path.basename(root_dir)} 文件夹"):
        img_dir = os.path.join(folder, "img1")
        det_file = os.path.join(folder, "det", "det.txt")

        if not os.path.exists(img_dir) or not os.path.exists(det_file):
            print(f"警告: {folder} 中缺少必要的文件夹或文件")
            continue

        # 读取检测文件
        detections = {}
        try:
            with open(det_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        frame_num = int(parts[0])
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])

                        if frame_num not in detections:
                            detections[frame_num] = []
                        detections[frame_num].append((x, y, w, h))
        except Exception as e:
            print(f"读取检测文件 {det_file} 时出错: {e}")
            continue

        # 扫描图像文件
        image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        for img_path in image_files:
            img_filename = os.path.basename(img_path)
            frame_num = int(img_filename.split('.')[0])

            # 提取帧号
            if img_filename.startswith('0'):
                # 假设格式为 000001.jpg
                frame_num = int(img_filename.split('.')[0])

            # 获取这个帧的检测结果
            frame_detections = detections.get(frame_num, [])

            # 将图像路径和检测结果一起保存
            data.append({
                'image_path': img_path,
                'detections': frame_detections
            })

    return data


# 收集训练和测试数据
print("收集训练和测试数据...")
train_data = collect_data(source_train)
test_data = collect_data(source_test)
all_data = train_data + test_data

print(f"总共收集了 {len(all_data)} 个图像及其标注")

# # 打乱数据
# random.seed(42)  # 设置随机种子以确保结果可复现
# random.shuffle(all_data)

# 计算分割点
total_files = len(all_data)
train_split = int(total_files * 0.7)
val_split = int(total_files * 0.1)

# 分割数据集
train_files = all_data[:train_split]
val_files = all_data[train_split:train_split + val_split]
test_files = all_data[train_split + val_split:]

print(f"总文件数: {total_files}")
print(f"训练集: {len(train_files)} 文件 (70%)")
print(f"验证集: {len(val_files)} 文件 (10%)")
print(f"测试集: {len(test_files)} 文件 (20%)")


# 将边界框转换为YOLO格式
def convert_to_yolo(x, y, w, h, img_width, img_height):
    # YOLO格式: class_id, x_center, y_center, width, height
    # 所有值都归一化到0-1之间
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height

    # 限制在0-1范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))

    return 0, x_center, y_center, norm_width, norm_height  # 0 是 UAV 类别的编号


# 处理并复制文件
def process_and_copy_files(file_list, target_img_dir, target_lbl_dir, start_index=1):
    from PIL import Image

    current_index = start_index
    processed = 0

    for data in tqdm(file_list, desc=f"处理文件到 {os.path.basename(target_img_dir)}"):
        img_path = data['image_path']
        detections = data['detections']

        # 生成新的文件名
        new_img_name = f"{current_index:06d}.jpg"
        new_lbl_name = f"{current_index:06d}.txt"

        # 目标路径
        img_dst_path = os.path.join(target_img_dir, new_img_name)
        lbl_dst_path = os.path.join(target_lbl_dir, new_lbl_name)

        try:
            # 复制图像
            shutil.copy2(img_path, img_dst_path)

            # 获取图像尺寸用于归一化
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            # 创建YOLO格式的标注文件
            with open(lbl_dst_path, 'w') as f:
                for det in detections:
                    x, y, w, h = det
                    yolo_format = convert_to_yolo(x, y, w, h, img_width, img_height)
                    f.write(
                        f"{yolo_format[0]} {yolo_format[1]:.6f} {yolo_format[2]:.6f} {yolo_format[3]:.6f} {yolo_format[4]:.6f}\n")

            processed += 1
            current_index += 1

        except Exception as e:
            print(f"处理文件 {img_path} 时出错: {e}")

    return processed, current_index


# 处理训练集文件
print("\n--- 处理训练集文件 ---")
train_processed, next_index = process_and_copy_files(
    train_files,
    os.path.join(target_images_base, "train"),
    os.path.join(target_labels_base, "train"),
    start_index=1
)
print(f"已处理 {train_processed} 个训练文件。")

# 处理验证集文件
print("\n--- 处理验证集文件 ---")
val_processed, next_index = process_and_copy_files(
    val_files,
    os.path.join(target_images_base, "val"),
    os.path.join(target_labels_base, "val"),
    start_index=next_index
)
print(f"已处理 {val_processed} 个验证文件。")

# 处理测试集文件
print("\n--- 处理测试集文件 ---")
test_processed, next_index = process_and_copy_files(
    test_files,
    os.path.join(target_images_base, "test"),
    os.path.join(target_labels_base, "test"),
    start_index=next_index
)
print(f"已处理 {test_processed} 个测试文件。")

print("\n所有文件处理完成！")
print(f"总处理文件数: {train_processed + val_processed + test_processed}")
