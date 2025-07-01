import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

def xyxy2xywh_normalized(xyxy, w, h):
    """
    将归一化的 xyxy 格式边界框转换为归一化的 xywh 格式。

    Args:
        xyxy (np.ndarray): 形状为 (n, 4) 的数组，包含归一化的 [x1, y1, x2, y2] 坐标。
        w (int): 图像宽度。
        h (int): 图像高度。

    Returns:
        (np.ndarray): 形状为 (n, 4) 的数组，包含归一化的 [x_center, y_center, width, height] 坐标。
    """
    # 反归一化以计算宽度和高度
    xyxy_unnorm = xyxy * np.array([w, h, w, h])
    
    # 计算中心点和宽高（未归一化）
    x_center = (xyxy_unnorm[:, 0] + xyxy_unnorm[:, 2]) / 2
    y_center = (xyxy_unnorm[:, 1] + xyxy_unnorm[:, 3]) / 2
    width = xyxy_unnorm[:, 2] - xyxy_unnorm[:, 0]
    height = xyxy_unnorm[:, 3] - xyxy_unnorm[:, 1]
    
    # 重新归一化
    x_center /= w
    y_center /= h
    width /= w
    height /= h
    
    return np.stack([x_center, y_center, width, height], axis=1)


# --- 配置路径 ---
base_path = Path(r"E:\PycharmProject\Datasets\CV\SKU110K")
annotations_dir = base_path / "annotations"
source_images_dir = base_path / "images" # 原始图像文件夹

# 目标文件夹结构
target_images_dir = base_path / "images_yolo" # 避免与原images冲突，先用临时名字或确保原images为空
target_labels_dir = base_path / "labels"

# CSV 文件列表和对应的目标子文件夹
csv_files = {
    "annotations_train.csv": "train",
    "annotations_val.csv": "val",
    "annotations_test.csv": "test"
}

# CSV 列名
col_names = ["image", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]

# --- 创建目标文件夹 ---
print("创建目标文件夹...")
for split in csv_files.values():
    os.makedirs(target_images_dir / split, exist_ok=True)
    os.makedirs(target_labels_dir / split, exist_ok=True)
print("目标文件夹创建完毕。")

# --- 处理每个 CSV 文件 ---
for csv_file, split in csv_files.items():
    print(f"\n--- 开始处理 {csv_file} ({split}集) ---")
    csv_path = annotations_dir / csv_file
    if not csv_path.exists():
        print(f"警告: {csv_path} 不存在，跳过处理。")
        continue

    # 读取 CSV
    print(f"正在读取 {csv_path}...")
    try:
        # 明确指定第一列为字符串类型，防止pandas自动推断
        df = pd.read_csv(csv_path, names=col_names, dtype={'image': str})
        print(f"成功读取 {len(df)} 条标注。")
    except Exception as e:
        print(f"错误：无法读取 {csv_path}: {e}")
        continue

    # 按图像名分组
    grouped = df.groupby("image")
    print(f"共找到 {len(grouped)} 张独立图像。")

    # 处理每个图像
    for image_id_from_csv, group in tqdm(grouped, desc=f"处理 {split} 集图像"):

        # --- 修正①：处理图像文件名 ---
        # 检查从CSV读取的图像ID是否已包含.jpg
        if str(image_id_from_csv).lower().endswith('.jpg'):
            image_name_with_ext = str(image_id_from_csv)
            # 获取不带后缀的基础名，用于标签文件
            image_base_name = image_name_with_ext[:-4]
        else:
            # 如果不包含.jpg，则手动添加
            image_name_with_ext = f"{image_id_from_csv}.jpg"
            image_base_name = str(image_id_from_csv) # 基础名就是ID本身

        source_image_path = source_images_dir / image_name_with_ext
        target_image_path = target_images_dir / split / image_name_with_ext
        # --- 修正②：确保使用正确的基础名生成标签路径 ---
        target_label_path = target_labels_dir / split / f"{image_base_name}.txt"

        # 检查原始图像文件是否存在
        if not source_image_path.exists():
            # 使用修正后的文件名再次打印警告
            print(f"警告: 图像文件 {source_image_path} 不存在，无法处理标注和移动。")
            continue

        # 理论上同一图像的所有行具有相同的宽高，取第一行的即可
        # 添加检查确保 group 非空
        if group.empty:
            print(f"警告: 图像 {image_name_with_ext} 在CSV中有记录但没有具体的标注行，跳过。")
            continue

        img_width = group.iloc[0]["image_width"]
        img_height = group.iloc[0]["image_height"]

        # --- 修正②：标签文件写入逻辑保持不变，但现在依赖于正确的 source_image_path 检查 ---
        yolo_lines = []
        for _, row in group.iterrows():
            # 提取坐标并归一化
            x1 = row["x1"] / img_width
            y1 = row["y1"] / img_height
            x2 = row["x2"] / img_width
            y2 = row["y2"] / img_height

            # 确保坐标在 [0, 1] 范围内
            x1, y1, x2, y2 = map(lambda x: max(0.0, min(1.0, x)), [x1, y1, x2, y2])

            if x1 >= x2 or y1 >= y2:
                 print(f"警告：图像 {image_name_with_ext} 中存在无效边界框: x1={row['x1']}, y1={row['y1']}, x2={row['x2']}, y2={row['y2']}，已跳过。")
                 continue

            # 准备用于转换的数据
            # xyxy_norm = np.array([[x1, y1, x2, y2]]) # 这行其实没在后面用到，可以注释掉

            # 转换为 xywh 格式
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_width = x2 - x1
            box_height = y2 - y1

            # SKU-110K 是单类别，类别索引为 0
            class_index = 0

            # 格式化为 YOLO 字符串
            yolo_line = f"{class_index} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"
            yolo_lines.append(yolo_line)

        # 只有在图像存在且至少有一条有效标注时，才写入标签文件和移动图像
        if yolo_lines:
            # 写入 YOLO 标签文件
            try:
                with open(target_label_path, "w", encoding="utf-8") as f:
                    f.writelines(yolo_lines)
            except Exception as e:
                print(f"错误：无法写入标签文件 {target_label_path}: {e}")
                continue # 如果标签写入失败，不移动图像

            # 移动图像文件
            try:
                # 如果目标文件已存在，先删除（shutil.move在某些情况下会报错）
                if target_image_path.exists():
                    os.remove(target_image_path)
                shutil.move(str(source_image_path), str(target_image_path))
            except Exception as e:
                print(f"错误：无法移动图像 {source_image_path} 到 {target_image_path}: {e}")
                # 如果移动失败，考虑是否要删除已创建的标签文件
                if target_label_path.exists():
                    try:
                        os.remove(target_label_path)
                        print(f"已删除对应的标签文件: {target_label_path}")
                    except Exception as del_e:
                        print(f"警告：无法删除标签文件 {target_label_path}: {del_e}")
        else:
            # 如果 yolo_lines 为空（例如所有边界框都无效），也打印一个信息
             print(f"信息：图像 {image_name_with_ext} 没有有效的标注行，未生成标签文件，未移动图像。")


    print(f"--- 完成处理 {csv_file} ---")


# --- (可选) 清理空的原始图像文件夹 ---
# 注意：请确保所有图像都已成功移动，再取消下面的注释！
# if not os.listdir(source_images_dir):
#     print(f"\n原始图像文件夹 {source_images_dir} 为空，正在删除...")
#     try:
#         os.rmdir(source_images_dir)
#         print("原始图像文件夹已删除。")
#     except OSError as e:
#         print(f"错误：无法删除原始图像文件夹 {source_images_dir}: {e}")
# else:
#     print(f"\n警告：原始图像文件夹 {source_images_dir} 中仍有文件，未删除。")

# --- (重要) 重命名目标图像文件夹 ---
# 将临时的 images_yolo 重命名为 images
final_images_dir = base_path / "images"
if target_images_dir.exists():
    print(f"\n将 {target_images_dir} 重命名为 {final_images_dir}...")
    try:
        # 如果最终目标文件夹已存在且非空，需要处理冲突
        if final_images_dir.exists() and os.listdir(final_images_dir):
             print(f"警告：目标文件夹 {final_images_dir} 已存在且非空。请手动检查并合并/删除。跳过重命名。")
        elif final_images_dir.exists(): # 存在但为空
             os.rmdir(final_images_dir)
             os.rename(target_images_dir, final_images_dir)
             print("重命名成功。")
        else: # 不存在
            os.rename(target_images_dir, final_images_dir)
            print("重命名成功。")

    except Exception as e:
        print(f"错误：重命名文件夹失败: {e}。请手动将 {target_images_dir} 重命名为 {final_images_dir}。")


print("\n--- 所有处理完成！---")
print(f"YOLO格式的图像已存放在: {final_images_dir}")
print(f"YOLO格式的标签已存放在: {target_labels_dir}")