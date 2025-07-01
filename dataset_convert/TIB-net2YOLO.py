import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import glob

# --- 配置 ---
tib_root = r"E:\PycharmProject\Datasets\CV\TIB-Net"
yolo_root = r"E:\PycharmProject\Datasets\CV\TIB-Net" # YOLO 数据集也存放在同一目录下

source_images_dir = os.path.join(tib_root, "JPEGImages") # 注意文件夹名称
source_annotations_dir = os.path.join(tib_root, "Annotations")

yolo_images_base = os.path.join(yolo_root, "images")
yolo_labels_base = os.path.join(yolo_root, "labels")

# 类别映射
class_mapping = {
    'uav': 0
}

# 数据集划分比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2 # test_ratio = 1.0 - train_ratio - val_ratio

# --- End 配置 ---

def convert_to_yolo(size, box):
    """将VOC的 (xmin, ymin, xmax, ymax) 转换为YOLO的 (x_center, y_center, width, height) 格式"""
    img_w, img_h = size
    if img_w <= 0 or img_h <= 0:
        print(f"警告: 无效的图像尺寸 {size}, 无法进行转换。")
        return None

    xmin, ymin, xmax, ymax = box

    # 检查原始坐标有效性
    if xmax <= xmin or ymax <= ymin:
         print(f"警告: 无效的原始边界框坐标 xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}。跳过此框。")
         return None

    # 钳制坐标到图像边界
    xmin = max(0.0, float(xmin))
    ymin = max(0.0, float(ymin))
    xmax = min(float(img_w), float(xmax))
    ymax = min(float(img_h), float(ymax))

    # 重新检查钳制后的有效性
    if xmax <= xmin or ymax <= ymin:
         print(f"警告: 钳制后的边界框无效 xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}。跳过此框。")
         return None

    dw = 1. / img_w
    dh = 1. / img_h

    x_center = (xmin + xmax) / 2.0 * dw
    y_center = (ymin + ymax) / 2.0 * dh
    w = (xmax - xmin) * dw
    h = (ymax - ymin) * dh

    # 限制值在0到1之间 (钳制后理论上不需要，但保险起见)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return (x_center, y_center, w, h)

if __name__ == "__main__":
    print("开始处理 TIB-Net 数据集...")

    # 1. 检查源目录是否存在
    if not os.path.isdir(source_images_dir):
        print(f"错误: 源图像目录未找到: {source_images_dir}")
        exit()
    if not os.path.isdir(source_annotations_dir):
        print(f"错误: 源标注目录未找到: {source_annotations_dir}")
        exit()

    # 2. 查找图像和标注文件对
    print(f"扫描源图像目录 (仅查找 .jpg): {source_images_dir}")
    # 使用 glob 只查找小写的 .jpg 文件
    image_paths = glob.glob(os.path.join(source_images_dir, '*.jpg'))

    file_pairs = []
    print("验证图像与标注文件的对应关系...")
    missing_xml_count = 0
    for img_path in tqdm(image_paths, desc="验证文件对"):
        base_filename = os.path.basename(img_path)
        name_part = os.path.splitext(base_filename)[0]
        xml_path = os.path.join(source_annotations_dir, f"{name_part}.xml")

        if os.path.exists(xml_path):
            file_pairs.append((img_path, xml_path))
        else:
            # print(f"警告: 图像 {img_path} 对应的标注文件 {xml_path} 未找到。")
            missing_xml_count += 1

    print(f"验证完成: 找到 {len(file_pairs)} 个有效的图像/标注对。")
    if missing_xml_count > 0:
        print(f"发现 {missing_xml_count} 个图像文件缺少对应的 XML 标注文件。")

    if not file_pairs:
        print("错误: 未找到任何有效的图像/标注对。程序退出。")
        exit()

    # 3. 划分数据集
    print("\n开始划分数据集...")
    random.shuffle(file_pairs)
    total_files = len(file_pairs)
    train_end_idx = int(total_files * train_ratio)
    val_end_idx = train_end_idx + int(total_files * val_ratio)

    train_pairs = file_pairs[:train_end_idx]
    val_pairs = file_pairs[train_end_idx:val_end_idx]
    test_pairs = file_pairs[val_end_idx:]

    splits_data = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

    print(f"数据集划分结果:")
    print(f"  训练集: {len(train_pairs)} 文件")
    print(f"  验证集: {len(val_pairs)} 文件")
    print(f"  测试集: {len(test_pairs)} 文件")

    # 4. 创建目标目录并处理文件
    print("\n开始创建目标目录并复制/转换文件...")
    os.makedirs(yolo_images_base, exist_ok=True)
    os.makedirs(yolo_labels_base, exist_ok=True)
    print(f"YOLO 图像目标基础路径: {yolo_images_base}")
    print(f"YOLO 标签目标基础路径: {yolo_labels_base}")

    for split_name, pairs in splits_data.items():
        target_img_dir = os.path.join(yolo_images_base, split_name)
        target_lbl_dir = os.path.join(yolo_labels_base, split_name)
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        print(f"\n--- 处理 {split_name} ({len(pairs)} 文件) ---")
        copied_count = 0
        converted_count = 0
        error_count = 0

        for img_src_path, ann_src_path in tqdm(pairs, desc=f"处理 {split_name}"):
            base_filename = os.path.basename(img_src_path)
            name_part = os.path.splitext(base_filename)[0]
            img_dst_path = os.path.join(target_img_dir, base_filename)
            lbl_dst_path = os.path.join(target_lbl_dir, f"{name_part}.txt")

            # 1. 复制图像文件
            if not os.path.exists(img_src_path):
                print(f"警告: 在复制阶段源图像文件未找到: {img_src_path}。跳过。")
                error_count += 1
                continue
            try:
                shutil.copy2(img_src_path, img_dst_path)
                copied_count += 1
            except Exception as e:
                print(f"错误: 复制图像 {img_src_path} 到 {img_dst_path} 失败: {e}")
                error_count += 1
                continue # 复制失败则不生成标签

            # 2. 处理XML并写入YOLO txt
            try:
                tree = ET.parse(ann_src_path)
                root = tree.getroot()

                size_elem = root.find('size')
                if size_elem is None:
                    print(f"警告: {ann_src_path} 中缺少 'size' 标签, 无法生成标注。")
                    open(lbl_dst_path, 'w').close() # 创建空文件
                    error_count += 1
                    continue
                img_width_elem = size_elem.find('width')
                img_height_elem = size_elem.find('height')
                if img_width_elem is None or img_height_elem is None:
                    print(f"警告: {ann_src_path} 中 'size' 标签缺少 'width' 或 'height', 无法生成标注。")
                    open(lbl_dst_path, 'w').close()
                    error_count += 1
                    continue
                try:
                    img_width = int(img_width_elem.text)
                    img_height = int(img_height_elem.text)
                    if img_width <= 0 or img_height <= 0:
                        print(f"警告: {ann_src_path} 中图像尺寸无效 ({img_width}x{img_height}), 无法生成标注。")
                        open(lbl_dst_path, 'w').close()
                        error_count += 1
                        continue
                except (ValueError, TypeError):
                    print(f"警告: {ann_src_path} 中无法解析 'width' 或 'height' 为整数, 无法生成标注。")
                    open(lbl_dst_path, 'w').close()
                    error_count += 1
                    continue

                yolo_lines = []
                objects_found = root.findall('object')
                if not objects_found:
                    print(f"警告: {ann_src_path} 中未找到 'object' 标签。生成的标注文件将为空。")

                for obj in objects_found:
                    class_name_elem = obj.find('name')
                    if class_name_elem is None:
                        print(f"警告: {ann_src_path} 中的一个对象缺少 'name' 标签, 跳过此对象。")
                        continue
                    class_name = class_name_elem.text

                    if class_name not in class_mapping:
                        # print(f"警告: 在 {ann_src_path} 中发现未知类别 '{class_name}'，已跳过此对象。") # TIB-Net 应该只有 uav
                        continue
                    class_id = class_mapping[class_name]

                    bndbox_elem = obj.find('bndbox')
                    if bndbox_elem is None:
                        print(f"警告: {ann_src_path} 中的对象 '{class_name}' 缺少 'bndbox' 标签, 跳过此对象。")
                        continue

                    try:
                        xmin = float(bndbox_elem.find('xmin').text)
                        ymin = float(bndbox_elem.find('ymin').text)
                        xmax = float(bndbox_elem.find('xmax').text)
                        ymax = float(bndbox_elem.find('ymax').text)
                    except (AttributeError, ValueError, TypeError) as e:
                        print(f"警告: 解析 {ann_src_path} 中 '{class_name}' 的 bndbox 坐标时出错: {e}, 跳过此对象。")
                        continue

                    yolo_bbox = convert_to_yolo((img_width, img_height), (xmin, ymin, xmax, ymax))
                    if yolo_bbox:
                        x_c, y_c, w, h = yolo_bbox
                        yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    else:
                        # convert_to_yolo 内部会打印警告
                        pass

                # 写入YOLO标注文件
                with open(lbl_dst_path, 'w') as f_out:
                    f_out.writelines(yolo_lines)
                converted_count += 1 # 无论是否有有效对象，只要处理了XML就计数

            except ET.ParseError:
                print(f"错误: 无法解析XML文件 {ann_src_path}, 将创建空标签文件。")
                open(lbl_dst_path, 'w').close()
                converted_count += 1
                error_count += 1
            except Exception as e:
                print(f"处理 {ann_src_path} 时发生未知错误: {e}, 将创建空标签文件。")
                open(lbl_dst_path, 'w').close()
                converted_count += 1
                error_count += 1

        print(f"--- {split_name} 处理完成 ---")
        print(f"已成功复制 {copied_count} 个图像文件。")
        print(f"已处理 {converted_count} 个标注文件。")
        if error_count > 0:
             print(f"处理过程中遇到 {error_count} 个错误或警告 (详情见上方日志)。")

    print("\n所有处理完成！") 