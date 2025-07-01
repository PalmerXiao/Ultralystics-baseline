import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import glob

# --- 配置 ---
anti_uav_root = r"E:\PycharmProject\Datasets\CV\2nd-Anti-UAV"
yolo_root = r"E:\PycharmProject\Datasets\CV\2nd-Anti-UAV" # YOLO 数据集也存放在同一目录下

images_base_src = os.path.join(anti_uav_root, "JPEGImages")
annotations_base_src = os.path.join(anti_uav_root, "Annotations")

yolo_images_dir = os.path.join(yolo_root, "images")
yolo_labels_dir = os.path.join(yolo_root, "labels")

# 类别映射
class_mapping = {
    'DRONE': 0
}

# 数据集划分比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2 # test_ratio = 1.0 - train_ratio - val_ratio

# --- End 配置 ---

def convert_to_yolo(size, box):
    """将VOC的 (xmin, ymin, xmax, ymax) 转换为YOLO的 (x_center, y_center, width, height) 格式"""
    # 检查size是否有效
    if size[0] == 0 or size[1] == 0:
        print(f"警告: 无效的图像尺寸 {size}, 无法进行转换。")
        return None
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    # 限制值在0到1之间
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return (x, y, w, h)

def find_image_annotation_pairs(image_base, annotation_base):
    """查找所有图像和对应的标注文件"""
    pairs = []
    print("开始扫描源文件...")
    
    if not os.path.isdir(image_base):
        print(f"警告: 图像目录未找到: {image_base}")
        return pairs
    if not os.path.isdir(annotation_base):
        print(f"警告: 标注目录未找到: {annotation_base}")
        return pairs

    print(f"扫描目录: {image_base}")
    # 使用 glob 查找所有 bmp 文件
    image_files = glob.glob(os.path.join(image_base, '*.bmp'))

    for img_path in tqdm(image_files, desc="扫描图像文件"):
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(annotation_base, f"{base_filename}.xml")

        if os.path.exists(ann_path):
            pairs.append((img_path, ann_path))
        else:
            print(f"警告: 图像 {img_path} 对应的标注文件 {ann_path} 未找到，已跳过。")
    print(f"扫描完成，共找到 {len(pairs)} 个有效的图像/标注对。")
    return pairs

def process_and_move(pairs, split_name, target_img_base, target_lbl_base):
    """处理单个划分的文件并移动"""
    target_img_dir = os.path.join(target_img_base, split_name)
    target_lbl_dir = os.path.join(target_lbl_base, split_name)
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)

    moved_count = 0
    converted_count = 0

    print(f"\n--- 开始处理和移动 {split_name} 数据 ({len(pairs)}个文件) ---")
    for img_src_path, ann_src_path in tqdm(pairs, desc=f"处理 {split_name}"):
        base_filename = os.path.basename(img_src_path)
        txt_filename = os.path.splitext(base_filename)[0] + '.txt'

        img_dst_path = os.path.join(target_img_dir, base_filename)
        lbl_dst_path = os.path.join(target_lbl_dir, txt_filename)

        # 1. 移动图像文件
        try:
            shutil.copy2(img_src_path, img_dst_path) # 使用 copy2 保留元数据
            moved_count += 1
        except Exception as e:
            print(f"错误: 复制图像 {img_src_path} 到 {img_dst_path} 失败: {e}")
            # 如果复制失败，则不处理标注
            continue

        # 2. 处理XML并写入YOLO txt
        try:
            tree = ET.parse(ann_src_path)
            root = tree.getroot()

            size_elem = root.find('size')
            if size_elem is None:
                 print(f"警告: {ann_src_path} 中缺少 'size' 标签, 无法生成标注。")
                 open(lbl_dst_path, 'w').close() # 创建空文件
                 continue
            img_width_elem = size_elem.find('width')
            img_height_elem = size_elem.find('height')
            if img_width_elem is None or img_height_elem is None:
                 print(f"警告: {ann_src_path} 中 'size' 标签缺少 'width' 或 'height', 无法生成标注。")
                 open(lbl_dst_path, 'w').close()
                 continue
            try:
                img_width = int(img_width_elem.text)
                img_height = int(img_height_elem.text)
                if img_width <= 0 or img_height <= 0:
                    print(f"警告: {ann_src_path} 中图像尺寸无效 ({img_width}x{img_height}), 无法生成标注。")
                    open(lbl_dst_path, 'w').close()
                    continue
            except ValueError:
                print(f"警告: {ann_src_path} 中无法解析 'width' 或 'height' 为整数, 无法生成标注。")
                open(lbl_dst_path, 'w').close()
                continue

            yolo_lines = []
            obj = root.find('object') # 假设只有一个对象
            if obj is None:
                 print(f"警告: {ann_src_path} 中未找到 'object' 标签。生成的标注文件将为空。")
            else:
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    print(f"警告: {ann_src_path} 中的对象缺少 'name' 标签, 跳过此对象。")
                else:
                    class_name = class_name_elem.text
                    if class_name not in class_mapping:
                        print(f"警告: 在 {ann_src_path} 中发现未知类别 '{class_name}'，已跳过。")
                    else:
                        class_id = class_mapping[class_name]
                        bndbox_elem = obj.find('bndbox')
                        if bndbox_elem is None:
                            print(f"警告: {ann_src_path} 中的对象 '{class_name}' 缺少 'bndbox' 标签, 跳过此对象。")
                        else:
                            try:
                                xmin = float(bndbox_elem.find('xmin').text)
                                ymin = float(bndbox_elem.find('ymin').text)
                                xmax = float(bndbox_elem.find('xmax').text)
                                ymax = float(bndbox_elem.find('ymax').text)

                                if xmax <= xmin or ymax <= ymin:
                                    print(f"警告: {ann_src_path} 中对象 '{class_name}' 的坐标无效。跳过此对象。")
                                else:
                                    xmin = max(0.0, xmin)
                                    ymin = max(0.0, ymin)
                                    xmax = min(img_width, xmax)
                                    ymax = min(img_height, ymax)
                                    if xmax <= xmin or ymax <= ymin:
                                        print(f"警告: {ann_src_path} 中对象 '{class_name}' 的坐标在限制后无效。跳过此对象。")
                                    else:
                                        b = (xmin, xmax, ymin, ymax)
                                        bb = convert_to_yolo((img_width, img_height), b)
                                        if bb:
                                            yolo_lines.append(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

                            except (AttributeError, ValueError) as e:
                                print(f"警告: 解析 {ann_src_path} 中 '{class_name}' 的 bndbox 坐标时出错: {e}, 跳过此对象。")

            # 写入YOLO标注文件
            with open(lbl_dst_path, 'w') as f_out:
                f_out.writelines(yolo_lines)
            if yolo_lines:
                converted_count += 1
            # 即使yolo_lines为空，如果XML存在且被解析，也认为label被处理了（生成了空文件）
            elif os.path.exists(ann_src_path):
                 converted_count += 1

        except ET.ParseError:
            print(f"错误: 无法解析XML文件 {ann_src_path}, 将创建空标签文件。")
            open(lbl_dst_path, 'w').close()
            converted_count += 1 # Count as processed
        except Exception as e:
            print(f"处理 {ann_src_path} 时发生未知错误: {e}, 将创建空标签文件。")
            open(lbl_dst_path, 'w').close()
            converted_count += 1 # Count as processed

    print(f"--- {split_name} 处理完成 ---")
    print(f"已复制 {moved_count} 个图像文件。")
    print(f"已转换/处理 {converted_count} 个标注文件。")

if __name__ == "__main__":
    print("开始处理 2nd-Anti-UAV 数据集...")

    # 1. 查找所有文件对
    all_pairs = find_image_annotation_pairs(images_base_src, annotations_base_src)

    if not all_pairs:
        print("错误: 未找到任何有效的图像/标注对，程序退出。")
        exit()

    # 2. 随机打乱并划分数据集
    random.shuffle(all_pairs)
    total_files = len(all_pairs)
    train_end_idx = int(total_files * train_ratio)
    val_end_idx = train_end_idx + int(total_files * val_ratio)

    train_pairs = all_pairs[:train_end_idx]
    val_pairs = all_pairs[train_end_idx:val_end_idx]
    test_pairs = all_pairs[val_end_idx:]

    print(f"\n数据集划分结果:")
    print(f"  训练集: {len(train_pairs)} 文件")
    print(f"  验证集: {len(val_pairs)} 文件")
    print(f"  测试集: {len(test_pairs)} 文件")

    # 3. 创建目标根目录
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    print(f"\nYOLO 图像目标路径: {yolo_images_dir}")
    print(f"YOLO 标签目标路径: {yolo_labels_dir}")

    # 4. 处理和移动各个划分的文件
    process_and_move(train_pairs, 'train', yolo_images_dir, yolo_labels_dir)
    process_and_move(val_pairs, 'val', yolo_images_dir, yolo_labels_dir)
    process_and_move(test_pairs, 'test', yolo_images_dir, yolo_labels_dir)

    print("\n所有处理完成！") 