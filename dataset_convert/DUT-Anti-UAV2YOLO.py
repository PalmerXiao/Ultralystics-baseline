import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import glob

# --- 配置 ---
dut_root = r"E:\PycharmProject\Datasets\CV\DUT Anti-UAV"
yolo_root = r"E:\PycharmProject\Datasets\CV\DUT Anti-UAV" # YOLO 数据集也存放在同一目录下

splits_to_process = ['train', 'val', 'test']

yolo_images_base = os.path.join(yolo_root, "images")
yolo_labels_base = os.path.join(yolo_root, "labels")

# 类别映射
class_mapping = {
    'UAV': 0
    # 如果实际XML中有其他类别（即使不关心），可能需要在这里添加它们以避免"未知类别"警告
    # 例如 'bird': 1, 'drone': 0, # 根据实际情况调整
}
# --- End 配置 ---

def convert_to_yolo(size, box):
    """将VOC的 (xmin, ymin, xmax, ymax) 转换为YOLO的 (x_center, y_center, width, height) 格式"""
    # 检查size是否有效
    if size[0] <= 0 or size[1] <= 0:
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

def process_original_split(split_name, source_root, target_img_base, target_lbl_base):
    """处理原始数据集的一个划分 (train, val, test)"""
    print(f"\n--- 开始处理 {split_name} --- ")

    source_img_dir = os.path.join(source_root, split_name, 'img')
    source_ann_dir = os.path.join(source_root, split_name, 'xml')

    target_img_dir = os.path.join(target_img_base, split_name)
    target_lbl_dir = os.path.join(target_lbl_base, split_name)

    # 检查源目录是否存在
    if not os.path.isdir(source_img_dir):
        print(f"错误: 源图像目录未找到: {source_img_dir}")
        return
    if not os.path.isdir(source_ann_dir):
        print(f"错误: 源标注目录未找到: {source_ann_dir}")
        return

    # 创建目标目录
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)

    # 查找所有图像文件 (忽略大小写)
    all_files_in_dir = glob.glob(os.path.join(source_img_dir, '*.*'))
    image_files = []
    supported_extensions = {'.jpg', '.jpeg', '.png'}
    for f_path in all_files_in_dir:
        # 获取小写扩展名
        ext = os.path.splitext(f_path)[1].lower()
        if ext in supported_extensions:
            image_files.append(f_path)

    print(f"在 {source_img_dir} 中找到 {len(image_files)} 个图像文件。")

    copied_images = 0
    converted_labels = 0

    for img_src_path in tqdm(image_files, desc=f"处理 {split_name}"):
        base_filename = os.path.basename(img_src_path)
        name_part = os.path.splitext(base_filename)[0]
        ann_src_path = os.path.join(source_ann_dir, f"{name_part}.xml")
        lbl_dst_path = os.path.join(target_lbl_dir, f"{name_part}.txt")
        img_dst_path = os.path.join(target_img_dir, base_filename)

        # 检查对应的XML文件是否存在
        if not os.path.exists(ann_src_path):
            print(f"警告: 图像 {img_src_path} 对应的标注文件 {ann_src_path} 未找到。跳过此图像。")
            continue

        # 1. 复制图像文件
        try:
            shutil.copy2(img_src_path, img_dst_path)
            copied_images += 1
        except Exception as e:
            print(f"错误: 复制图像 {img_src_path} 到 {img_dst_path} 失败: {e}")
            continue # 如果复制失败，则不处理标注

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
            except (ValueError, TypeError): # 添加TypeError以防text为None
                print(f"警告: {ann_src_path} 中无法解析 'width' 或 'height' 为整数, 无法生成标注。")
                open(lbl_dst_path, 'w').close()
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
                    print(f"警告: 在 {ann_src_path} 中发现未知或不关心的类别 '{class_name}'，已跳过此对象。")
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

                # 检查坐标有效性
                if xmax <= xmin or ymax <= ymin:
                   print(f"警告: {ann_src_path} 中对象 '{class_name}' 的坐标无效 (xmax={xmax}<=xmin={xmin} or ymax={ymax}<=ymin={ymin})。跳过此对象。")
                   continue

                # 将坐标限制在图像边界内
                xmin = max(0.0, xmin)
                ymin = max(0.0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)
                # 再次检查有效性
                if xmax <= xmin or ymax <= ymin:
                   print(f"警告: {ann_src_path} 中对象 '{class_name}' 的坐标在限制到图像边界后无效。跳过此对象。")
                   continue

                # 转换到YOLO格式
                b = (xmin, xmax, ymin, ymax)
                bb = convert_to_yolo((img_width, img_height), b)
                if bb:
                    yolo_lines.append(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

            # 写入YOLO标注文件
            with open(lbl_dst_path, 'w') as f_out:
                f_out.writelines(yolo_lines)
            # 只要XML文件存在且被处理（即使没有有效对象），就计数
            converted_labels += 1

        except ET.ParseError:
            print(f"错误: 无法解析XML文件 {ann_src_path}, 将创建空标签文件。")
            open(lbl_dst_path, 'w').close()
            converted_labels += 1 # Count as processed
        except Exception as e:
            print(f"处理 {ann_src_path} 时发生未知错误: {e}, 将创建空标签文件。")
            open(lbl_dst_path, 'w').close()
            converted_labels += 1 # Count as processed

    print(f"--- {split_name} 处理完成 ---")
    print(f"已复制 {copied_images} 个图像文件到 {target_img_dir}")
    print(f"已转换/处理 {converted_labels} 个标注文件到 {target_lbl_dir}")


if __name__ == "__main__":
    print("开始转换 DUT-Anti-UAV 数据集到 YOLO 格式...")

    # 创建YOLO根目录
    os.makedirs(yolo_images_base, exist_ok=True)
    os.makedirs(yolo_labels_base, exist_ok=True)
    print(f"YOLO 图像目标路径: {yolo_images_base}")
    print(f"YOLO 标签目标路径: {yolo_labels_base}")

    # 处理每个预定义的划分
    for split in splits_to_process:
        process_original_split(split, dut_root, yolo_images_base, yolo_labels_base)

    print("\n所有处理完成！") 