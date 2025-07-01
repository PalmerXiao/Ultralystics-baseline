import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- 配置 ---
voc_root = r"E:\PycharmProject\Datasets\CV\HIT-UAV\normal_xml" # HIT-UAV 数据集根目录
yolo_root = r"E:\PycharmProject\Datasets\CV\HIT-UAV\normal_xml" # YOLO数据集也存放在同一目录下

annotations_dir = os.path.join(voc_root, "Annotations")
images_dir = os.path.join(voc_root, "JPEGImages")
image_sets_dir = os.path.join(voc_root, "ImageSets", "Main")

yolo_images_dir = os.path.join(yolo_root, "images")
yolo_labels_dir = os.path.join(yolo_root, "labels")

# 类别映射
class_mapping = {
    'Person': 0,
    'Bicycle': 1,
    'Car': 2,
    'OtherVehicle': 3
}
# --- End 配置 ---

def convert_to_yolo(size, box):
    """将VOC的 (xmin, ymin, xmax, ymax) 转换为YOLO的 (x_center, y_center, width, height) 格式"""
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

def process_split(split_name):
    """处理单个数据集划分 (train, val, test)"""
    print(f"\n--- 开始处理 {split_name} ---")

    split_file = os.path.join(image_sets_dir, f"{split_name}.txt")
    if not os.path.exists(split_file):
        print(f"错误: {split_file} 文件未找到!")
        return

    # 创建目标文件夹
    target_img_dir = os.path.join(yolo_images_dir, split_name)
    target_lbl_dir = os.path.join(yolo_labels_dir, split_name)
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_lbl_dir, exist_ok=True)

    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    print(f"找到 {len(image_ids)} 个 {split_name} 图像.")

    copied_images = 0
    converted_labels = 0

    for image_id in tqdm(image_ids, desc=f"处理 {split_name}"):
        # 1. 复制图像文件
        src_img_path = os.path.join(images_dir, f"{image_id}.jpg")
        dst_img_path = os.path.join(target_img_dir, f"{image_id}.jpg")
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path) # 使用 copy2 保留元数据
            copied_images += 1
        else:
            print(f"警告: 图像文件未找到 {src_img_path}")
            continue # 如果图像不存在，则跳过此图像的处理

        # 2. 处理XML标注文件
        src_xml_path = os.path.join(annotations_dir, f"{image_id}.xml")
        dst_lbl_path = os.path.join(target_lbl_dir, f"{image_id}.txt")

        if not os.path.exists(src_xml_path):
            print(f"警告: 标注文件未找到 {src_xml_path}, 将创建空标签文件。")
            # 对于没有标注文件的图像，创建一个空的txt文件是常见的做法
            open(dst_lbl_path, 'w').close()
            continue

        try:
            tree = ET.parse(src_xml_path)
            root = tree.getroot()

            # 获取图像尺寸
            size_elem = root.find('size')
            if size_elem is None:
                 print(f"警告: {src_xml_path} 中缺少 'size' 标签, 跳过此文件。")
                 open(dst_lbl_path, 'w').close()
                 continue
            img_width_elem = size_elem.find('width')
            img_height_elem = size_elem.find('height')
            if img_width_elem is None or img_height_elem is None:
                 print(f"警告: {src_xml_path} 中 'size' 标签缺少 'width' 或 'height', 跳过此文件。")
                 open(dst_lbl_path, 'w').close()
                 continue

            try:
                img_width = int(img_width_elem.text)
                img_height = int(img_height_elem.text)
                if img_width <= 0 or img_height <= 0:
                    print(f"警告: {src_xml_path} 中图像尺寸无效 ({img_width}x{img_height}), 跳过此文件。")
                    open(dst_lbl_path, 'w').close()
                    continue
            except ValueError:
                print(f"警告: {src_xml_path} 中无法解析 'width' 或 'height' 为整数, 跳过此文件。")
                open(dst_lbl_path, 'w').close()
                continue

            yolo_lines = []
            for obj in root.findall('object'):
                # 获取类别名称
                class_name_elem = obj.find('name')
                if class_name_elem is None:
                    print(f"警告: {src_xml_path} 中的一个对象缺少 'name' 标签, 跳过此对象。")
                    continue
                class_name = class_name_elem.text

                if class_name not in class_mapping:
                    print(f"警告: 在 {src_xml_path} 中发现未知类别 '{class_name}'，已跳过此对象。")
                    continue
                class_id = class_mapping[class_name]

                # 获取边界框
                bndbox_elem = obj.find('bndbox')
                if bndbox_elem is None:
                    print(f"警告: {src_xml_path} 中的对象 '{class_name}' 缺少 'bndbox' 标签, 跳过此对象。")
                    continue

                try:
                    xmin = float(bndbox_elem.find('xmin').text)
                    ymin = float(bndbox_elem.find('ymin').text)
                    xmax = float(bndbox_elem.find('xmax').text)
                    ymax = float(bndbox_elem.find('ymax').text)
                except (AttributeError, ValueError) as e:
                    print(f"警告: 解析 {src_xml_path} 中 '{class_name}' 的 bndbox 坐标时出错: {e}, 跳过此对象。")
                    continue

                # 检查坐标有效性
                if xmax <= xmin or ymax <= ymin:
                   print(f"警告: {src_xml_path} 中对象 '{class_name}' 的坐标无效 (xmax={xmax}<=xmin={xmin} or ymax={ymax}<=ymin={ymin})。跳过此对象。")
                   continue

                # 将坐标限制在图像边界内
                xmin = max(0.0, xmin)
                ymin = max(0.0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)
                # 再次检查有效性，防止限制后出现xmax <= xmin等情况
                if xmax <= xmin or ymax <= ymin:
                   print(f"警告: {src_xml_path} 中对象 '{class_name}' 的坐标在限制到图像边界后无效。跳过此对象。")
                   continue

                # 转换到YOLO格式
                b = (xmin, xmax, ymin, ymax) # 顺序调整以匹配 convert_to_yolo
                bb = convert_to_yolo((img_width, img_height), b)
                yolo_lines.append(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

            # 写入YOLO标注文件
            with open(dst_lbl_path, 'w') as f_out:
                f_out.writelines(yolo_lines)
            if yolo_lines: # 只有成功写入了至少一行才算转换成功
                converted_labels += 1
            elif not os.path.exists(src_xml_path): # 如果源xml就不存在，也算"成功"处理（生成空文件）
                 converted_labels +=1 # 或者根据需要调整逻辑
            # else: # 如果XML存在但没有有效对象，不算转换成功
            #     print(f"信息: {src_xml_path} 中没有找到有效对象，生成的标签文件 {dst_lbl_path} 为空。")

        except ET.ParseError:
            print(f"错误: 无法解析XML文件 {src_xml_path}, 将创建空标签文件。")
            open(dst_lbl_path, 'w').close()
        except Exception as e:
            print(f"处理 {src_xml_path} 时发生未知错误: {e}, 将创建空标签文件。")
            open(dst_lbl_path, 'w').close()


    print(f"--- {split_name} 处理完成 ---")
    print(f"已复制 {copied_images} 个图像文件到 {target_img_dir}")
    print(f"已转换/处理 {converted_labels} 个标注文件到 {target_lbl_dir}")


if __name__ == "__main__":
    print("开始转换 HIT-UAV VOC 到 YOLO 格式...")

    # 检查源文件夹是否存在
    if not os.path.isdir(annotations_dir):
        print(f"错误: Annotations 文件夹未找到: {annotations_dir}")
        exit()
    if not os.path.isdir(images_dir):
        print(f"错误: JPEGImages 文件夹未找到: {images_dir}")
        exit()
    if not os.path.isdir(image_sets_dir):
        print(f"错误: ImageSets/Main 文件夹未找到: {image_sets_dir}")
        exit()

    # 创建YOLO根目录和子目录 (如果不存在)
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    print(f"YOLO 图像目标路径: {yolo_images_dir}")
    print(f"YOLO 标签目标路径: {yolo_labels_dir}")

    # 处理每个划分
    splits = ['train', 'val', 'test']
    for split in splits:
        process_split(split)

    print("\n所有处理完成！") 