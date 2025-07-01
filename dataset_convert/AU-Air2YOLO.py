import os
import shutil
import json
from tqdm import tqdm
import random
import glob

# --- 配置 ---
au_air_root = r"E:\PycharmProject\Datasets\CV\AU-Air"
yolo_root = r"E:\PycharmProject\Datasets\CV\AU-Air" # YOLO 数据集也存放在同一目录下

source_images_dir = os.path.join(au_air_root, "images")
annotations_json_path = os.path.join(au_air_root, "annotations.json")

yolo_images_base = os.path.join(yolo_root, "images")
yolo_labels_base = os.path.join(yolo_root, "labels")

# 数据集划分比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2 # test_ratio = 1.0 - train_ratio - val_ratio

# --- End 配置 ---

def convert_bbox_to_yolo(size, box):
    """将 AU-Air 的 (top, left, height, width) 转换为YOLO的 (x_center, y_center, width, height) 格式"""
    img_w, img_h = size
    if img_w <= 0 or img_h <= 0:
        print(f"警告: 无效的图像尺寸 {size}, 无法进行转换。")
        return None

    top = box['top']
    left = box['left']
    h = box['height']
    w = box['width']

    # 检查边界框值是否有效
    if w <= 0 or h <= 0:
        print(f"警告: 无效的边界框尺寸 width={w}, height={h}。跳过此框。")
        return None
    if left < 0 or top < 0 or (left + w) > img_w or (top + h) > img_h:
        # 尝试钳制到边界，如果仍然无效则跳过
        print(f"警告: 边界框坐标 ({left},{top},{w},{h}) 超出图像边界 ({img_w},{img_h})。尝试钳制。")
        x1 = max(0.0, float(left))
        y1 = max(0.0, float(top))
        x2 = min(float(img_w), float(left + w))
        y2 = min(float(img_h), float(top + h))
        left = x1
        top = y1
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            print(f"警告: 钳制后的边界框无效 width={w}, height={h}。跳过此框。")
            return None

    dw = 1. / img_w
    dh = 1. / img_h

    x_center = (left + w / 2.0) * dw
    y_center = (top + h / 2.0) * dh
    norm_w = w * dw
    norm_h = h * dh

    # 再次限制值在0到1之间 (钳制后理论上不需要，但保险起见)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return (x_center, y_center, norm_w, norm_h)

if __name__ == "__main__":
    print("开始处理 AU-Air 数据集...")

    # 1. 检查源文件和目录是否存在
    if not os.path.isdir(source_images_dir):
        print(f"错误: 源图像目录未找到: {source_images_dir}")
        exit()
    if not os.path.exists(annotations_json_path):
        print(f"错误: 标注 JSON 文件未找到: {annotations_json_path}")
        exit()

    # 2. 加载并处理 JSON 标注
    print(f"加载标注文件: {annotations_json_path}")
    try:
        with open(annotations_json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 解析 JSON 文件失败: {e}")
        exit()
    except Exception as e:
        print(f"错误: 读取 JSON 文件时发生未知错误: {e}")
        exit()

    # 提取类别 (虽然直接用bbox里的class，但可以打印出来确认)
    categories = data.get('categories', [])
    if categories:
        print("数据集类别:")
        for i, cat in enumerate(categories):
            print(f"  {i}: {cat}")
    else:
        print("警告: JSON 文件中未找到 'categories' 列表。")

    # 将标注按图像名组织
    annotations_dict = {}
    print("处理 JSON 标注...")
    json_annotations = data.get('annotations', [])
    if not json_annotations:
        print("错误: JSON 文件中未找到 'annotations' 列表或列表为空。")
        exit()

    processed_count = 0
    skipped_count = 0
    for ann in tqdm(json_annotations, desc="处理JSON条目"):
        img_name = ann.get('image_name')
        # --- 处理潜在的键名不一致问题 ("image_width:" vs "image_width") ---
        img_width = ann.get('image_width')
        if img_width is None:
             img_width = ann.get('image_width:') # 尝试带冒号的键
        img_height = ann.get('image_height')
        # --- 结束处理键名不一致 ---
        bboxes = ann.get('bbox', [])

        if not img_name or img_width is None or img_height is None:
            print(f"警告: JSON 中条目缺少 image_name, image_width 或 image_height。跳过。条目内容: {ann}")
            skipped_count += 1
            continue

        try:
            img_width = float(img_width)
            img_height = float(img_height)
            if img_width <= 0 or img_height <= 0:
                 print(f"警告: 图像 '{img_name}' 的尺寸无效 ({img_width}x{img_height})。跳过。")
                 skipped_count += 1
                 continue
        except (ValueError, TypeError):
             print(f"警告: 图像 '{img_name}' 的尺寸无法解析为数字 ({img_width}, {img_height})。跳过。")
             skipped_count += 1
             continue

        annotations_dict[img_name] = {
            'width': img_width,
            'height': img_height,
            'bboxes': bboxes
        }
        processed_count += 1

    print(f"JSON 处理完成: {processed_count} 个有效条目, {skipped_count} 个被跳过。")
    if processed_count == 0:
        print("错误: 未能从 JSON 中处理任何有效的标注信息。")
        exit()

    # 3. 获取图像文件列表并过滤
    print(f"扫描源图像目录: {source_images_dir}")
    # 使用 glob 获取所有 jpg 文件 (忽略大小写)
    # --- 修改开始: 使用更健壮的方法查找文件 --- 
    all_files_in_dir = glob.glob(os.path.join(source_images_dir, '*.*'))
    all_img_files_paths = [] # 重命名以避免歧义
    supported_extensions = {'.jpg', '.jpeg'} # AU-Air 似乎只有jpg/jpeg
    for f_path in all_files_in_dir:
        ext = os.path.splitext(f_path)[1].lower()
        if ext in supported_extensions:
            all_img_files_paths.append(f_path)
    # --- 修改结束 ---
    # all_img_files_paths = glob.glob(os.path.join(source_images_dir, '*.jpg')) + \
    #                       glob.glob(os.path.join(source_images_dir, '*.JPG')) + \
    #                       glob.glob(os.path.join(source_images_dir, '*.jpeg')) + \
    #                       glob.glob(os.path.join(source_images_dir, '*.JPEG'))

    valid_image_filenames = []
    print("验证图像文件与标注的对应关系...")
    found_count = 0
    missing_ann_count = 0
    for img_path in tqdm(all_img_files_paths, desc="验证图像"):
        img_filename = os.path.basename(img_path)
        if img_filename in annotations_dict:
            valid_image_filenames.append(img_filename)
            found_count += 1
        else:
            # print(f"警告: 图像文件 {img_filename} 在 JSON 标注中未找到对应条目。将不被包含在数据集中。")
            missing_ann_count += 1

    print(f"验证完成: {found_count} 个图像文件有对应的标注。 {missing_ann_count} 个图像文件缺少标注 (将被忽略)。")
    # 检查是否有图像标注了但没有对应文件 (可选)
    missing_file_count = 0
    image_filenames_set = {os.path.basename(p) for p in all_img_files_paths} # Create a set for faster lookup
    for ann_img_name in tqdm(annotations_dict, desc="检查缺失图像文件"):
        # 需要构建完整路径来检查，或者只比较文件名列表
        if ann_img_name not in image_filenames_set:
             # print(f"警告: JSON 中标注的图像 {ann_img_name} 在图像目录中未找到对应文件。该标注将被忽略。")
             missing_file_count += 1
    if missing_file_count > 0:
        print(f"另外发现 {missing_file_count} 个 JSON 标注条目缺少对应的图像文件 (将被忽略)。")

    if not valid_image_filenames:
        print("错误: 没有找到任何既有图像文件又有对应标注的数据。程序退出。")
        exit()

    # 4. 划分数据集
    print("开始划分数据集...")
    random.shuffle(valid_image_filenames)
    total_files = len(valid_image_filenames)
    train_end_idx = int(total_files * train_ratio)
    val_end_idx = train_end_idx + int(total_files * val_ratio)

    train_files = valid_image_filenames[:train_end_idx]
    val_files = valid_image_filenames[train_end_idx:val_end_idx]
    test_files = valid_image_filenames[val_end_idx:]

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    print(f"数据集划分结果:")
    print(f"  训练集: {len(train_files)} 文件")
    print(f"  验证集: {len(val_files)} 文件")
    print(f"  测试集: {len(test_files)} 文件")

    # 5. 创建目标目录并处理文件
    print("\n开始创建目标目录并复制/转换文件...")
    os.makedirs(yolo_images_base, exist_ok=True)
    os.makedirs(yolo_labels_base, exist_ok=True)
    print(f"YOLO 图像目标基础路径: {yolo_images_base}")
    print(f"YOLO 标签目标基础路径: {yolo_labels_base}")

    for split_name, file_list in splits.items():
        target_img_dir = os.path.join(yolo_images_base, split_name)
        target_lbl_dir = os.path.join(yolo_labels_base, split_name)
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        print(f"\n--- 处理 {split_name} ({len(file_list)} 文件) ---")
        copied_count = 0
        converted_count = 0

        for img_filename in tqdm(file_list, desc=f"处理 {split_name}"):
            img_src_path = os.path.join(source_images_dir, img_filename)
            img_dst_path = os.path.join(target_img_dir, img_filename)
            name_part = os.path.splitext(img_filename)[0]
            lbl_dst_path = os.path.join(target_lbl_dir, f"{name_part}.txt")

            # 1. 复制图像文件
            if not os.path.exists(img_src_path):
                print(f"警告: 在复制阶段源图像文件未找到: {img_src_path}。跳过。")
                continue
            try:
                shutil.copy2(img_src_path, img_dst_path)
                copied_count += 1
            except Exception as e:
                print(f"错误: 复制图像 {img_src_path} 到 {img_dst_path} 失败: {e}")
                continue # 复制失败则不生成标签

            # 2. 生成标签文件
            annotation_data = annotations_dict.get(img_filename)
            if not annotation_data:
                print(f"警告: 未找到图像 {img_filename} 的标注数据 (理论上不应发生)。跳过标签生成。")
                open(lbl_dst_path, 'w').close() # 创建空文件
                continue

            img_width = annotation_data['width']
            img_height = annotation_data['height']
            bboxes = annotation_data['bboxes']
            yolo_lines = []

            if not bboxes:
                 print(f"信息: 图像 {img_filename} 在 JSON 中没有边界框标注。生成空标签文件。")

            for bbox in bboxes:
                class_id = bbox.get('class')
                # 检查 class_id 是否有效 (应该是非负整数)
                if class_id is None or not isinstance(class_id, int) or class_id < 0:
                    print(f"警告: 图像 {img_filename} 中的一个边界框缺少有效 'class' ID。跳过此框。框数据: {bbox}")
                    continue

                yolo_bbox = convert_bbox_to_yolo((img_width, img_height), bbox)
                if yolo_bbox:
                    x_c, y_c, w, h = yolo_bbox
                    yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            # 写入YOLO标注文件
            try:
                with open(lbl_dst_path, 'w') as f_out:
                    f_out.writelines(yolo_lines)
                converted_count += 1
            except Exception as e:
                print(f"错误: 写入标签文件 {lbl_dst_path} 失败: {e}")

        print(f"--- {split_name} 处理完成 ---")
        print(f"已复制 {copied_count} 个图像文件。")
        print(f"已生成 {converted_count} 个标签文件。")

    print("\n所有处理完成！") 