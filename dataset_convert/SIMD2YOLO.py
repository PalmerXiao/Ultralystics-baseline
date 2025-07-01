import os
import shutil
from tqdm import tqdm
import glob

# --- 配置 ---
simd_root = r"E:\PycharmProject\Datasets\CV\SIMD"
delete_original_folders = True # 设置为True，在移动完成后删除空的 training 和 validation 文件夹
# --- End 配置 ---

# 定义源文件夹和目标文件夹的基础路径
source_dirs = {
    'train': os.path.join(simd_root, "training"),
    'val': os.path.join(simd_root, "validation")
}

target_images_base = os.path.join(simd_root, "images")
target_labels_base = os.path.join(simd_root, "labels")

# 创建目标文件夹结构
for split in ['train', 'val']:
    os.makedirs(os.path.join(target_images_base, split), exist_ok=True)
    os.makedirs(os.path.join(target_labels_base, split), exist_ok=True)

print("目标文件夹结构已创建或已存在。")

# 处理每个划分 (train, val)
for split, source_dir in source_dirs.items():
    print(f"\n--- 开始处理 {split} 数据 ({source_dir}) ---")

    if not os.path.isdir(source_dir):
        print(f"警告: 源文件夹 {source_dir} 不存在，跳过此划分。")
        continue

    target_img_dir = os.path.join(target_images_base, split)
    target_lbl_dir = os.path.join(target_labels_base, split)

    # 查找所有 jpg 和 txt 文件
    all_files = glob.glob(os.path.join(source_dir, '*.*'))
    jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]
    txt_files = [f for f in all_files if f.lower().endswith('.txt')]

    print(f"找到 {len(jpg_files)} 个图像文件和 {len(txt_files)} 个标签文件。")

    moved_images = 0
    moved_labels = 0

    # 移动图像文件
    print(f"移动图像文件到 {target_img_dir}...")
    for src_path in tqdm(jpg_files, desc=f"移动 {split} 图像"):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(target_img_dir, filename)
        try:
            shutil.move(src_path, dst_path)
            moved_images += 1
        except Exception as e:
            print(f"移动文件 {src_path} 到 {dst_path} 时出错: {e}")

    # 移动标签文件
    print(f"移动标签文件到 {target_lbl_dir}...")
    for src_path in tqdm(txt_files, desc=f"移动 {split} 标签"):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(target_lbl_dir, filename)
        try:
            shutil.move(src_path, dst_path)
            moved_labels += 1
        except Exception as e:
            print(f"移动文件 {src_path} 到 {dst_path} 时出错: {e}")

    print(f"--- {split} 处理完成 ---")
    print(f"已移动 {moved_images} 个图像文件。")
    print(f"已移动 {moved_labels} 个标签文件。")

    # (可选) 删除空的源文件夹
    if delete_original_folders:
        try:
            if not os.listdir(source_dir): # 检查文件夹是否为空
                os.rmdir(source_dir)
                print(f"已删除空的源文件夹: {source_dir}")
            else:
                # 如果文件夹不为空（例如，有非jpg/txt文件或移动失败），则打印警告
                print(f"警告: 源文件夹 {source_dir} 不为空，未删除。")
        except OSError as e:
            print(f"删除文件夹 {source_dir} 时出错: {e}")


print("\n所有文件移动完成！")
