
import os
import shutil
import random
from tqdm import tqdm
import glob

# --- 配置 ---
dataset_root = r'E:\PycharmProject\Datasets\CV\UAVDT'
# --- End 配置 ---

# 定义源文件夹路径
source_train_img = os.path.join(dataset_root, "train", "images")
source_train_lbl = os.path.join(dataset_root, "train", "labels")
source_val_img = os.path.join(dataset_root, "val", "images")
source_val_lbl = os.path.join(dataset_root, "val", "labels")

# 定义目标文件夹路径
target_images_base = os.path.join(dataset_root, "images")
target_labels_base = os.path.join(dataset_root, "labels")

# 创建目标文件夹结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(target_images_base, split), exist_ok=True)
    os.makedirs(os.path.join(target_labels_base, split), exist_ok=True)

print("目标文件夹结构已创建或已存在。")

# 获取所有图像文件名（不包括路径）
train_images = [os.path.basename(f) for f in glob.glob(os.path.join(source_train_img, "*.jpg"))]
val_images = [os.path.basename(f) for f in glob.glob(os.path.join(source_val_img, "*.jpg"))]
all_images = train_images + val_images

# 打乱文件顺序
random.seed(42)  # 设置随机种子以确保结果可复现
random.shuffle(all_images)

# 计算分割点
total_files = len(all_images)
train_split = int(total_files * 0.7)
val_split = int(total_files * 0.1)

# 分割数据集
train_files = all_images[:train_split]
val_files = all_images[train_split:train_split+val_split]
test_files = all_images[train_split+val_split:]

print(f"总文件数: {total_files}")
print(f"训练集: {len(train_files)} 文件 (70%)")
print(f"验证集: {len(val_files)} 文件 (10%)")
print(f"测试集: {len(test_files)} 文件 (20%)")

# 定义复制函数
def copy_files(file_list, source_img_dirs, source_lbl_dirs, target_img_dir, target_lbl_dir):
    copied_images = 0
    copied_labels = 0
    
    for filename in tqdm(file_list, desc=f"复制文件到 {os.path.basename(target_img_dir)}"):
        # 处理图像文件
        basename = os.path.splitext(filename)[0]
        label_filename = f"{basename}.txt"
        
        # 查找源文件
        img_src_path = None
        lbl_src_path = None
        
        for img_dir in source_img_dirs:
            temp_path = os.path.join(img_dir, filename)
            if os.path.exists(temp_path):
                img_src_path = temp_path
                break
                
        for lbl_dir in source_lbl_dirs:
            temp_path = os.path.join(lbl_dir, label_filename)
            if os.path.exists(temp_path):
                lbl_src_path = temp_path
                break
        
        # 复制图像文件
        if img_src_path:
            img_dst_path = os.path.join(target_img_dir, filename)
            try:
                shutil.copy2(img_src_path, img_dst_path)
                copied_images += 1
            except Exception as e:
                print(f"复制图像文件 {img_src_path} 到 {img_dst_path} 时出错: {e}")
        
        # 复制标签文件
        if lbl_src_path:
            lbl_dst_path = os.path.join(target_lbl_dir, label_filename)
            try:
                shutil.copy2(lbl_src_path, lbl_dst_path)
                copied_labels += 1
            except Exception as e:
                print(f"复制标签文件 {lbl_src_path} 到 {lbl_dst_path} 时出错: {e}")
    
    return copied_images, copied_labels

# 准备源目录列表
source_img_dirs = [source_train_img, source_val_img]
source_lbl_dirs = [source_train_lbl, source_val_lbl]

# 复制训练集文件
print("\n--- 复制训练集文件 ---")
train_img_copied, train_lbl_copied = copy_files(
    train_files, 
    source_img_dirs, 
    source_lbl_dirs, 
    os.path.join(target_images_base, "train"), 
    os.path.join(target_labels_base, "train")
)
print(f"已复制 {train_img_copied} 个训练图像和 {train_lbl_copied} 个训练标签文件。")

# 复制验证集文件
print("\n--- 复制验证集文件 ---")
val_img_copied, val_lbl_copied = copy_files(
    val_files, 
    source_img_dirs, 
    source_lbl_dirs, 
    os.path.join(target_images_base, "val"), 
    os.path.join(target_labels_base, "val")
)
print(f"已复制 {val_img_copied} 个验证图像和 {val_lbl_copied} 个验证标签文件。")

# 复制测试集文件
print("\n--- 复制测试集文件 ---")
test_img_copied, test_lbl_copied = copy_files(
    test_files, 
    source_img_dirs, 
    source_lbl_dirs, 
    os.path.join(target_images_base, "test"), 
    os.path.join(target_labels_base, "test")
)
print(f"已复制 {test_img_copied} 个测试图像和 {test_lbl_copied} 个测试标签文件。")

print("\n所有文件复制完成！")
