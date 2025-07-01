## 该代码的功能是处理指定文件夹中的所有XML文件，提取船只的中心点，并生成一个散点图
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
from collections import Counter

def get_bbox_centers(xml_path):
    """
    Extract center points and classes from XML file
    Returns: centers list [(x, y, class_name), ...]
    """
    centers = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        centers.append((center_x, center_y, name))
    
    return centers, (width, height)

def normalize_coordinates(centers, original_size, target_size=(1920, 1080)):
    """
    将坐标点归一化到目标尺寸
    
    Args:
        centers: 原始坐标点列表 [(x, y, class_name), ...]
        original_size: 原始图像尺寸 (width, height)
        target_size: 目标尺寸，默认1920x1080
        
    Returns:
        normalized_centers: 归一化后的坐标点列表
    """
    orig_width, orig_height = original_size
    target_width, target_height = target_size
    
    # 计算缩放比例
    width_scale = target_width / orig_width
    height_scale = target_height / orig_height
    
    # 对所有点进行坐标转换
    normalized_centers = []
    for x, y, class_name in centers:
        norm_x = x * width_scale
        norm_y = y * height_scale
        normalized_centers.append((norm_x, norm_y, class_name))
    
    return normalized_centers

def get_fixed_colors():
    """
    返回固定的颜色列表，使用明亮的16进制颜色代码
    """
    return [
        '#66CC33',     # 草绿色
        '#33B4FF',     # 天空蓝
        '#FF9933',     # 明橙色
        '#FF3366',     # 亮玫红
        '#FF6699',     # 粉红色
        '#9966FF',     # 亮紫色
        '#33CCCC',     # 碧绿色
        '#FF9966',     # 珊瑚色
        '#99CC33',     # 柠檬绿
        '#6699FF'      # 天蓝色
    ]

def plot_centers(xml_folder, output_folder, font_size):
    """
    Process all XML files in the folder and generate scatter plot
    """
    os.makedirs(output_folder, exist_ok=True)
    
    all_centers = []
    target_size = (1920, 1080)  # 目标尺寸
    
    # 收集所有中心点数据并归一化
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_folder, filename)
            centers, (width, height) = get_bbox_centers(xml_path)
            # 归一化坐标
            normalized_centers = normalize_coordinates(centers, (width, height), target_size)
            all_centers.extend(normalized_centers)
    
    # 统计每个类别的数量并排序
    class_counts = Counter(center[2] for center in all_centers)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 获取固定的颜色列表
    colors = get_fixed_colors()
    
    # 创建类别到颜色的映射
    class_to_color = {}
    for i, (class_name, _) in enumerate(sorted_classes):
        if i < len(colors):
            class_to_color[class_name] = colors[i]
        else:
            class_to_color[class_name] = 'darkgray'
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 按照类别数量从多到少的顺序绘制散点
    for class_name, count in sorted_classes:
        points = [(x, y) for x, y, name in all_centers if name == class_name]
        if points:
            x, y = zip(*points)
            plt.scatter(x, y, c=class_to_color[class_name], 
                       label=f'{class_name}', alpha=0.6, s=15)
    
    plt.xlabel('Width of images(pixel)', fontsize=font_size)
    plt.ylabel('Height of images(pixel)', fontsize=font_size)
    plt.legend(loc='lower right', frameon=True, fontsize=font_size-3)
    
    # 设置坐标轴范围为目标尺寸
    plt.xlim(0, target_size[0])
    plt.ylim(target_size[1], 0)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(200))
    ax.yaxis.set_major_locator(plt.MultipleLocator(200))
    ax.tick_params(axis='both', labelsize=font_size)
    
    output_path = os.path.join(output_folder, 'data_scatter_Mcship.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Normalized scatter plot has been saved to: {output_path}")

if __name__ == "__main__":
    # Configuration
    FONT_SIZE = 25
    XML_FOLDER = r"F:\c盘\suanfa\ultralytics-20250220\ultralytics-main\dataset\VOCdevkit\Annotations"
    OUTPUT_FOLDER = r"F:\c盘\suanfa\ultralytics-20250220\ultralytics-main\dataset"
    
    plot_centers(XML_FOLDER, OUTPUT_FOLDER, FONT_SIZE)