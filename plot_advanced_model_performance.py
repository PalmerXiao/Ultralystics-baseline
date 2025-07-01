import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch
from matplotlib.colors import to_rgb, LinearSegmentedColormap
import gc

def lighten_color(color, amount=0.5):
    """增加颜色的亮度"""
    try:
        c = np.array(to_rgb(color))
        return tuple(np.clip(c + (1 - c) * amount, 0, 1))
    except ValueError:
        return color

def darken_color(color, amount=0.3):
    """降低颜色的亮度"""
    try:
        c = np.array(to_rgb(color))
        return tuple(np.clip(c * (1 - amount), 0, 1))
    except ValueError:
        return color

def create_sphere_colormap(base_color):
    """创建球体的渐变色，从中心到边缘"""
    # 从亮色到基本色到暗色
    light_color = lighten_color(base_color, 0.6)
    dark_color = darken_color(base_color, 0.4)
    
    cdict = {
        'red': [(0.0, light_color[0], light_color[0]),
                (0.5, base_color[0], base_color[0]),
                (1.0, dark_color[0], dark_color[0])],
        'green': [(0.0, light_color[1], light_color[1]),
                  (0.5, base_color[1], base_color[1]),
                  (1.0, dark_color[1], dark_color[1])],
        'blue': [(0.0, light_color[2], light_color[2]),
                 (0.5, base_color[2], base_color[2]),
                 (1.0, dark_color[2], dark_color[2])]
    }
    
    return LinearSegmentedColormap('SphereGradient', cdict)

def plot_3d_sphere(ax, x, y, radius, color, zorder=1):
    """绘制真实的3D球体效果，带有阴影和单一高光"""
    # 获取深色和浅色版本
    dark_color = darken_color(color, 0.5)
    
    # 绘制阴影（在球体下方偏右）
    shadow = Circle((x + radius*0.15, y - radius*0.1), radius*1.05,
                   facecolor='black', alpha=0.2, 
                   edgecolor=None, linewidth=0, zorder=zorder-1)
    ax.add_patch(shadow)
    
    # 绘制主球体
    sphere = Circle((x, y), radius, 
                   facecolor=color, alpha=1.0,
                   edgecolor=dark_color, linewidth=2,
                   zorder=zorder)
    ax.add_patch(sphere)
    
    # 添加高光 (左上方，单一光源)
    highlight_size = radius * 0.55
    highlight_x = x - radius * 0.2
    highlight_y = y + radius * 0.3
    
    highlight = Circle((highlight_x, highlight_y), highlight_size,
                      facecolor='white', alpha=0.7,
                      edgecolor=None, linewidth=0,
                      zorder=zorder+1)
    ax.add_patch(highlight)
    
    return sphere

def plot_model_comparison(data, dataset_name="VisDrone", 
                         title=None,
                         xlabel="Number of Parameters (M)", 
                         ylabel="mAP50 (%)",
                         gflops_to_radius_scale=2.5,  # 大幅增大球体尺寸
                         x_min=None, x_max=None,
                         y_min=None, y_max=None,
                         output_filename=None,
                         figsize=(14, 10),
                         add_group_outlines=False):
    """
    绘制模型性能比较图，带有3D球体效果
    """
    if not data:
        print("没有提供数据进行绘图。")
        return

    # 清理内存
    plt.close('all')
    gc.collect()

    # 设置标题和文件名
    if title is None:
        title = f"Performance on {dataset_name} Dataset"
    
    if output_filename is None:
        output_filename = f"{dataset_name.lower()}_model_comparison.png"

    # 提取数据
    names = [d['name'] for d in data]
    map_values = [d['map50'] for d in data]
    params_m = [d['params'] / 1_000_000 for d in data]
    gflops_values = [d['gflops'] for d in data]
    groups = [d.get('group', 'Default') for d in data]

    # 定义每个模型系列的RGB颜色
    group_colors = {
        'YOLOv5': (0.1, 0.6, 0.3),      # 绿色
        'YOLOv8': (0.95, 0.8, 0.6),     # 浅褐色
        'YOLOv9': (0.95, 0.95, 0.0),    # 黄色
        'BiGA-YOLO': (0.9, 0.2, 0.2),   # 红色
        'BSR5-DETR': (0.2, 0.8, 0.2),   # 绿色
        'RT-DETR': (0.2, 0.6, 0.9),     # 蓝色
        'BSR5-YOLO': (0.9, 0.2, 0.2),   # 红色
        'AMSP-UOD': (0.1, 0.1, 0.7),    # 深蓝色
        'Default': (0.5, 0.5, 0.5)      # 灰色
    }

    # 设置图形
    plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置白色背景和网格
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # 计算坐标范围
    if x_min is None:
        x_min = min(params_m) * 0.8 if params_m else 0
    if x_max is None:
        x_max = max(params_m) * 1.2 if params_m else 30
    if y_min is None:
        y_min = min(map_values) * 0.97 if map_values else 0
    if y_max is None:
        y_max = max(map_values) * 1.03 if map_values else 100
    
    # 设置轴范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 确保纵横比例相等（使圆形显示为圆形而不是椭圆）
    ax.set_aspect('equal', adjustable='box')
    
    # 存储每个球体的信息，以便后期处理
    spheres = []
    
    # 绘制每个点
    for i, (name, map_val, param, gflop, group) in enumerate(
            zip(names, map_values, params_m, gflops_values, groups)):
        
        # 获取颜色
        base_color = group_colors.get(group, group_colors['Default'])
        
        # 计算球体半径 (基于GFLOPS)
        radius = gflops_to_radius_scale * np.sqrt(gflop) / 10
        
        # 绘制3D球体
        sphere = plot_3d_sphere(
            ax, param, map_val, radius, base_color, 
            zorder=10 + i
        )
        
        # 存储信息
        spheres.append({
            'name': name,
            'x': param,
            'y': map_val,
            'radius': radius,
            'color': base_color,
            'group': group
        })
        
        # 添加连接线和标签
        # 计算连接线的起点和终点
        line_start_x = param
        line_start_y = map_val
        
        # 根据球体位置决定标签的放置位置
        if param < (x_max - x_min) * 0.5 + x_min:
            # 左侧球体，标签放右侧
            label_x = param + radius * 1.2
            ha = 'left'
            # 连接线终点
            line_end_x = param + radius * 1.0
        else:
            # 右侧球体，标签放左侧
            label_x = param - radius * 1.2
            ha = 'right'
            # 连接线终点
            line_end_x = param - radius * 1.0
        
        # 决定标签的垂直位置
        if map_val > (y_max - y_min) * 0.5 + y_min:
            # 上半部分的球体，标签放在下面
            label_y = map_val - radius * 0.8
            va = 'top'
            # 连接线终点
            line_end_y = map_val - radius * 0.6
        else:
            # 下半部分的球体，标签放在上面
            label_y = map_val + radius * 0.8
            va = 'bottom'
            # 连接线终点
            line_end_y = map_val + radius * 0.6
        
        # 绘制连接线
        ax.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
               color='black', linestyle='-', linewidth=1, zorder=5+i)
        
        # 添加标签文本 (模型名称，参数量和性能)
        label_text = f"{name}\nParam:{param:.2f}M\nAP:{map_val:.1f}%"
        ax.text(
            label_x, label_y, label_text,
            fontname='Arial', fontsize=9, fontweight='bold',
            ha=ha, va=va, zorder=20+i,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
        )
    
    # 如果启用组群轮廓，绘制组群轮廓 (实验性功能)
    if add_group_outlines:
        # 按组分组
        group_data = {}
        for s in spheres:
            if s['group'] not in group_data:
                group_data[s['group']] = []
            group_data[s['group']].append(s)
        
        # 对于每个有多个点的组，绘制轮廓
        for group, spheres_in_group in group_data.items():
            if len(spheres_in_group) >= 3:  # 至少需要3个点
                # 提取坐标
                xs = [s['x'] for s in spheres_in_group]
                ys = [s['y'] for s in spheres_in_group]
                
                # 简单的凸包轮廓 (仅示例)
                color = group_colors.get(group, group_colors['Default'])
                ax.plot(xs + [xs[0]], ys + [ys[0]], 
                       '--', color=color, alpha=0.6, linewidth=2, zorder=1)

    # 创建图例
    legend_elements = []
    legend_labels = []
    unique_groups = sorted(list(set(groups)))
    
    for group in unique_groups:
        if group in group_colors:
            color = group_colors[group]
            # 使用圆形标记
            patch = Patch(facecolor=color, edgecolor=darken_color(color, 0.5),
                         label=group)
            legend_elements.append(patch)
            legend_labels.append(group)
    
    # 添加图例
    legend = ax.legend(
        legend_elements, legend_labels,
        loc='lower right', title="Models",
        frameon=True, framealpha=0.9, edgecolor='black',
        fontsize=10, title_fontsize=11
    )
    
    # 设置轴标签和标题
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # 设置刻度标签
    ax.tick_params(axis='both', labelsize=10)

    # 保存原始长宽比的设置
    orig_size = fig.get_size_inches()
    
    # 调整布局（避免使用tight_layout，它可能会破坏长宽比）
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    try:
        # 恢复原始长宽比
        fig.set_size_inches(orig_size)
        
        # 保存和显示图像
        if output_filename:
            plt.savefig(output_filename, dpi=180, bbox_inches='tight')
            print(f"图像已保存至 {output_filename}")
        
        plt.show()
    finally:
        # 清理内存
        plt.close('all')
        gc.collect()

# 示例用法
if __name__ == "__main__":
    # 示例数据 - VisDrone数据集上的结果
    visdrone_data = [
        {'name': 'YOLOv5n', 'map50': 33.0, 'gflops': 7.1, 'params': 2504894, 'group': 'YOLOv5'},
        {'name': 'YOLOv5s', 'map50': 39.0, 'gflops': 23.8, 'params': 9115406, 'group': 'YOLOv5'},
        {'name': 'YOLOv5m', 'map50': 42.1, 'gflops': 64.0, 'params': 25051006, 'group': 'YOLOv5'},
        {'name': 'YOLOv8n', 'map50': 33.5, 'gflops': 8.1, 'params': 3007598, 'group': 'YOLOv8'},
        {'name': 'YOLOv8s', 'map50': 39.6, 'gflops': 28.5, 'params': 11129454, 'group': 'YOLOv8'},
        {'name': 'YOLOv8m', 'map50': 42.6, 'gflops': 78.7, 'params': 25845550, 'group': 'YOLOv8'},
        {'name': 'BiGA-YOLO-n', 'map50': 34.2, 'gflops': 7.6, 'params': 2140771, 'group': 'BiGA-YOLO'},
        {'name': 'BiGA-YOLO-s', 'map50': 41.0, 'gflops': 26.7, 'params': 7935779, 'group': 'BiGA-YOLO'},
        {'name': 'BiGA-YOLO-m', 'map50': 44.2, 'gflops': 74.8, 'params': 18811555, 'group': 'BiGA-YOLO'},
    ]

    # 绘制VisDrone数据集的结果
    plot_model_comparison(
        visdrone_data,
        dataset_name="VisDrone",
        gflops_to_radius_scale=2.5  # 大幅增大球体大小
    )

    """
    # 使用其他数据集的示例
    coco_data = [
        # 在这里添加COCO数据集上的结果
        {'name': 'Model1', 'map50': 45.0, 'gflops': 10.0, 'params': 3000000, 'group': 'GroupA'},
        # ...更多数据
    ]

    plot_model_comparison(  
        coco_data,
        dataset_name="COCO",
        gflops_to_radius_scale=0.4
    )
    """ 