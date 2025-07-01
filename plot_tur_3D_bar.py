## 这个绘制3D对比图，但是这里面对数据归一化了，如有需要可以让AI去掉归一化
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager as fm
import matplotlib.ticker as ticker

def create_3d_barchart(data, colors, order_methods, order_metrics, output_file):
    """
    Creates a 3D bar chart from the provided data and saves it to a file.
    
    Args:
        data (dict): Dictionary containing methods, metrics, and values
        colors (list): List of colors for the different metrics
        order_methods (list): List of methods in the desired display order
        order_metrics (list): List of metrics in the desired display order
        output_file (str): Output filename for the saved plot including path
    """
    # 设置字体为新罗马字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Reorder the methods and metrics according to the provided order
    methods = order_methods
    metrics = order_metrics
    
    # 数据归一化处理
    normalized_data = {}
    for metric in metrics:
        if metric not in data:
            print(f"Warning: Metric '{metric}' not found in data dictionary")
            continue
            
        # Create a new array with values in the correct order based on order_methods
        metric_data = []
        for method in methods:
            if method in data['models']:
                # Find the index of the method in the original data
                idx = data['models'].index(method)
                metric_data.append(data[metric][idx])
            else:
                print(f"Warning: Method '{method}' not found in data dictionary")
                metric_data.append(0)  # Default value when method is not found
        
        # Now normalize the reordered data
        min_val = min(metric_data)
        max_val = max(metric_data)
        
        # 如果是Params和GFLOPs这类越小越好的指标，进行反向归一化
        if metric in ['Params(M)', 'GFLOPs']:
            # 反向归一化 (1 - 归一化值) - 值越小，归一化后越接近1
            normalized_data[metric] = [1 - ((val - min_val) / (max_val - min_val)) if max_val != min_val else 0.5 for val in metric_data]
        else:
            # 正向归一化 - 值越大，归一化后越接近1
            normalized_data[metric] = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for val in metric_data]
    
    # 创建具有等宽比例的3D坐标
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴比例为相等
    # 计算合适的坐标范围
    num_methods = len(methods)
    num_metrics = len(metrics)
    
    # 创建3D坐标系，设置x和y的范围使得每个单位长度相等
    ax.set_box_aspect([num_methods/num_metrics, 1, 1])
    
    # 设置三维效果参数
    ax.view_init(elev=20, azim=-45)
    
    # 使用正方形作为底面的柱状图 (dx和dy设为相等)
    dx = 0.5
    dy = 0.5
    
    # 生成柱状图 - 使用归一化数据
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            if metric in normalized_data:
                ax.bar3d(
                    x=i, y=j, z=0,
                    dx=dx, dy=dy, dz=normalized_data[metric][i],
                    color=colors[j % len(colors)], shade=True, edgecolor='none', alpha=0.9
                )
    
    # 设置坐标轴范围，确保有足够的空间显示标签
    ax.set_xlim(0, len(methods))
    ax.set_ylim(0, len(metrics))
    ax.set_zlim(0, 1.1)  # 归一化后的z轴范围为0-1
    
    # 设置坐标轴标签和刻度
    ax.set_xticks(np.arange(len(methods)) + dx/2)
    ax.set_yticks(np.arange(len(metrics)) + dy/2)
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_zticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
    
    # 清除默认的标签
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 手动添加与轴平面平行的x轴标签
    for i, method in enumerate(methods):
        ax.text(i+dx/2+0.6, -0.8, 0, method, horizontalalignment='right', verticalalignment='center',
                fontsize=10, zdir='y', fontfamily='Times New Roman')  # 沿着y方向投影
    
    # 手动添加与轴平面平行的y轴标签
    for j, metric in enumerate(metrics):
        ax.text(len(methods)+1.0, j+dy/2-0.5, 0, metric, horizontalalignment='left', verticalalignment='center',
                fontsize=10, zdir='x', fontfamily='Times New Roman')  # 沿着x方向投影
    
    # 添加z轴标签
    # ax.set_zlabel('Normalized Score', fontsize=12, labelpad=10)
    
    # 添加三维网格线
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "color": "#666666"})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "color": "#666666"})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "color": "#666666"})
    
    # 添加图例
    patches = [plt.Rectangle((0,0),1,1, fc=color) for color in colors[:len(metrics)]]
    legend = ax.legend(patches, metrics, bbox_to_anchor=(0.95, 0.66), title='Metrics', fontsize=10, prop={'family': 'Times New Roman'})
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == "__main__":
    data = {
        'models': [
            "MyModel", "Model1", "Model2", "Model3", "Model4", "Model5", "Model6"
        ],
        'metrics':      ["P", "R", "mAP@0.5", "mAP@0.5:0.95"],
        'P':            [88.2, 78.5, 64.9, 66.8, 56.1, 50.2, 48.3],
        'R':            [88.3, 68.6, 66.0, 64.6, 52.3, 40.1, 20.9],
        'mAP@0.5':      [87.6, 80.9, 78.2, 73.8, 68.6, 59.5, 50.9],
        'mAP@0.5:0.95': [86.9, 80.5, 76.4, 63.6, 52.6, 48.1, 43.7]
    }
    
    # 定义显示顺序（可以自由调整）
    order_methods = ["MyModel", "Model1", "Model2", "Model3", "Model4", "Model5", "Model6"]  # 逆序排列
    order_metrics = ['P', 'R', 'mAP@0.5', 'mAP@0.5:0.95']
    
    # 自定义颜色-没有其他建议，试过了，3D配色不好配，都很丑，主要原因是因为立方体各个面颜色都不一样，看上去很乱
    #colors = ['#8FA2CD', '#F8BC7E', '#CC976B', '#E07F86'] ## NATURE 可以试试这个，我个人不喜欢
    colors = cm.Blues(np.linspace(0.3, 0.8, len(data['metrics'])))
    
    # 设置输出路径和文件名
    output_file = '3d.png'
    
    # 创建3D柱状图
    create_3d_barchart(data, colors, order_methods, order_metrics, output_file)