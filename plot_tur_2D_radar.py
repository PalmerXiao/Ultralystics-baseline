## 绘制雷达图，这里面对数据归一化了
import matplotlib.pyplot as plt
import numpy as np

def create_radar_chart(data, colors, markers, markersizes, order_models, order_metrics, output_file):
    """
    Creates a radar chart from the provided data and saves it to a file.
    
    Args:
        data (dict): Dictionary containing models, metrics, and values
        colors (list): List of colors in the same order as order_metrics
        markers (list): List of marker styles for each metric in the same order as order_metrics
        markersizes (list): List of marker sizes for each metric in the same order as order_metrics
        order_models (list): List of models in the desired display order
        order_metrics (list): List of metrics in the desired display order
        output_file (str): Output filename for the saved plot including path
    """
    # 设置字体为新罗马字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 获取模型和指标
    models = order_models
    metrics = order_metrics
    
    # 准备调整后的指标数据
    adjusted_data = []
    
    # 为每个指标找到对应的原始索引并重排数据
    for metric in metrics:
        if metric not in data['metrics']:
            print(f"Warning: Metric '{metric}' not found in data dictionary")
            continue
            
        # 对每个模型按照新顺序重排数据
        metric_values = []
        for model in models:
            if model in data['models']:
                # 获取模型在原始数据中的索引
                model_idx = data['models'].index(model)
                # 获取该模型对应指标的值
                metric_values.append(data[metric][model_idx])
            else:
                print(f"Warning: Model '{model}' not found in data dictionary")
                metric_values.append(0)  # 默认值
        
        adjusted_data.append(metric_values)
    
    # 归一化处理（基于重组后的数据）
    max_values = [max(metric_data) for metric_data in adjusted_data]
    normalized_data = [
        [value/max_val*100 for value in metric_data] 
        for metric_data, max_val in zip(adjusted_data, max_values)
    ]
    
    # 雷达图坐标计算
    angles = np.linspace(0, 2*np.pi, len(models), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    
    # 绘制每个指标 - 直接使用传入的颜色、标记和大小，顺序与metrics一致
    for idx, metric in enumerate(metrics):
        values = normalized_data[idx] + [normalized_data[idx][0]]  # 闭合数据
        ax.plot(angles, values, color=colors[idx], linewidth=2, label=metric,
                marker=markers[idx], markersize=markersizes[idx], markeredgecolor='white', markeredgewidth=0.5)
        ax.fill(angles, values, color=colors[idx], alpha=0.1)
    
    # 极坐标设置
    ax.set_theta_offset(np.pi/2)  # 设置0度方向为顶部
    ax.set_theta_direction(-1)    # 顺时针方向
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models, fontsize=12, color='black')
    ax.tick_params(axis='x', pad=15)  # 调整标签位置
    
    # 径向轴设置
    ax.set_rlabel_position(0)
    plt.yticks([40, 50, 60, 70, 80, 90, 100], ["40%", "50%", "60%", "70%", "80%", "90%", "100%"], 
               color="grey", size=11)
    plt.ylim(40, 100)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == "__main__":
    data = {
        'models': [
            "MyModel", "Model1", "Model2", "Model3", "Model4", "Model5", "Model6"
        ],
        'metrics':      ["P", "R", "mAP@0.5", "mAP@0.5:0.95"],
        'P':            [95.2, 82.5, 75.9, 83.8, 86.1, 73.2, 85.3],
        'R':            [93.3, 84.6, 69.0, 64.6, 81.3, 50.1, 66.9],
        'mAP@0.5':      [97.6, 71.9, 83.2, 73.8, 80.6, 59.5, 72.9],
        'mAP@0.5:0.95': [96.9, 64.5, 80.4, 51.6, 73.6, 78.1, 80.7]
    }
    # 定义显示顺序（可以自由调整）
    order_models = ["MyModel", "Model1", "Model2", "Model3", "Model4", "Model5", "Model6"]
    order_metrics = ['P', 'R', 'mAP@0.5:0.95', 'mAP@0.5']
    
    # 自定义颜色 - 与 order_metrics 一一对应 (搭配建议，这个图随便搭配，颜色不是很怪就可以，个人倾向，纯色渐变)
    # colors = ['#5EB2F7', '#369FF4', '#0E8CF2', '#0B75CB'] ## 深蓝
    # colors = ['#03A9F4', '#28B9FD', '#58C9FD', '#88D8FE'] ## 淡蓝
    # colors = ['#449A47', '#55B458', '#71C174', '#8ECD90'] ## 绿色
    # colors = ['#FF9800', '#FFA824', '#FFB649', '#FFC56D'] ## 橘色
    # colors = ['#78BC9F', '#74C69D', '#95D5B2', '#B7E4C7'] ## 淡绿
    colors = ['#5EA0C7', '#6BC179', '#BEDEAB', '#B4D7E5'] ## NATURE
    # colors = ['#E07F86', '#F8BC7E', '#8FA2CD', '#CC976B'] ## NATURE1
    
    # 标记和标记大小 - 与 order_metrics 一一对应
    markers = ['o', 's', '^', 'D']      # 圆形、方形、三角形、菱形，与order_metrics一一对应
    markersizes = [8, 8, 8, 8]          # 对应标记大小，与order_metrics一一对应
    
    # 设置输出路径和文件名
    output_file = 'model_comparison_radar.png'
    
    # 创建雷达图
    create_radar_chart(data, colors, markers, markersizes, order_models, order_metrics, output_file)
