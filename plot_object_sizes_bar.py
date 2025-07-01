import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse

def plot_object_sizes(language='cn'):
    """
    绘制各类别目标尺寸分布的堆叠柱状图
    
    参数:
        language: 'cn' 表示中文, 'en' 表示英文
    """
    # 设置中文字体
    if language == 'cn':
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 数据
    categories_en = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    categories_cn = ['行人', '人群', '自行车', '汽车', '厢型车', '卡车', '三轮车', '篷车', '公交车', '摩托车']
    
    categories = categories_cn if language == 'cn' else categories_en
    
    small_percentages = [84.1, 88.6, 70.6, 49.1, 44.6, 31.0, 48.0, 44.4, 27.2, 76.3]
    medium_percentages = [15.5, 11.1, 28.4, 43.5, 46.2, 52.5, 48.3, 50.7, 55.8, 23.0]
    large_percentages = [0.4, 0.3, 1.0, 7.4, 9.1, 16.5, 3.8, 4.9, 17.1, 0.7]
    
    # 设置图形大小
    plt.figure(figsize=(14, 8))
    
    # 设置柱状图的宽度
    bar_width = 0.8
    
    # 创建位置数组
    x = np.arange(len(categories))
    
    # 标签文本
    if language == 'cn':
        small_label = '小目标 (s)'
        medium_label = '中目标 (m)'
        large_label = '大目标 (l)'
        x_label = '类别'
        y_label = '百分比 (%)'
        title = '各类别中小、中、大目标的占比'
    else:
        small_label = 'Small (s)'
        medium_label = 'Medium (m)'
        large_label = 'Large (l)'
        x_label = 'Category'
        y_label = 'Percentage (%)'
        title = 'Distribution of Small, Medium and Large Objects by Category'
    
    # 创建堆叠柱状图 - 调整顺序：大目标在底部，中目标在中间，小目标在顶部
    plt.bar(x, large_percentages, bar_width, label=large_label, color='#e74c3c')
    plt.bar(x, medium_percentages, bar_width, bottom=large_percentages, label=medium_label, color='#2ecc71')
    plt.bar(x, small_percentages, bar_width, bottom=np.array(large_percentages) + np.array(medium_percentages), label=small_label, color='#3498db')
    
    # 添加标签和标题
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(x, categories, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=16)
    
    # 添加图例
    plt.legend(fontsize=16)
    
    # 在每个柱子上添加百分比标签
    for i in range(len(categories)):
        # 大目标标签
        if large_percentages[i] > 5:  # 只有当百分比大于5%时才显示标签，避免拥挤
            plt.text(i, large_percentages[i]/2, f'{large_percentages[i]}%', 
                    ha='center', va='center', fontsize=10, color='white')
        
        # 中目标标签
        if medium_percentages[i] > 5:
            plt.text(i, large_percentages[i] + medium_percentages[i]/2, f'{medium_percentages[i]}%', 
                    ha='center', va='center', fontsize=10, color='white')
        
        # 小目标标签
        if small_percentages[i] > 5:
            plt.text(i, large_percentages[i] + medium_percentages[i] + small_percentages[i]/2, 
                    f'{small_percentages[i]}%', ha='center', va='center', fontsize=10, color='white')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_file = 'object_size_distribution_cn.png' if language == 'cn' else 'object_size_distribution_en.png'
    plt.savefig(output_file, dpi=300)
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='绘制目标尺寸分布柱状图')
    parser.add_argument('--language', '-l', type=str, default='cn', choices=['cn', 'en'],
                        help='显示语言: cn为中文, en为英文 (默认: cn)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 绘制图表
    plot_object_sizes(args.language) 