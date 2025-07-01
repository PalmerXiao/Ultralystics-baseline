import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np

def plot_object_sizes_pie(language='cn'):
    """
    绘制单一类别目标尺寸分布的饼图（类别）
    
    参数:
        language: 'cn' 表示中文, 'en' 表示英文
    """
    # 设置中文字体
    if language == 'cn':
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # UAV数据
    dataset_name = 'UAVSwarm'
    small_percentage = 85.5
    medium_percentage = 13.6
    large_percentage = 0.9
    
    # 数据和标签
    sizes = [small_percentage, medium_percentage, large_percentage]
    
    # 标签文本
    if language == 'cn':
        small_label = f'小目标 (s)'
        medium_label = f'中目标 (m)'
        large_label = f'大目标 (l)'
        title = f'{dataset_name}数据集中小、中、大目标的分布'
    else:
        small_label = f'Small (s)'
        medium_label = f'Medium (m)'
        large_label = f'Large (l)'
        title = f'Distribution of Small, Medium and Large Objects in {dataset_name} Dataset'
    
    labels = [small_label, medium_label, large_label]
    
    # 颜色
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # 突出显示最大的部分
    explode = (0.1, 0, 0)  # 小目标部分突出
    
    # 设置图形大小
    plt.figure(figsize=(10, 8))
    
    # 创建饼图 - 将标签放在饼图内部
    # 使用一个函数来确定标签，避免浮点数精度问题
    def autopct_format(pct):
        # 根据百分比确定对应的标签索引
        # 使用接近判断而不是精确匹配
        total = sum(sizes)
        val = pct * total / 100.0
        # 找到最接近的值的索引
        idx = np.argmin(np.abs(np.array(sizes) - val))
        return f'{labels[idx]}\n{pct:.1f}%'
    
    wedges, texts, autotexts = plt.pie(
        sizes, 
        explode=explode, 
        labels=None,  # 不显示外部标签
        colors=colors, 
        autopct=autopct_format,
        shadow=True, 
        startangle=90,
        textprops={'fontsize': 14, 'color': 'white', 'weight': 'bold'}
    )
    
    # 添加标题
    plt.title(title, fontsize=16)
    
    # 确保饼图是圆形的
    plt.axis('equal')
    
    # 添加图例
    plt.legend(wedges, [f'{labels[i]} ({sizes[i]}%)' for i in range(len(labels))], 
              loc='best', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_file = 'uav_size_distribution_pie_cn.png' if language == 'cn' else 'uav_size_distribution_pie_en.png'
    plt.savefig(output_file, dpi=300)
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='绘制UAV类别目标尺寸分布饼图')
    parser.add_argument('--language', '-l', type=str, default='cn', choices=['cn', 'en'],
                        help='显示语言: cn为中文, en为英文 (默认: cn)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 绘制图表
    plot_object_sizes_pie(args.language) 