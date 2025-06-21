import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
np.random.seed(0)

# 生成模拟数据
data = [
    [1.51, 2.71, 1.44, 0.72, 0.66, 1.59],
    [1.34, 3.88, 2.61, 0.49, 1.01, 2.13],
    [1.22, 3.43, 3.32, 1.21, 0.66, 2.35],
    [1.59, 4.01, 2.09, 1.58, 1.45, 1.92],
    [1.25, 2.15, 0.92, 0.23, 0.36, 0.99]
]

# 创建图像
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制箱线图
box = ax.boxplot(
    data, labels=['ResNet18', 'GoogLeNet', 'RegNet', 'ShuffleNetV2', 'SHCA-MobileNetV3'],
    patch_artist=True, showmeans=True, meanline=True
)

# 设置箱子的颜色（浅紫色填充，边框稍深）
for patch in box['boxes']:
    patch.set_facecolor('#e1bee7')  # 浅紫色
    patch.set_edgecolor('#7e57c2')  # 深紫色边框
    patch.set_linewidth(1.5)

# 设置所有线条颜色为深紫色（边框色）
for whisker in box['whiskers']:
    whisker.set(color='#7e57c2', linewidth=1.5)
for cap in box['caps']:
    cap.set(color='#7e57c2', linewidth=1.5)
for median in box['medians']:
    median.set(color='#7e57c2', linewidth=2)

# 去除均值虚线
for mean in box['means']:
    mean.set_visible(False)  # 隐藏均值线

# 标注每个数据点（橙色）
for i, data_points in enumerate(data):
    # 使用橙色标注数据点
    ax.scatter([i + 1] * len(data_points), data_points, color='#ff7043', zorder=5)

# 设置网格线（浅灰色实线）
ax.grid(True, axis='y', linestyle='-', color='lightgray', linewidth=0.7)

# 设置背景为纯白色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 设置标题、标签
ax.set_title('EER(%)', fontsize=14)
ax.set_ylabel('EER(%)', fontsize=12)

# 保存图像为PNG文件

plt.savefig('boxplot_with_points_EER.svg', format='svg', bbox_inches='tight')
# 显示图像
plt.show()
