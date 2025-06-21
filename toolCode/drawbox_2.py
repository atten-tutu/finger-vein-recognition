import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)


data = [
    [98.17, 94.79, 97.22, 99.42, 99.44, 98.17],
    [97.91, 93.24, 94.07, 99.65, 98.99, 96.27],
    [98.60, 90.64, 93.78, 98.68, 99.46, 96.63],
    [97.95, 92.02, 97.10, 97.99, 98.32, 97.57],
    [98.75, 95.36, 99.08, 99.97, 99.77, 99.01]
]


fig, ax = plt.subplots(figsize=(10, 6))


box = ax.boxplot(
    data, labels=['ResNet18', 'GoogLeNet', 'RegNet', 'ShuffleNetV2', 'SHCA-MobileNetV3'],
    patch_artist=True, showmeans=True, meanline=True
)


for patch in box['boxes']:
    patch.set_facecolor('#f8bbd0')  # 粉色
    patch.set_edgecolor('#f06292')  # 深粉色边框
    patch.set_linewidth(1.5)


for whisker in box['whiskers']:
    whisker.set(color='#f06292', linewidth=1.5)
for cap in box['caps']:
    cap.set(color='#f06292', linewidth=1.5)
for median in box['medians']:
    median.set(color='#f06292', linewidth=2)


for mean in box['means']:
    mean.set_visible(False)  # 隐藏均值线


for i, data_points in enumerate(data):
    # 使用橙色标注数据点
    ax.scatter([i + 1] * len(data_points), data_points, color='#ff7043', zorder=5)


ax.grid(True, axis='y', linestyle='-', color='lightgray', linewidth=0.7)


fig.patch.set_facecolor('white')
ax.set_facecolor('white')


ax.set_title('TAR@FAR=0.01(%)', fontsize=14, fontweight='bold')
ax.set_ylabel('TAR@FAR=0.01(%)', fontsize=12)


plt.savefig('boxplot_with_points.svg', format='svg', bbox_inches='tight')

plt.show()