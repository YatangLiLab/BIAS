import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 seaborn 风格
sns.set(style="whitegrid")  # 使用白色背景网格
colors = sns.color_palette("husl", 6)  # 使用 Husl 调色板生成两种颜色

# 读取 CSV 文件
file_path = 'different_cs.csv'  # 确保文件路径正确
data = pd.read_csv(file_path)

# 检查数据
print("数据预览：")
print(data.head())

# 设置直方图的参数
bins = np.array([i * 0.05 for i in range(int(-0.3 / 0.05), int(1.0 / 0.05) + 1)])  # 桶的边界从 -0.3 到 1.0，步长为 0.05

# 计算两个数据集的直方图频率
hist_1_1cc, _ = np.histogram(data['1-1cc'], bins=bins)
hist_1_2cc, _ = np.histogram(data['1-2cc'], bins=bins)
hist_1_3cc, _ = np.histogram(data['1-3cc'], bins=bins)
hist_1_4cc, _ = np.histogram(data['1-4cc'], bins=bins)
hist_1_5cc, _ = np.histogram(data['1-5cc'], bins=bins)
hist_1_6cc, _ = np.histogram(data['1-6cc'], bins=bins)

# 计算每个桶内数量的比例 (1-4cc / 1-1cc)
# with np.errstate(divide='ignore', invalid='ignore'):  # 忽略除零和 NaN 警告
#     ratio = hist_1_4cc / hist_1_1cc
#     ratio = np.nan_to_num(ratio, nan=0.0, posinf=3, neginf=0)  # 处理 NaN 和无穷大

# 创建画布 (1行3列)
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# 左侧：正常直方图
axes[0].hist(
    [data['1-1cc'],data['1-2cc'],data['1-3cc'], data['1-4cc'],data['1-5cc'],data['1-6cc']],
    bins=bins,
    label=['1-1cc','1-2cc','1-3cc','1-4cc','1-5cc','1-6cc'],
    alpha=0.8,
    color=colors,  # 使用 Seaborn 配色
    edgecolor='black'
)
axes[0].set_title('Normal Histogram', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_xlim(-0.3, 1.0)
axes[0].legend(fontsize=10)

# 中间：对数刻度直方图
axes[1].hist(
    [data['1-1cc'],data['1-2cc'],data['1-3cc'], data['1-4cc'],data['1-5cc'],data['1-6cc']],
    bins=bins,
    label=['1-1cc','1-2cc','1-3cc','1-4cc','1-5cc','1-6cc'],
    alpha=0.8,
    color=colors,  # 使用 Seaborn 配色
    edgecolor='black'
)
axes[1].set_yscale('log')  # 设置 y 轴为对数刻度
axes[1].set_title('Log-Scale Histogram', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Value', fontsize=12)
axes[1].set_ylabel('Frequency (Log Scale)', fontsize=12)
axes[1].set_xlim(-0.3, 1.0)
axes[1].legend(fontsize=10)

# 右侧：桶内数量的比例图
# bin_centers = (bins[:-1] + bins[1:]) / 2  # 计算桶的中心位置
# axes[2].bar(
#     bin_centers,
#     ratio,
#     width=np.diff(bins)[0],
#     color=sns.color_palette("coolwarm", 1)[0],  # 使用 coolwarm 调色板中的红色
#     alpha=0.7,
#     edgecolor='black'
# )
# axes[2].axhline(1, color='red', linestyle='--', label='Baseline = 1')
# axes[2].set_title('Ratio of Bin Counts (1-4cc / 1-1cc)', fontsize=14, fontweight='bold')
# axes[2].set_xlabel('Value', fontsize=12)
# axes[2].set_ylabel('Ratio', fontsize=12)
# axes[2].set_xlim(-0.3, 1.0)
# axes[2].legend(fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()