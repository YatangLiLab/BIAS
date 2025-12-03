import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据：一个 6x3 的二维数组
data = np.array([
    [0.310, 0.293, 0.286],
    [0.304, 0.216, 0.286],
    [0.301, 0.245, 0.286],
    [0.307, 0.293, 0.296],
    [0.305, 0.293, np.nan],
    [0.307, np.nan, np.nan]
]).T

# 将数据转换为 DataFrame（可选）
df = pd.DataFrame(data, index=['1', '2', '3'], columns=['1', '2', '3', '4', '5', '6'])



# 绘制热力图
# sns.set(font_scale=1.5)
plt.figure(figsize=(9, 4.5))
sns.heatmap(df, annot=True, annot_kws={"size": 20},fmt=".3f", cmap="RdBu_r", cbar=False,vmin=0.28,)
plt.gca().invert_yaxis() 
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# 设置坐标轴标签
plt.xlabel('Center',fontsize = 24)
plt.ylabel('Delta',fontsize = 24)
plt.tight_layout()
# 显示图形
plt.show()