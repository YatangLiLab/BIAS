import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

def multi_violin_subplot(todo_keys=None, metric_=None):
    """
    绘制包含多个主键的提琴图，每个主键的所有分组绘制在同一个子图中，并在提琴图下方标注样本数量。
    """
    # 加载数据
    csv_item = pd.read_csv('processed_different_cs.csv')
    attribute_dict = csv_item.to_dict()

    # 定义主键集合
    key_set = {
        # first line
        'object': ['O0', 'O1', 'O2', 'O>/3'],
        'people': ['P0', 'P1', 'P2', 'P>/3'],
        'Camera': ['Camera stable', 'Camera slow', 'Camera fast'],
        # second line left
        'Content': ['Content stable', 'Content slow', 'Content fast'],
        'time': [' night', 'day', ' indoor'],
        # second line right
        'Category': ['Daily activity', 'Sport', 'Social activity', 'Artistic performance', 'Animal', 'Artifact', 'Landscape'],
        # third line
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
    }

    Display_keyset = {
        # first line
        'object': ['O=0', 'O=1', 'O=2', 'O>/3'],
        'people': ['P=0', 'P=1', 'P=2', 'P>/3'],
        'Camera': ['stable', 'slow', 'fast'],
        # second line left
        'Content': ['stable', 'slow', 'fast'],
        'time': [ 'night', 'day', 'indoor'],
        # second line right
        'Category': ['Daily', 'Sport', 'Social', 'Artistic', 'Animal', 'Artifact', 'Landscape'],
        # third line
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
    }

    # 筛选需要的主键
    if todo_keys is None:
        todo_keys = list(key_set.keys())

    # 创建画布和子图布局
    fig = plt.figure(figsize=(15, 15))


    # 遍历每个主键
    for idx, todo_key in enumerate(todo_keys):
        # 动态调整子图布局
        if todo_key == 'GTPeaks' or todo_key == 'DBSCAN Peak':
            ax = plt.subplot(2, 14, (19,28))  # GTPeaks单独占一行
        elif todo_key == 'object':
            ax = plt.subplot(2, 14, (1,4))
        elif todo_key == 'people':
            ax = plt.subplot(2, 14, (5,8))
        elif todo_key == 'Camera':
            ax = plt.subplot(2, 14, (9,11))
        elif todo_key == 'Content':
            ax = plt.subplot(2, 14, (12,14))
        elif todo_key == 'time':
            ax = plt.subplot(2, 14, (15,18))
        elif todo_key == 'Category':
            continue
            ax = plt.subplot(3, 2, 4)

        # 数据处理
        multiple_axis = [[] for _ in range(len(key_set[todo_key]))]
        for index in attribute_dict['video_name'].keys():
            for i, key in enumerate(key_set[todo_key]):
                if todo_key == 'GTPeaks':
                    if int(attribute_dict[todo_key][index]) == int(key):  # 匹配数字
                        multiple_axis[i].append(attribute_dict[metric_][index])
                if todo_key == 'DBSCAN Peak':
                    if key == '>10':  # 匹配数字
                        if attribute_dict[todo_key][index] >10:
                            multiple_axis[i].append(attribute_dict[metric_][index])
                    elif int(attribute_dict[todo_key][index]) == int(key):  # 匹配数字
                        multiple_axis[i].append(attribute_dict[metric_][index])
                elif attribute_dict[key][index] == 1 or attribute_dict[key][index] == '1':
                    multiple_axis[i].append(attribute_dict[metric_][index])

        # 过滤掉空组
        available_idxs = [i for i, group in enumerate(multiple_axis) if len(group) > 0]
        multiple_axis = [multiple_axis[i] for i in available_idxs]
        key_labels = [key_set[todo_key][i] for i in available_idxs]

        # 在当前子图中绘制提琴图
        data = []
        labels = []
        for group, label in zip(multiple_axis, key_labels):
            data.extend(group)
            labels.extend([label] * len(group))

        df_violin = pd.DataFrame({
            'value': data,
            'group': labels
        })

        if todo_key == 'Category':
            my_order = df_violin.groupby(by=["group"])["value"].median().sort_values().iloc[::-1].index
            sns.violinplot(
                data=df_violin,
                x='group',
                y='value',
                color='red',  # 统一颜色
                alpha=0.6,     # 设置透明度
                saturation = 0.7,
                inner="box",  # 显示四分位数
                cut=0,
                bw_method=0.3,
                linewidth=2,
                ax=ax,
                order=my_order
            )
            # print(my_order)
            new_order = [key_set[todo_key].index(val) for val in my_order]
            # print(new_order)
            Display_keyset[todo_key] = [Display_keyset[todo_key][new_o] for new_o in new_order]
            # print(Display_keyset[todo_key])
            # quit()
        else:
            sns.violinplot(
                data=df_violin,
                x='group',
                y='value',
                color='red',  # 统一颜色
                alpha=0.6,     # 设置透明度
                saturation = 0.7,
                inner='box',# "quartile",  # 显示四分位数
                cut=0,
                bw_method=0.3,
                linewidth=2,
                ax=ax
            )

        # 添加标题和标签
        ax.set_title(f'{todo_key.strip().capitalize()}')
        ax.set_xlabel('')
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(Display_keyset[todo_key])
        # ax.set_xlabel(todo_key)
        if idx % 3 == 0:
            ax.set_ylabel('Correlation Coefficient')
        else:
            ax.set_ylabel('')
        # ax.tick_params(axis='x', rotation=45)
        # ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 标注样本数量（在提琴图下方）
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        for i, (group, label) in enumerate(zip(multiple_axis, key_labels)):
            sample_count = len(group)
            ax.set_ylim(bottom=y_min - 0.15 * y_range)
            ax.text(i, y_min - 0.05 * y_range, f'n={sample_count}', 
                    ha='center', va='top',  color='black')

    # 添加总标题和保存图像
    fig.suptitle(f'Violin Plots of {metric_} for Multiple Keys', fontsize=16)
    plt.tight_layout()  # 调整布局以容纳标题
    plt.savefig(f'2line/2_4_violin_subplot_{metric_}.png')
    plt.close()

def main():
    """
    遍历所有主键和度量指标，调用multi_violin_subplot函数生成提琴图。
    """
    keys = ['object', 'people', 'Camera', 'Content', 'Category', 'time','DBSCAN Peak',]# 'GTPeaks']
    metrics_ = ['Diff_1-1', 'Diff_1-2', 'Diff_1-3', 'Diff_1-4', 'Diff_1-5', 'Diff_1-6',
                'MGF-1-1', 'MGF-1-2', 'MGF-1-3', 'MGF-1-4', 'MGF-1-5', 'MGF-1-6']

    for metric_ in metrics_:
        multi_violin_subplot(keys, metric_)

if __name__ == '__main__':
    main()