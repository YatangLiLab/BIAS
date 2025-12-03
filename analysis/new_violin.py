import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def trad_violin_subplot(todo_keys=None, metric_=None,delta = 4):
    """
    使用两个 CSV 文件绘制提琴图：
    - 静态属性（如 object、people）来自 DHF1k_attribute-all.csv
    - 动态属性（如 GTPeaks、DBSCAN Peak）和 metric_ 来自 processed_different_cs.csv
    """

    # 读取静态属性（每个 video 一行）
    df_static = pd.read_csv('DHF1k_attribute-all.csv')
    
    # 读取动态指标（每个 video 多行，对应不同 index）
    df_dynamic = pd.read_csv(f'MGF-1-{delta}-full-7-0.csv')
    
    # 统一 'video' 列为字符串类型以避免合并错误
    df_static['video'] = df_static['video'].astype(str)
    df_dynamic['video'] = df_dynamic['video'].astype(str)
    
    # 合并两个数据集：通过 'video' 字段关联
    df_merged = pd.merge(df_dynamic, df_static, on='video', how='left')

    # 定义主键集合与显示标签
    key_set = {
        'object': ['O0', 'O1', 'O2', 'O>/3'],
        'people': ['P0', 'P1', 'P2', 'P>/3'],
        'Camera': ['Camera stable', 'Camera slow', 'Camera fast'],
        'Content': ['Content stable', 'Content slow', 'Content fast'],
        'time': [' night', 'day', ' indoor'],
        'Category': ['Daily activity', 'Sport', 'Social activity', 'Artistic performance', 'Animal', 'Artifact', 'Landscape'],
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
    }

    Display_keyset = {
        'object': ['O=0', 'O=1', 'O=2', 'O>/3'],
        'people': ['P=0', 'P=1', 'P=2', 'P>/3'],
        'Camera': ['stable', 'slow', 'fast'],
        'Content': ['stable', 'slow', 'fast'],
        'time': ['night', 'day', 'indoor'],
        'Category': ['Daily', 'Sport', 'Social', 'Artistic', 'Animal', 'Artifact', 'Landscape'],
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10'],
    }

    Display_keyname = {
        'object': '#Objects',
        'people': '#People',
        'Camera': 'Camera Speed',
        'Content': 'Content Speed',
        'time': 'Video Time',
        'Category': 'Video Category',
        'GTPeaks': 'Number of GMM Clusters',
        'DBSCAN Peak': 'Number of DBSCAN Clusters',
    }

    Display_Metric_name = {
        'cc':'Correlation Coefficient',
        'sim':'Similarity',
        'nss':'Normalized Scanpath Saliency',
        'auc_judd':'AUC Judd',
        'kldiv':'Kullback-Leibler Divergence',
        'auc_shuffled_score':'shuffled AUC',
        'ks':'Kolmogorov-Smirnov test'
    }

    # 筛选需要分析的主键
    if todo_keys is None:
        todo_keys = list(key_set.keys())

    # 创建画布
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

        # 初始化分组数据
        multiple_axis = [[] for _ in range(len(key_set[todo_key]))]

        # 按 key 分组提取数据
        for i, key in enumerate(key_set[todo_key]):
            if todo_key == 'GTPeaks':
                mask = df_merged['GTPeaks'].astype(int) == int(key)
            elif todo_key == 'DBSCAN Peak':
                if key == '>10':
                    mask = df_merged['DBSCAN Peak'].astype(int) > 10
                else:
                    mask = df_merged['DBSCAN Peak'].astype(int) == int(key)
            elif todo_key == 'ks':
                if key == '0.5':
                    mask = abs(df_merged['ks'].astype(float)) - 0.5 < 1e-3
                else:
                    mask = df_merged['ks'].astype(int) == int(key)
            else:
                mask = df_merged[key] == 1  # 静态属性为 1 表示属于该分组

            # 提取 metric_ 值
            values = df_merged.loc[mask, metric_].dropna().tolist()
            multiple_axis[i] = values

        # 过滤空组
        available_idxs = [i for i, group in enumerate(multiple_axis) if len(group) > 0]
        multiple_axis = [multiple_axis[i] for i in available_idxs]
        key_labels = [key_set[todo_key][i] for i in available_idxs]

        # 构建绘图数据框
        data, labels = [], []
        for group, label in zip(multiple_axis, key_labels):
            data.extend(group)
            labels.extend([label] * len(group))
        df_violin = pd.DataFrame({'value': data, 'group': labels})

        # 绘制提琴图
        if todo_key == 'Category':
            # 按中位数排序
            my_order = df_violin.groupby('group')['value'].median().sort_values(ascending=False).index
            sns.violinplot(data=df_violin, x='group', y='value', color='red', alpha=0.6,
                          inner='box', cut=0, bw_method=0.3, linewidth=2, ax=ax, order=my_order)
            # 更新显示标签顺序
            new_order = [key_set[todo_key].index(val) for val in my_order]
            Display_keyset[todo_key] = [Display_keyset[todo_key][new_o] for new_o in new_order]
        else:
            sns.violinplot(data=df_violin, x='group', y='value', color='red', alpha=0.6,
                          inner='box', cut=0, bw_method=0.3, linewidth=2, ax=ax)

        # 设置标题和标签
        ax.set_title(Display_keyname[todo_key])
        ax.set_xlabel('')
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(Display_keyset[todo_key])
        if idx % 3 == 0:
            ax.set_ylabel(Display_Metric_name[metric_])
        else:
            ax.set_ylabel('')

        # 标注样本数量
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        for i, (group, label) in enumerate(zip(multiple_axis, key_labels)):
            sample_count = len(group)
            ax.set_ylim(bottom=y_min - 0.15 * y_range)
            ax.text(i, y_min - 0.05 * y_range, f'n={sample_count}', 
                    ha='center', va='top', color='black')

    # 保存图像
    fig.suptitle(f'Violin Plots of c = 1 d = {delta} for Multiple Keys', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'stat_imgs/ViolinsDBSCAN1-{delta}/violin_subplot_{metric_}.png')
    plt.close()
def multi_violin_subplot(todo_keys=None, metric_=None,delta = 4):
    """
    使用两个 CSV 文件绘制提琴图：
    - 静态属性（如 object、people）来自 DHF1k_attribute-all.csv
    - 动态属性（如 GTPeaks、DBSCAN Peak）和 metric_ 来自 processed_different_cs.csv
    """

    # 读取静态属性（每个 video 一行）
    df_static = pd.read_csv('DHF1k_attribute-all.csv')
    
    # 读取动态指标（每个 video 多行，对应不同 index）
    df_dynamic = pd.read_csv(f'MGF-1-{delta}-full-7-0.csv')
    
    # 统一 'video' 列为字符串类型以避免合并错误
    df_static['video'] = df_static['video'].astype(str)
    df_dynamic['video'] = df_dynamic['video'].astype(str)
    
    # 合并两个数据集：通过 'video' 字段关联
    df_merged = pd.merge(df_dynamic, df_static, on='video', how='left')

    # 定义主键集合与显示标签
    key_set = {
        'object': ['O0', 'O1', 'O2', 'O>/3'],
        'people': ['P0', 'P1', 'P2', 'P>/3'],
        'Camera': ['Camera stable', 'Camera slow', 'Camera fast'],
        'ks': ['1','0','0.5'],
        'Content': ['Content stable', 'Content slow', 'Content fast'],
        'time': [' night', 'day', ' indoor'],
        'Category': ['Daily activity', 'Sport', 'Social activity', 'Artistic performance', 'Animal', 'Artifact', 'Landscape'],
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
    }

    Display_keyset = {
        'object': ['O=0', 'O=1', 'O=2', 'O>/3'],
        'people': ['P=0', 'P=1', 'P=2', 'P>/3'],
        'Camera': ['stable', 'slow', 'fast'],
        'Content': ['stable', 'slow', 'fast'],
        'time': ['night', 'day', 'indoor'],
        'Category': ['Daily', 'Sport', 'Social', 'Artistic', 'Animal', 'Artifact', 'Landscape'],
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10'],
        'ks': ['Same','Not','Blank']
    }

    Display_keyname = {
        'ks':'Kolmogorov-Smirnov',
        'object': '#Objects',
        'people': '#People',
        'Camera': 'Camera Speed',
        'Content': 'Content Speed',
        'time': 'Capture Time',
        'Category': 'Video Category',
        'GTPeaks': 'Number of GMM Clusters',
        'DBSCAN Peak': '#Clusters',
    }

    Display_Metric_name = {
        'cc':'Correlation Coefficient',
        'sim':'Similarity',
        'nss':'Normalized Scanpath Saliency',
        'auc_judd':'AUC Judd',
        'kldiv':'Kullback-Leibler Divergence',
        'auc_shuffled_score':'shuffled AUC',
        'ks':'Kolmogorov-Smirnov test'
    }

    # 筛选需要分析的主键
    if todo_keys is None:
        todo_keys = list(key_set.keys())

    # 创建画布
    fig = plt.figure(figsize=(15, 6))

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

        # 初始化分组数据
        multiple_axis = [[] for _ in range(len(key_set[todo_key]))]

        # 按 key 分组提取数据
        for i, key in enumerate(key_set[todo_key]):
            if todo_key == 'GTPeaks':
                mask = df_merged['GTPeaks'].astype(int) == int(key)
            elif todo_key == 'DBSCAN Peak':
                if key == '>10':
                    mask = df_merged['DBSCAN Peak'].astype(int) > 10
                else:
                    mask = df_merged['DBSCAN Peak'].astype(int) == int(key)
            elif todo_key == 'ks':
                if key == '0.5':
                    mask = abs(df_merged['ks'].astype(float) - 0.5) < 1e-3
                else:
                    mask = df_merged['ks'].astype(int) == int(key)
            else:
                mask = df_merged[key] == 1  # 静态属性为 1 表示属于该分组

            # 提取 metric_ 值
            values = df_merged.loc[mask, metric_].dropna().tolist()
            if metric_ == 'kldiv':
                values = [min(5, v) for v in values] # let biggest as 5
            multiple_axis[i] = values

        # 过滤空组
        available_idxs = [i for i, group in enumerate(multiple_axis) if len(group) > 0]
        multiple_axis = [multiple_axis[i] for i in available_idxs]
        key_labels = [key_set[todo_key][i] for i in available_idxs]

        # 构建绘图数据框
        data, labels = [], []
        for group, label in zip(multiple_axis, key_labels):
            data.extend(group)
            labels.extend([label] * len(group))
        df_violin = pd.DataFrame({'value': data, 'group': labels})

        # 绘制提琴图
        if todo_key == 'Category':
            # 按中位数排序
            my_order = df_violin.groupby('group')['value'].median().sort_values(ascending=False).index
            sns.violinplot(data=df_violin, x='group', y='value', color='red', alpha=0.6,
                          inner='box', cut=0, bw_method=0.3, linewidth=2, ax=ax, order=my_order)
            # 更新显示标签顺序
            new_order = [key_set[todo_key].index(val) for val in my_order]
            Display_keyset[todo_key] = [Display_keyset[todo_key][new_o] for new_o in new_order]
            ax.set_xticks(size=15)
        else:
            sns.violinplot(data=df_violin, x='group', y='value', color='red', alpha=0.6,
                          inner='box', cut=0, bw_method=0.3, linewidth=2, ax=ax)

        # 设置标题和标签
        ax.set_title(Display_keyname[todo_key],fontdict={'fontsize': 15})
        ax.set_xlabel('', fontsize=15)
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(Display_keyset[todo_key])
        ax.tick_params(axis='both',labelsize=12)
        if todo_key == 'time' or todo_key == 'object':
            ax.set_ylabel(Display_Metric_name[metric_], fontsize=15)
        else:
            ax.set_ylabel('')

        # 标注样本数量
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        for i, (group, label) in enumerate(zip(multiple_axis, key_labels)):
            sample_count = len(group)
            ax.set_ylim(bottom=y_min - 0.15 * y_range)
            ax.text(i, y_min - 0.05 * y_range, f'n={sample_count}', 
                    ha='center', va='top', color='black',fontdict={'fontsize': 10})

    # 保存图像
    # fig.suptitle(f'Violin Plots of c = 1 d = {delta} for Multiple Keys', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'2line/2_4_violin_subplot_{metric_}_8.png')
    plt.close()

def multi_bar_plot(todo_keys=None, metric_='ks',delta=4):
    '''
    plot barplot for multiple keys, plot the kolmogorov-smirnov test.
    till now we still use the 0-1 code, should we use probs in the future?
    '''
    # 读取静态属性（每个 video 一行）
    df_static = pd.read_csv('DHF1k_attribute-all.csv')
    
    # 读取动态指标（每个 video 多行，对应不同 index）
    df_dynamic = pd.read_csv(f'sm1-4-600-2.csv')#(f'MGF-1-{delta}-full-7-0.csv')
    
    # 统一 'video' 列为字符串类型以避免合并错误
    df_static['video'] = df_static['video'].astype(str)
    df_dynamic['video'] = df_dynamic['video'].astype(str)
    
    # 合并两个数据集：通过 'video' 字段关联
    df_merged = pd.merge(df_dynamic, df_static, on='video', how='left')

    # 定义主键集合与显示标签
    key_set = {
        'object': ['O0', 'O1', 'O2', 'O>/3'],
        'people': ['P0', 'P1', 'P2', 'P>/3'],
        'Camera': ['Camera stable', 'Camera slow', 'Camera fast'],
        'Content': ['Content stable', 'Content slow', 'Content fast'],
        'time': [' night', 'day', ' indoor'],
        'Category': ['Daily activity', 'Sport', 'Social activity', 'Artistic performance', 'Animal', 'Artifact', 'Landscape'],
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
    }

    Display_keyset = {
        'object': ['O=0', 'O=1', 'O=2', 'O>/3'],
        'people': ['P=0', 'P=1', 'P=2', 'P>/3'],
        'Camera': ['stable', 'slow', 'fast'],
        'Content': ['stable', 'slow', 'fast'],
        'time': ['night', 'day', 'indoor'],
        'Category': ['Daily', 'Sport', 'Social', 'Artistic', 'Animal', 'Artifact', 'Landscape'],
        'GTPeaks': ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
        'DBSCAN Peak': ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
    }

    Display_Metric_name = {
        'cc':'Correlation Coefficient',
        'sim':'Similarity',
        'nss':'Normalized Scanpath Saliency',
        'auc_judd':'AUC Judd',
        'kldiv':'Kullback-Leibler Divergence',
        'auc_shuffled_score':'shuffled AUC',
        'ks':'Kolmogorov-Smirnov test',
        'p_val':'p-value'
    }

    Display_keyname = {
        'ks':'Kolmogorov-Smirnov',
        
        'object': '#Objects',
        'people': '#People',
        'Camera': 'Camera Speed',
        'Content': 'Content Speed',
        'time': 'Capture Time',
        'Category': 'Video Category',
        'GTPeaks': 'Number of GMM Clusters',
        'DBSCAN Peak': '#Clusters',
    }

    # 筛选需要分析的主键
    if todo_keys is None:
        todo_keys = list(key_set.keys())

    # 创建画布
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

        # 初始化分组数据
        multiple_axis = [[] for _ in range(len(key_set[todo_key]))]

        # 按 key 分组提取数据
        for i, key in enumerate(key_set[todo_key]):
            if todo_key == 'GTPeaks':
                mask = df_merged['GTPeaks'].astype(int) == int(key)
            elif todo_key == 'DBSCAN Peak':
                if key == '>10':
                    mask = df_merged['DBSCAN Peak'].astype(int) > 10
                else:
                    mask = df_merged['DBSCAN Peak'].astype(int) == int(key)
            else:
                mask = df_merged[key] == 1  # 静态属性为 1 表示属于该分组

            # 提取 metric_ 值
            values = df_merged.loc[mask, metric_].dropna().tolist()
            multiple_axis[i] = values

        # 过滤空组
        available_idxs = [i for i, group in enumerate(multiple_axis) if len(group) > 0]
        multiple_axis = [multiple_axis[i] for i in available_idxs]
        key_labels = [key_set[todo_key][i] for i in available_idxs]

        # 构建绘图数据框
        data, labels = [], []
        for group, label in zip(multiple_axis, key_labels):
            data.extend(group)
            labels.extend([label] * len(group))
        df_violin = pd.DataFrame({'value': data, 'group': labels})

        # 绘制提琴图
        if todo_key == 'Category':
            # 按中位数排序
            my_order = df_violin.groupby('group')['value'].mean().sort_values(ascending=False).index
            sns.violinplot(data=df_violin, x='group', y='value', color='red', alpha=0.6,
                        linewidth=2, ax=ax, order=my_order,inner='quartile')
            ax.set_xticks(size=15)
            # 更新显示标签顺序
            new_order = [key_set[todo_key].index(val) for val in my_order]
            Display_keyset[todo_key] = [Display_keyset[todo_key][new_o] for new_o in new_order]
        else:
            sns.violinplot(data=df_violin, x='group', y='value', color='red', alpha=0.6,
                        linewidth=2, ax=ax,inner='quartile')

        # 设置标题和标签
        ax.set_title(Display_keyname[todo_key])
        ax.set_xlabel('')
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(Display_keyset[todo_key])
        if idx % 3 == 0:
            ax.set_ylabel(Display_Metric_name[metric_])
        else:
            ax.set_ylabel('')

        # 标注样本数量
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        for i, (group, label) in enumerate(zip(multiple_axis, key_labels)):
            sample_count = len(group)
            ax.set_ylim(bottom=y_min - 0.15 * y_range)
            ax.text(i, y_min - 0.05 * y_range, f'n={sample_count}', 
                    ha='center', va='top', color='black')

    # 保存图像
    # fig.suptitle(f'Bar Plots of c = 1 d = {delta} for Multiple Keys', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'2line/2_4_violin_subplot_{metric_}.png')
    plt.close()
def main():
    keys = ['object', 'people', 'Camera', 'Content', 'Category', 'time', 'DBSCAN Peak']
    metrics_ = ['cc','sim','nss','auc_judd','kldiv','auc_shuffled_score','ks']#['Diff_1-1', 'Diff_1-2', 'Diff_1-3', 'Diff_1-4', 'Diff_1-5', 'Diff_1-6',
               # 'MGF-1-1', 'MGF-1-2', 'MGF-1-3', 'MGF-1-4', 'MGF-1-5', 'MGF-1-6']
    
    # for metric_ in metrics_:
    #     multi_violin_subplot(keys + ['ks'], metric_,1)
    #     multi_violin_subplot(keys + ['ks'], metric_,4)
    #     trad_violin_subplot(keys, metric_,1)
    #     trad_violin_subplot(keys, metric_,4)
    # multi_bar_plot(keys,'ks',1)
    multi_violin_subplot(keys,'auc_judd',4)
    multi_violin_subplot(keys,'cc',4)

if __name__ == '__main__':
    main()