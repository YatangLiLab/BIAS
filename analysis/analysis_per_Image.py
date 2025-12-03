import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import stats  # 用于统计检验
import seaborn as sns    # 用于更美观的箱线图

def create_new_file():

    csv_item = pd.read_csv('DHF1k_attribute-all.csv')
    attribute_dict = csv_item.to_dict()
    new_string = []
    with open('processed_per_image_non_opt.json') as f:
        for line in f.readlines():
            video_index = line[11:15]
            new_line = line.strip()
            attributes = [str(key) + ',' + str(attribute_dict[key][int(video_index)-1]) for key in attribute_dict.keys()]
            new_line = new_line + ',' + ','.join(attributes) + '\n'
            new_string.append(new_line)
    with open('processed_file.json', 'w') as f:
        f.writelines(new_string)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

def multi_hist_with_stats(todo_key=None, metric_=None):
    plt.close()
    csv_item = pd.read_csv('processed_different_cs.csv')
    attribute_dict = csv_item.to_dict()
    
    key_set = {
        'object': ['O0', 'O1', 'O2', 'O>/3'],
        'people': ['P0', 'P1', 'P2', 'P>/3'],
        'Camera': ['Camera stable', 'Camera slow', 'Camera fast'],
        'Content': ['Content stable', 'Content slow', 'Content fast'],
        'Category': ['Daily activity', 'Sport', 'Social activity', 'Artistic performance', 'Animal', 'Artifact', 'Landscape'],
        'time': ['day', ' night', ' indoor'],
        'GTPeaks': ['2','3','4','5','6','7','8','9','10','11','12']
    }
    
    # 数据分组逻辑保持不变
    multiple_axis = [[] for _ in range(len(key_set[todo_key]))]
    for index in attribute_dict['video_name'].keys():
        for i, key in enumerate(key_set[todo_key]):
            if todo_key == 'GTPeaks':
                if int(attribute_dict['GTPeaks'][index]) == int(key):  # 匹配数字
                    multiple_axis[i].append(attribute_dict[metric_][index])

            elif attribute_dict[key][index] == 1 or attribute_dict[key][index] == '1':
                multiple_axis[i].append(attribute_dict[metric_][index])
    
    available_idxs = [i for i, group in enumerate(multiple_axis) if len(group) > 0]
    if len(available_idxs) < 2:  # 如果没有至少两个可用的分组，则不绘制图
        print(f'Not enough data for {todo_key}')
        return
    multiple_axis = [multiple_axis[i] for i in available_idxs]
    key_set[todo_key] = [key_set[todo_key][i] for i in available_idxs]
    key_labels = key_set[todo_key]
    
    # ====== 新增KS检验模块 ======
    def run_ks_tests(data_groups):
        n_groups = len(data_groups)
        p_values = np.ones((n_groups, n_groups))  # 初始化为1
        significant = np.zeros((n_groups, n_groups), dtype=bool)
        
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                if len(data_groups[i]) == 0 or len(data_groups[j]) == 0:
                    continue  # 跳过空组
                stat, p = stats.ks_2samp(data_groups[i], data_groups[j])
                p_values[i,j] = p
                p_values[j,i] = p  # 对称矩阵
                
                # 使用Bonferroni校正（调整阈值）
                corrected_alpha = 0.05 / (n_groups*(n_groups-1)/2)
                significant[i,j] = p < corrected_alpha
                significant[j,i] = significant[i,j]
        
        return p_values, significant

    p_matrix, sig_matrix = run_ks_tests(multiple_axis)
    # ==========================

    # 可视化部分重构
    plt.figure(figsize=(20, 6))
    
    # 重构后的直方图（左）
    ax1 = plt.subplot(1, 3, 1)
    # 创建长格式数据
    hist_data = []
    for i, data in enumerate(multiple_axis):
        for value in data:
            hist_data.append({
                'value': value,
                todo_key: key_set[todo_key][i]
            })
    df_hist = pd.DataFrame(hist_data)
    
    # 使用seaborn绘制分列直方图
    sns.histplot(
        data=df_hist,
        x='value',
        hue=todo_key,
        multiple='dodge',  # 分列显示
        bins=20,
        palette='Set2',
        edgecolor='w',
        linewidth=0.5,
        stat='count',
        ax=ax1,
        legend=True
    )
    ax1.set_title(f'Histogram of {metric_} by {todo_key}')
    ax1.set_xlabel(metric_)
    # ax1.legend(
    #     title=todo_key,
    #     bbox_to_anchor=(1.05, 1),
    #     loc='upper left'
    # )

    # 优化后的箱线图（中）
    ax2 = plt.subplot(1, 3, 2)
    violin_data = []
    sig_annotations = []  # 原有显著性标注
    effect_annotations = []  # 新增效应量标注
    
    # 针对Diff指标的特殊处理
    if metric_.startswith('Diff_1-'):
        for i, (group, label) in enumerate(zip(multiple_axis, key_labels)):
            # 执行单样本检验（原有逻辑保持不变）
            if len(group) >= 3:
                t_stat, p_val = stats.ttest_1samp(group, 0)
                sig = p_val < 0.05
                direction = "↑" if np.mean(group) > 0 else "↓"
                p_val *= len(multiple_axis)  # Bonferroni校正
            else:
                p_val = 1.0
                sig = False
                direction = ""
            sig_annotations.append((i, p_val, sig, direction))
            
            # 新增效应量计算
            if len(group) >= 2:
                mean_diff = np.mean(group)
                std_diff = np.std(group, ddof=1)
                cohen_d = mean_diff / std_diff
                ci = stats.t.interval(0.95, len(group)-1,
                                     loc=mean_diff,
                                     scale=std_diff/np.sqrt(len(group)))
            else:
                mean_diff = cohen_d = ci = np.nan
            effect_annotations.append((i, mean_diff, cohen_d, ci))
    else:
        # 非Diff指标的初始化
        sig_annotations = [(i, 1.0, False, "") for i in range(len(multiple_axis))]
        effect_annotations = [(i, np.nan, np.nan, (np.nan, np.nan)) for i in range(len(multiple_axis))]


    # 绘制小提琴图
    df_violin = pd.DataFrame({
        'value': [v for group in multiple_axis for v in group],
        todo_key: [label for label, group in zip(key_labels, multiple_axis) 
                   for _ in group]
    })
    
    sns.violinplot(
        data=df_violin,
        x=todo_key,
        y='value',
        hue=todo_key,
        palette='Set2',
        inner="quartile",
        cut=0,
        bw_method=0.3,
        legend=False,
        density_norm="area",
        ax=ax2
    )

    # sns.stripplot(
    #     data=df_violin,
    #     x=todo_key,
    #     y='value',
    #     color='black',
    #     size=3,
    #     alpha=0.4,
    #     ax=ax2
    # )
    # 计算标注位置
    x_coords = np.arange(len(key_labels))
    y_max = df_violin['value'].max()
    y_min = df_violin['value'].min()
    y_range = y_max - y_min
    y_pos_sig = y_max + y_range * 0.05
    y_pos_eff = y_pos_sig + y_range * 0.12
    counts = [len(group) for group in multiple_axis]

    ax2.set_ylim(bottom=y_min - y_range*0.1)
    for i, count in enumerate(counts):
        ax2.text(i, y_min - 0.05 * y_range, f'n={count}', ha='center', va='bottom', fontsize=10, color='black')
    # 添加显著性标注
    if metric_.startswith('Diff_1-'):
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
        
        #print([y_max, y_min, y_range, y_pos])
        #quit()
    
        for i, (x_pos, (_, p_val, sig, direction)) in enumerate(zip(x_coords, sig_annotations)):
            text = f"{direction}\np={p_val:.2f}" if p_val >= 0.001 else f"{direction}\np<0.001"
            color = ('red' if direction == '↑' else 'blue')if sig else 'gray'
        
            # 使用数据坐标系定位
            ax2.text(x_pos, y_pos_sig, text, 
                horizontalalignment='center',verticalalignment='center',
                color=color, fontsize=10,
                bbox=dict(facecolor='none', edgecolor='none', pad=2),
                transform=ax2.transData)  
            
        for i, (x_pos, (_, mean_diff, d, (ci_low, ci_high))) in enumerate(zip(x_coords, effect_annotations)):
            if np.isnan(mean_diff):
                continue
            eff_text = f"Δ={mean_diff:.2f}\nd={d:.2f}"
            ci_text = f"95% CI\n[{ci_low:.2f}, {ci_high:.2f}]"
            
            # 效应量主标注
            # ax2.text(x_pos, y_pos_eff, eff_text,
            #         ha='center', va='bottom',
            #         color='purple', fontsize=10,
            #         bbox=dict(facecolor='white', edgecolor='none', pad=2),
            #         transform=ax2.transData)
            
            # 置信区间标注
            ax2.text(x_pos, y_pos_eff, ci_text,
                    ha='center', va='top',
                    color='gray', fontsize=8,
                    transform=ax2.transData)
        ax2.set_ylim(bottom=y_min - y_range*0.1, top=y_pos_eff + y_range*0.1)
        # 调整y轴范围以容纳标注
    
        

    ax2.set_title(f'Distribution of {metric_}')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)

    # 优化后的KS检验热力图（右）
    ax3 = plt.subplot(1, 3, 3)
    # 设置固定的颜色范围
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))  # 隐藏上三角
    
    formatted_p = np.vectorize(lambda x: f"<0.001" if x < 0.001 else f"{x:.2f}")(p_matrix)
    
    sns.heatmap(p_matrix, 
                annot=formatted_p,  # 使用格式化后的文本
                fmt="",
                cmap=cmap,
                mask=mask,
                xticklabels=key_set[todo_key],
                yticklabels=key_set[todo_key],
                ax=ax3,
                vmin=0,    # 固定最小值
                vmax=0.1,  # 固定最大值
                cbar_kws={'label': 'p-value', 
                         'ticks': [0, 0.05, 0.1],
                         'format': '%.2f'})
    
    # 添加显著性标记
    for i in range(len(key_set[todo_key])):
        for j in range(i+1, len(key_set[todo_key])):
            if sig_matrix[i,j]:
                # 调整标注位置
                ax3.text(j+0.5, i+0.5, '*', 
                         ha='center', va='center', 
                         color='red', fontsize=14, fontweight='bold')

    ax3.set_title('KS Test p-values')
    
    plt.tight_layout()
    plt.savefig(f'stat_imgs/ks_combined_{todo_key}_{metric_}.png')
    plt.close()

    # 返回结果保持不变
    return {
        'metric': metric_,
        'group': todo_key,
        'anova_p': stats.f_oneway(*multiple_axis).pvalue if len(multiple_axis)>2 else None,
        'ttest_p': stats.ttest_ind(*multiple_axis).pvalue if len(multiple_axis)==2 else None,
        'ks_p_matrix': p_matrix,
        'significant_pairs': [(key_set[todo_key][i], key_set[todo_key][j]) 
                             for i in range(len(key_set[todo_key])) 
                             for j in range(i+1, len(key_set[todo_key])) 
                             if sig_matrix[i,j]]
    }

def main():
    keys = ['GTPeaks','object','people','Camera','Content','Category','time']
    metrics_ = ['GTPeaks','Diff_1-1','Diff_1-2','Diff_1-3','Diff_1-4','Diff_1-5','Diff_1-6','1-1cc','1-2cc','1-3cc','1-4cc','1-5cc','1-6cc','MGF-1-1','MGF-1-2','MGF-1-3','MGF-1-4','MGF-1-5','MGF-1-6']
    for key in keys:
        for metric_ in metrics_:
            multi_hist_with_stats(key, metric_)

if __name__ == '__main__':
    main()