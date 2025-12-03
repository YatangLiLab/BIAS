import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm


def error_bar_Image():
    csv_item = pd.read_csv('processed_different_cs.csv')
    attribute_dict = csv_item.to_dict()
    print(attribute_dict.keys())

    metric_set = ['1-1cc','1-2cc','1-3cc','1-4cc','1-5cc','1-6cc'] # ['cc', 'sim', 'nss', 'auc_judd', 'kldiv', 'aucs'] 
    metric_set2 = ['MGF-1-1','MGF-1-2','MGF-1-3','MGF-1-4','MGF-1-5','MGF-1-6']
    score_values = [[] for idx in range(len(metric_set))]
    score_values2 = [[] for idx in range(len(metric_set2))]
    for index in range(len(metric_set)):
        # for each frame in any video would have an index.
        score_values[index] = [attribute_dict[metric_set[index]][idx] for idx in range(len(attribute_dict[metric_set[index]]))]
    mean_lst = [np.mean(score_values[idx]) for idx in range(len(metric_set))]
    var_lst = [np.std(score_values[idx]) for idx in range(len(metric_set))]
    
    x_lst = np.arange(1,len(mean_lst)+1,1)
    plt.errorbar(x_lst+0.01,mean_lst,yerr=var_lst,fmt='-o',c='blue',label = 'Itti style cc')
    # x_DoG = 4.01
    # lst_DoG = [attribute_dict['1-4DoG'][idx] for idx in range(len(attribute_dict['1-4DoG']))]
    # mean_DoG = np.mean(lst_DoG)
    # var_DoG = np.std(lst_DoG)
    for index in range(len(metric_set)):
        # for each frame in any video would have an index.
        score_values2[index] = [attribute_dict[metric_set2[index]][idx] for idx in range(len(attribute_dict[metric_set2[index]]))]
    mean_lst2 = [np.mean(score_values2[idx]) for idx in range(len(metric_set2))]
    var_lst2 = [np.std(score_values2[idx]) for idx in range(len(metric_set2))]
    # mgf_x = 3.98
    plt.errorbar(x_lst - 0.01,mean_lst2,yerr=var_lst2,fmt='-o',c='green',label = 'Multiple Gaussian Fitting cc')
    for idx in range(len(metric_set)):
        print(f'{metric_set[idx]} mean: {mean_lst[idx]} var: {var_lst[idx]}')
        print(f'{metric_set2[idx]} mean: {mean_lst2[idx]} var: {var_lst2[idx]}')
    # plt.errorbar(x_DoG,mean_DoG,yerr=var_DoG,fmt='-o',c='red',label = '1-4DoG cc')
    plt.title('Mean and Variance of CC for Final Saliency Map')
    plt.legend()
    plt.show()

def violin_plot_Image(mode = 'cc'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # 读取数据
    csv_item = pd.read_csv('processed_different_cs.csv' if mode == 'cc' else 'processed_different_cs-auc.csv')
    attribute_dict = csv_item.to_dict()
    if mode == 'cc':
        metric_set = ['1-1cc', '1-2cc', '1-3cc', '1-4cc', '1-5cc', '1-6cc']
        metric_set2 = ['MGF-1-1', 'MGF-1-2', 'MGF-1-3', 'MGF-1-4', 'MGF-1-5', 'MGF-1-6']
    elif mode == 'auc':
        metric_set = ['1-1AUC-J', '1-2AUC-J', '1-3AUC-J', '1-4AUC-J', '1-5AUC-J', '1-6AUC-J']
        metric_set2 = ['MGF1AUC', 'MGF1AUC', 'MGF3AUC', 'MGF4AUC', 'MGF5AUC', 'MGF6AUC']
    
    # 准备数据用于提琴图
    data = []
    for idx, metric in enumerate(metric_set):
        values = [attribute_dict[metric][i] for i in range(len(attribute_dict[metric]))]
        data.extend([{'Method': 'C-S', 'Metric': f'1-{idx+1}', 'Value': v} for v in values])
        
    for idx, metric in enumerate(metric_set2):
        values = [attribute_dict[metric][i] for i in range(len(attribute_dict[metric]))]
        data.extend([{'Method': 'G-WTA', 'Metric': f'1-{idx+1}', 'Value': v} for v in values])
    
    df = pd.DataFrame(data)
    
    # 创建提琴图
    plt.figure(figsize=(8, 4))
    colors = sns.color_palette("RdBu_r", 2)
    ax = sns.violinplot(x='Metric', y='Value', hue='Method', data=df, 
                        split=True, inner="quart", gap=.1, palette=colors, # 'box', 'quart', 'stick', 'point'
                        saturation = 1., cut = 0,linecolor='black',linewidth=1.5)
    
    # # 标注均值和方差
    # for i, metric in enumerate(df['Metric'].unique()):
    #     for j, method in enumerate(df['Method'].unique()):
    #         subset = df[(df['Metric'] == metric) & (df['Method'] == method)]
    #         mean_val = np.mean(subset['Value'])
    #         var_val = np.var(subset['Value'], ddof=1)  # 使用样本方差
            
    #         # 计算标注位置（左右分开）
    #         x_pos = i
    #         if j == 0:  # 左侧
    #             x_offset = -0.2
    #         else:  # 右侧
    #             x_offset = 0.2
            
    #         # 在图中标注均值和方差
    #         ax.text(x_pos + x_offset, mean_val + 0.05, 
    #                 f"Mean: {mean_val:.2f}\nVar: {var_val:.2f}", 
    #                 fontsize=8, ha='center', va='bottom', color='black')

    # 设置标题和标签
    # plt.title('Distribution of Correlation Coefficient for Different Metrics', fontsize=14)
    plt.xlabel('Center-Delta pair', fontsize=20)
    plt.ylabel('Correlation Coefficient' if mode == 'cc' else 'Area Under Curve-Judd', fontsize=20)
    plt.legend(fontsize=15) # title = 'Method'
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
def main():
    # error_bar_Image()
    violin_plot_Image(mode = 'cc')
    violin_plot_Image(mode = 'auc')

if __name__ == '__main__':
    main()