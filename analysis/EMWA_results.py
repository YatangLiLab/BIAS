import matplotlib.pyplot as plt
# import re
import json
import numpy as np
import tqdm
from scipy.stats import norm
import seaborn as sns

def read_json_file(path,_metric='cc'):
    '''
read and load file from self designed json file.\n
structure :\n
[{"video_name": "0699", "cc": 0.41810125596995723, "sim": 0.23508152813229016, "nss": 2.2591696901164533, "auc_judd": 0.902335277213981, "kldiv": 1.8462455705137018, "auc_shuffled_score": 0.5535298895251908}]\n
in each line.\n
a simple way to load it might be like this:\n
data = exec(f.readline())
datas += data
    '''
    name_metrics = [[],[]]
    # cc_pattern = re.compile(r"\[\{\"video_name\": \"(\d+)\", \"cc\": ([\d.]+)")
    with open(path, 'r') as f:
        for line in tqdm.tqdm(f,total=100):
            json_pre = line.strip().replace('[', '').replace(']', '')
            jsons = json_pre.split('},')
            jsons = [_json + '}' for _json in jsons[:-1]]
            print(jsons[-1])
            # print(jsons[0])
            datas = [json.loads(jsons[i]) for i in tqdm.trange(len(jsons))]
            for data in datas:
                name_metrics[0].append(f"{data['video_name']}{data['index']}")
                name_metrics[1].append(data[_metric])
    return name_metrics

def main():
    paths= ['sm_1_4.json','single_image.json'] #'0305.json','03015.json','0505.json','1010.json','0203.json',
    data_collection = []
    for path in paths:
        data_collection.append(read_json_file(path))
    #    plt.plot(data_collection[-1][0], data_collection[-1][1],label=path.split('/')[-1].split('.')[0]+f'mean = {np.mean(np.array(data_collection[-1][1]))}')     
    #plt.title('different methods and their results(CC as an example)')
    #plt.legend()
    #plt.show()

    # data_collection = []
    # for path in paths:
    #     data_collection.append(read_json_file(path,'auc_judd'))
    #     plt.plot(data_collection[-1][0], data_collection[-1][1],label=path.split('/')[-1].split('.')[0]+f'mean = {np.mean(np.array(data_collection[-1][1]))}')     
    # plt.title('different methods and their results(CC as an example)')
    # plt.legend()
    # plt.show()

    # for index, path in enumerate(paths):
    #     plt.plot(data_collection[index][0] ,\
    #              np.array(data_collection[index][1]) - np.array(data_collection[0][1]) ,label=path.split('/')[-1].split('.')[0]+f'mean = {np.mean(np.array(data_collection[index][1]))}')     
    # plt.title('different methods and their results(CC as an example), Baseline 0303 = non_optal_val')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 6))  # 设置图像大小
    # for index, path in enumerate(paths):
    #     cc_values = data_collection[index][1]
    #     mean, std = norm.fit(cc_values)  # 拟合高斯分布
    #     x = np.linspace(min(cc_values), max(cc_values), 100)
    #     y = norm.pdf(x, mean, std)  # 计算概率密度
    #     plt.plot(x, y, label=f'{path.split("/")[-1].split(".")[0]} (mean={mean:.4f}, std={std:.4f})')

    # plt.title('Gaussian Distribution of CC Values')
    # plt.xlabel('CC Value')
    # plt.ylabel('Probability Density')
    # plt.legend()
    # plt.show()
        # === 新增部分：Violin Plot 对比 ===
    labels = ['With\nEWMA', 'Without\nEWMA']
    data_to_plot = [data_collection[0][1], data_collection[1][1]]

    plt.figure(figsize=(4.5, 4.5))
    violin_parts = plt.violinplot(data_to_plot, showmeans=False, showmedians=False, showextrema=False,)

    # 设置统一的颜色
    # color = 'skyblue'
    colors = sns.color_palette("RdBu_r", 2)
    for idx, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[1-idx])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        pc.set_linewidth(2)

    # 添加箱型图元素（中位数、IQR）
    quartile1, medians, quartile3 = np.percentile(data_to_plot, [25, 50, 75], axis=1)
    whiskers = np.array([
        np.min(data_to_plot[0]), np.max(data_to_plot[0]),
        np.min(data_to_plot[1]), np.max(data_to_plot[1])
    ]).reshape(2, 2)

    inds = range(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskers[:, 0], whiskers[:, 1], color='k', linestyle='-', lw=1)

    # 标注样本数量
    # sample_sizes = [len(data_collection[0][1]), len(data_collection[1][1])]
    x_ticks = np.arange(1, len(labels)+1)
    # for i, size in enumerate(sample_sizes):
    #     plt.text(x_ticks[i], min(min(data_to_plot)) - 0.02, f'n={size}', ha='center', fontsize=10)

    plt.xticks(x_ticks, labels)
    plt.xticks(size=20)
    plt.ylabel('Correlation Coefficient',fontdict={'size': 20})
    plt.yticks(size=12,rotation=45)
    # plt.title('')
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    data_collection = []
    for path in paths:
        data_collection.append(read_json_file(path,'auc_judd'))
    #     plt.plot(data_collection[-1][0], data_collection[-1][1],label=path.split('/')[-1].split('.')[0]+f'mean = {np.mean(np.array(data_collection[-1][1]))}')     
    # plt.title('different methods and their results(CC as an example)')
    # plt.legend()
    # plt.show()

    # for index, path in enumerate(paths):
    #     plt.plot(data_collection[index][0] ,\
    #              np.array(data_collection[index][1]) - np.array(data_collection[0][1]) ,label=path.split('/')[-1].split('.')[0]+f'mean = {np.mean(np.array(data_collection[index][1]))}')     
    # plt.title('different methods and their results(CC as an example), Baseline 0303 = non_optal_val')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(10, 6))  # 设置图像大小
    # for index, path in enumerate(paths):
    #     cc_values = data_collection[index][1]
    #     mean, std = norm.fit(cc_values)  # 拟合高斯分布
    #     x = np.linspace(min(cc_values), max(cc_values), 100)
    #     y = norm.pdf(x, mean, std)  # 计算概率密度
    #     plt.plot(x, y, label=f'{path.split("/")[-1].split(".")[0]} (mean={mean:.4f}, std={std:.4f})')

    # plt.title('Gaussian Distribution of CC Values')
    # plt.xlabel('CC Value')
    # plt.ylabel('Probability Density')
    # plt.legend()
    # plt.show()
        # === 新增部分：Violin Plot 对比 ===
    labels = ['With\nEWMA', 'Without\nEWMA']
    data_to_plot = [data_collection[0][1], data_collection[1][1]]

    plt.figure(figsize=(4.5, 4.5))
    violin_parts = plt.violinplot(data_to_plot, showmeans=False, showmedians=False, showextrema=False)

    # 设置统一的颜色
    colors = sns.color_palette("RdBu_r", 2)
    for idx, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[1-idx])
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        pc.set_linewidth(2)

    # 添加箱型图元素（中位数、IQR）
    quartile1, medians, quartile3 = np.percentile(data_to_plot, [25, 50, 75], axis=1)
    whiskers = np.array([
        np.min(data_to_plot[0]), np.max(data_to_plot[0]),
        np.min(data_to_plot[1]), np.max(data_to_plot[1])
    ]).reshape(2, 2)

    inds = range(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskers[:, 0], whiskers[:, 1], color='k', linestyle='-', lw=1)

    # 标注样本数量
    # sample_sizes = [len(data_collection[0][1]), len(data_collection[1][1])]
    x_ticks = np.arange(1, len(labels)+1)
    # for i, size in enumerate(sample_sizes):
    #     plt.text(x_ticks[i], min(min(data_to_plot)) - 0.02, f'n={size}', ha='center', fontsize=10)

    plt.xticks(x_ticks, labels)
    plt.ylabel('Area Under Curve-Judd',fontdict={'size': 20})
    plt.yticks(size=12,rotation=45)
    # plt.title('')
    plt.xticks(size=20)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()