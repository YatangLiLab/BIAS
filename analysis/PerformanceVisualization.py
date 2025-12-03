import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from adjustText import adjust_text  # 引入 adjustText

def read_Computation_Cost(path):
    """
    read the Computation Cost and performance of the methods
    """
    tgt_file = pd.read_csv(path)
    method = tgt_file['Method']
    CC = tgt_file['CC']
    DLM = tgt_file['DLM']
    Cost = tgt_file['CPUTime(s)']
    AUC_J  = tgt_file['AUC-J']
    cc_adjustx = tgt_file['cc_adjust_x']
    auc_adjustx = tgt_file['auc_adjust_x']
    cc_adjusty = tgt_file['cc_adjust_y']
    auc_adjusty = tgt_file['auc_adjust_y']
    return (method, CC, AUC_J, DLM, Cost,cc_adjustx,auc_adjustx,cc_adjusty,auc_adjusty)

def plot_method_performance(method, metric, DLM, Cost, metric_name, adjustx,adjusty):
    trad_method, DL_method = [], []
    assert len(method) == len(metric)== len(DLM)==len(Cost)
    for idx in range(len(method)):
        if DLM[idx]==1:
            DL_method.append((method[idx], metric[idx], Cost[idx],adjustx[idx],adjusty[idx]))
        else:
            trad_method.append((method[idx], metric[idx], Cost[idx],adjustx[idx],adjusty[idx]))
    trad_method.sort(key=lambda x: x[2])
    print('\n'.join([f'Trad: {metric}' for metric in trad_method]))
    DL_method.sort(key=lambda x: x[2])
    print('\n'.join([f'DL: {metric}' for metric in DL_method]))
    
    our_rst_dict = {"CC":0.32298526358915197, 'sim': 0.2212780317847266,
                    'nss': 1.6697392133198607, 'AUC-J': 0.8493504589203005,
                    'kldiv': 2.184719825044664, 'auc_shuffled_score': 0.5605199855734727}
    our_method = ('Our Method', 0.012, our_rst_dict[metric_name])

    our_rst_dict_bottom_up = {"CC":0.357726964, 'sim': 0.236874223,
                    'nss': 1.850554107, 'AUC-J': 0.869286522,
                    'kldiv': 2.043760466, 'auc_shuffled_score': 0.577393977}
    our_method_bottom_up = ('Our Method \n(Bottom-Up)', 0.012, our_rst_dict_bottom_up[metric_name])

    our_rst_no_GWTA = {'model': 'MGF-1-4-new', 'CC':  0.30725712933469596, 'sim': 0.18352146820999837,
                        'nss': 1.6259288722923588, 'AUC-J': 0.8278725727907127,
                          'kldiv': 2.2275838415602762, 'auc_shuffled_score': 0.5817989327305786}
    our_method_no_GWTA = ('Our Method \n(No Gaussian WTA)', 0.012, our_rst_no_GWTA[metric_name])


    trad_names = [x[0] for x in trad_method]
    trad_xs = np.array([x[2] for x in trad_method])
    trad_ys = np.array([x[1] for x in trad_method])
    trad_adjxs = np.array([x[3] for x in trad_method])
    trad_adjys = np.array([x[4] for x in trad_method])

    DL_names = [x[0] for x in DL_method]
    DL_xs = np.array([x[2] for x in DL_method])
    DL_ys = np.array([x[1] for x in DL_method] )
    DL_adjxs = np.array([x[3] for x in DL_method])
    DL_adjys = np.array([x[4] for x in DL_method])

    # 开始绘图
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=DL_xs, y=DL_ys, label='DL-based Methods', color='r')
    sns.scatterplot(x=trad_xs, y=trad_ys, label='Traditional Methods', color='b')
    sns.scatterplot(x=[our_method[1]], y=[our_method[2]], label=our_method[0], color='green', marker='*',s=150)
    sns.scatterplot(x=[our_method_bottom_up[1]], y=[our_method_bottom_up[2]], color='green', marker='*',s=150)
    sns.scatterplot(x=[our_method_no_GWTA[1]], y=[our_method_no_GWTA[2]], color='green', marker='*',s=150)

    # 添加所有文本标签
    texts = []
    for name, x, y,adjx,adjy  in zip(trad_names, trad_xs, trad_ys,trad_adjxs,trad_adjys):
        texts.append(plt.text(x*adjx, y+adjy, name, color='blue'))
    for name, x, y, adjx,adjy in zip(DL_names, DL_xs, DL_ys,DL_adjxs,DL_adjys):
        texts.append(plt.text(x*adjx, y+adjy, name, color='red'))
    # 添加我们的方法的标签
    texts.append(plt.text(our_method[1]*0.6, our_method[2]+0.005, our_method[0], color='green'))
    texts.append(plt.text(our_method_bottom_up[1]*0.6, our_method_bottom_up[2]+0.007, our_method_bottom_up[0], color='green'))
    texts.append(plt.text(our_method_no_GWTA[1]*0.6, our_method_no_GWTA[2]-(0.025 if metric_name == 'CC' else 0.02), our_method_no_GWTA[0], color='green'))

    plt.xlabel('Equivalent Computation Time on CPU/s',fontdict={'fontsize': 12})
    plt.ylabel(f'Performance ({metric_name})',fontdict={'fontsize': 12})
    plt.legend(loc='upper left')
    # plt.title('Comparison of Computation Cost and Performance')
    plt.xscale('log')
    # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
    plt.tight_layout()
    plt.show()


def print_mean_value(todo_keys=None, metric_=None,delta = 4):

    # 读取静态属性（每个 video 一行）
    df_static = pd.read_csv('DHF1k_attribute-all.csv')
    
    # 读取动态指标（每个 video 多行，对应不同 index）
    df_dynamic = pd.read_csv(f'sm1-4-600.csv')
    
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

    if todo_keys is None:
        todo_keys = list(key_set.keys())

    for idx, todo_key in enumerate(todo_keys):
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

        available_idxs = [i for i, group in enumerate(multiple_axis) if len(group) > 0]
        multiple_axis = [multiple_axis[i] for i in available_idxs]
        key_labels = [key_set[todo_key][i] for i in available_idxs]
        print(key_labels)
        print([len(group) for group in multiple_axis])
        print([np.mean(group) for group in multiple_axis])


if __name__ == '__main__':
    path = r'./methodsAndComputationCost-wous.csv'
    method, CC, AUC_J, DLM, Cost, cc_adjustx,auc_adjustx,cc_adjusty,auc_adjusty = read_Computation_Cost(path)
    #keys = ['object', 'people', 'Camera', 'Content', 'Category', 'time', 'DBSCAN Peak']
    #metrics_ = ['cc','sim','nss','auc_judd','kldiv','auc_shuffled_score']#,'ks']
    #for metric_ in metrics_:
    #    print_mean_value(todo_keys=keys, metric_=metric_)

    plot_method_performance(method, CC, DLM, Cost, 'CC',cc_adjustx,cc_adjusty)
    plot_method_performance(method, AUC_J, DLM, Cost, 'AUC-J',auc_adjustx,auc_adjusty)