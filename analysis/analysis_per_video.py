import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm

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

def multi_hist(todo_key=None,metric_=None):
    csv_item = pd.read_csv('processedYOLOstats.csv')
    attribute_dict = csv_item.to_dict()
    print(attribute_dict.keys())
    
    key_set = {'object': ['O0', 'O1', 'O2', 'O>/3'],\
               'people':['P0', 'P1', 'P2', 'P>/3'],\
                'Camera':['Camera stable', 'Camera slow', 'Camera fast'],\
               'Content':['Content stable', 'Content slow', 'Content fast'],\
                'Category':['Daily activity', 'Sport', 'Social activity', 'Artistic performance', 'Animal', 'Artifact', 'Landscape'],\
                'time':['day', ' night', ' indoor']}
    if todo_key == None:
        todo_key = input('which axis is your want: object/people/Camera/Content/Category/time\n')
    if metric_ == None:
        metric_ = input('which metric is your want: cc/nss/nf\n')
    metric_set = ['Entity','total_size','mean_size']# ['1-1cc','1-2cc','1-3cc','1-4cc','1-5cc','1-6cc','GTPeaks','1-4peaknum'] # ['cc', 'sim', 'nss', 'auc_judd', 'kldiv', 'aucs']
    if todo_key not in key_set.keys():
        print(f'wrong key {todo_key} not in {key_set.keys()}')
        return
    if metric_ not in metric_set:
        print(f'wrong metric {metric_} not in {metric_set}')
        return
    
    
    multiple_axis = [[] for _ in range(len(key_set[todo_key]))]
    # overall_axis = []

    for index in tqdm.tqdm(attribute_dict['video_name'].keys()):
        # print(attribute_dict['video_name'])
        item_flag = True
        for i, key in enumerate(key_set[todo_key]):
            if attribute_dict[key][index] == 1 or attribute_dict[key][index] == '1':
                # print('hit')
                multiple_axis[i].append(attribute_dict[metric_][index])
                item_flag = False
            # overall_axis.append(attribute_dict[metric_][index])
        # if item_flag:
        #     print(index)
        #     print(attribute_dict[key][index])
        #     for i, key in enumerate(key_set[todo_key]):
        #         print(attribute_dict[key][index])
        #         print(attribute_dict[key][index]=='1')
        #         print(attribute_dict[key][index]==1)
        #     raise Exception('wrong attribute')
    # plt.hist(multiple_axis, bins=20)
    
    
    plt.close()
    # plt.figure(figsize=(10, 6))
    size_fig = 3 if todo_key == 'Category' else 2
    _, axs = plt.subplots(size_fig,size_fig,figsize=(15, 15))
    min_value, max_value = min([min(i) for i in multiple_axis]), max([max(i) for i in multiple_axis])
    max_density = 0
    for data in multiple_axis:
        hist, bin_edges = np.histogram(data, bins=20, range=(min_value, max_value), density=True)
        max_density = max(max_density, max(hist))

    for i in range(len(multiple_axis)):
        axs[i%size_fig][i//size_fig].hist(multiple_axis[i], bins=20, label=f'{key_set[todo_key][i]},cnt={len(multiple_axis[i])}',density=True,alpha = 0.4,range=(min_value,max_value))
        axs[i%size_fig][i//size_fig].set_ylabel('Density')
        axs[i%size_fig][i//size_fig].set_xlabel(metric_)
        axs[i%size_fig][i//size_fig].axvline(np.mean(multiple_axis[i]), 0, 1, color='red', linestyle='--', label=f'Mean = {np.mean(multiple_axis[i]):.3f}')
        axs[i%size_fig][i//size_fig].legend()
        axs[i%size_fig][i//size_fig].set_ylim(0, max_density)

    # plt.ylabel('Density')
    # plt.xlabel(metric_)
    plt.suptitle(f'{metric_} distribution for {todo_key}')
    # plt.title(f'{metric_} distribution for {todo_key}')
    plt.savefig(f'video_imgs/{todo_key}_{metric_}.png')
    # plt.show()

def main():
    keys = ['object','people','Camera','Content','Category','time']
    metrics_ = ['Entity','total_size','mean_size']# ['1-1cc','1-2cc','1-3cc','1-4cc','1-5cc','1-6cc','GTPeaks','1-4peaknum']
    for key in keys:
        for metric_ in metrics_:
            multi_hist(key,metric_)

if __name__ == '__main__':
    main()