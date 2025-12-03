import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm


def multi_hist():
    csv_item = pd.read_csv('processed_different_cs.csv')
    attribute_dict = csv_item.to_dict()
    print(attribute_dict.keys())
    
    key_set = {'object': ['O0', 'O1', 'O2', 'O>/3'],\
               'people':['P0', 'P1', 'P2', 'P>/3'],\
                'Camera':['Camera stable', 'Camera slow', 'Camera fast'],\
               'Content':['Content stable', 'Content slow', 'Content fast'],\
                'Category':['Daily activity', 'Sport', 'Social activity', 'Artistic performance', 'Animal', 'Artifact', 'Landscape'],\
                'time':['day', ' night', ' indoor']}

    metric_set = ['1-1cc','1-2cc','1-3cc','1-4cc','1-5cc','1-6cc'] # ['cc', 'sim', 'nss', 'auc_judd', 'kldiv', 'aucs'] 
    for i in range(len(metric_set)):
        attribute_dict[f'rank{i+1}'] = {}
    multi_hist = [[] for _ in range(len(metric_set))]
    for index in tqdm.tqdm(attribute_dict['video_name'].keys()):
        # for each frame in any video would have an index.
        scores = [attribute_dict[metric_set[idx]][index] for idx in range(len(metric_set))]
        rank = np.argsort(scores)[::-1]
        rank = [r+1 for r in rank]
        for _i in range(len(metric_set)):
            attribute_dict[f'rank{_i+1}'][index] = rank[_i]
    # pd.DataFrame(attribute_dict).to_csv('demo.csv')
    fig, axs = plt.subplots(2,3,figsize=(15, 15))
    for i in range(6):
        print([i//3,i//2])
        axs[i//3,i%3].hist(attribute_dict[f'rank{i+1}'].values(),label=f'Rank{i+1}',density=True,alpha = 0.6,edgecolor = 'black',bins=np.arange(1, 8) - 0.5)
        axs[i//3,i%3].set_ylabel('Density')
        axs[i//3,i%3].set_title(f'Rank{i+1}')
        axs[i//3,i%3].set_xlabel('Best Delta for c = 1')
    plt.show()

    # plt.ylabel('Density')
    # plt.xlabel(metric_)
    # plt.suptitle(f'{metric_} distribution for {todo_key}')
    # plt.title(f'{metric_} distribution for {todo_key}')
    # plt.savefig(f'imgs/{todo_key}_{metric_}.png')
    # plt.show()

def main():
    multi_hist()

if __name__ == '__main__':
    main()