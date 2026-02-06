import os
import tqdm

target_parent_folder = 'E:/Li Lab/MS-TCA/MyMSTCN/data/Ittidataset/itti_dataset/val'
for folder in tqdm.tqdm(os.listdir(target_parent_folder)):
    folder_path = os.path.join(target_parent_folder, folder)
    for _file in tqdm.tqdm(os.listdir(folder_path)):
        if _file.endswith('.png'):
            os.remove(os.path.join(folder_path, _file))
