import numpy as np
import tqdm
import os
import pickle

itti_npy_path = 'data/Ittidataset/Ittifeature_npy'
salfom_npy_path = 'data/Salfomdataset/Salfomfeature_npy'
original_data_path = 'data/RGBdataset/name_feature_dict.json'
saliency_npy_path = 'data/Saldataset/SMfeature_npy'

RGB_annotation_path = 'data/annotation-Mar9th-25fps.pkl'
Sal_annotation_path = 'data/saliency_annotation.pkl'

RGB_split_num = (1355, 1355 + 290, 1355 + 290 + 290)  # 290 条
Sal_spit_num  = (1355, 1355 + 264, 1355 + 264 + 279)  # 279 条


Sal_anno = pickle.load(open(Sal_annotation_path, 'rb'))# [0:Sal_spit_num[1]]
sal_names = set([f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in Sal_anno])

print(f'Sal 视频数: {len(sal_names)}')

# each element is (('v_Mfru8T-bHEE', 3.12, 26.16), ('Control Loss', 11.128526999999998, 13.180663, 2), ('Collision w/ Road Obstacle', 13.180663, 15.272419, 20))

print('only select sal as dict key, for we only have them as videos')
full_vids = pickle.load(open(Sal_annotation_path, 'rb'))
video_names = [f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in full_vids]
for idx, name in enumerate(tqdm.tqdm(video_names)):
    itti_saliency_map = np.load(os.path.join(itti_npy_path, f'{idx}.npy'))
    sal_saliency_map = np.load(os.path.join(salfom_npy_path, f'{idx}.npy'))
    saliency_map = np.load(os.path.join(saliency_npy_path, f'{idx}.npy'))
    time = Sal_anno[idx][0][2] - Sal_anno[idx][0][1]
    used_frames = round(time * 5)
    assert itti_saliency_map.shape[0] == sal_saliency_map.shape[0], f'{name} {time} {used_frames}, itti: {itti_saliency_map.shape[0]}, sal: {sal_saliency_map.shape[0]}'
    assert abs(itti_saliency_map.shape[0]-used_frames) <= 1, f'{name} {time}  {used_frames}, itti: {itti_saliency_map.shape[0]}, sal: {sal_saliency_map.shape[0]}'
    assert abs(itti_saliency_map.shape[0]-round(0.81*saliency_map.shape[0])) <= 5, f'{name} {time} {used_frames}, itti: {itti_saliency_map.shape[0]}, saliency map*0.81: {round(0.81*saliency_map.shape[0])}'
