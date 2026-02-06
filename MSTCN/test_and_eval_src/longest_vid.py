import numpy as np
import tqdm
import os
import pickle

itti_npy_path = 'data/Ittidataset/Ittifeature_npy'

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
max_length = -1
video_names = [f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in full_vids]
for idx, name in enumerate(tqdm.tqdm(video_names)):
    itti_saliency_map = np.load(os.path.join(itti_npy_path, f'{idx}.npy'))
    max_length = max(itti_saliency_map.shape[0], max_length)
print(max_length) # 321
    