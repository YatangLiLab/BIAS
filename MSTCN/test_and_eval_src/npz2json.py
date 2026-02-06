import numpy as np
import os
import jsonlines
import argparse
import pickle
import tqdm

Sal_spit_num  = (1355, 1355 + 264, 1355 + 264 + 279)  # 279 条

Sal_annotation_path = 'data/saliency_annotation.pkl'
Sal_anno = pickle.load(open(Sal_annotation_path, 'rb'))[Sal_spit_num[1]:Sal_spit_num[2]]
sal_names = [f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in Sal_anno]
# (('v_OLCTPfweyk8', 3.08, 14.96), ('Left Turn Across Path at Non-Signalized Junctions', 6.034801, 6.978481, 1), ('Collision w/ Vehicle', 6.978481, 8.48124, 19))
name2anno_dict = {f'{anno[0][0][2:]}_{int(anno[0][1])}_{int(anno[0][2])}':anno for anno in Sal_anno}

path = 'test_pred/DSTA_label/'
npzpath = os.path.join(path, 'npzs')
thres = 0.5

npzs = os.listdir(npzpath)
npzs = [npz for npz in npzs if npz.endswith('.npz')]
names = [npz.replace('_feature_result.npz','') for npz in npzs]
for name in tqdm.tqdm(names):
    print(name)
    assert name2anno_dict.get(name)
print('all test available!')

npzs = [os.path.join(npzpath,npz) for npz in npzs]
results = []
for name, npz in tqdm.tqdm(zip(names, npzs),total = 279): # test size
    prob = np.load(npz,allow_pickle=True)['score']
    sep = lambda x: 1 if x >= thres else 0
    predlabel = [sep(p) for p in prob]
    anno = name2anno_dict[name]
    start_time = anno[0][1]
    end_time = anno[0][2]
    traffic_start = anno[1][1]
    traffic_end = anno[2][2]
    video_seq_len = len(predlabel)
    labelling = lambda x: 1 if traffic_start<=(x/video_seq_len*(end_time-start_time))<=traffic_end else 0
    mask = [1.]*video_seq_len
    gt_label = [labelling(i) for i in range(video_seq_len)]

    results.append({'predicted':predlabel,'target':gt_label,'mask':mask})

file = "SM_predictions.jsonl"
def write_jsonl(f, data):
    with jsonlines.open(f, "w") as f:
        f.write_all(data)

write_jsonl(os.path.join(path,file),results)



