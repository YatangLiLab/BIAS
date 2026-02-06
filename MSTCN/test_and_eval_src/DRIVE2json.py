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

path = 'test_pred/DADA2KS_Full_SACAE_Final/eval/results.jsonl'
name_score_dict = {}
with jsonlines.open(path, "r") as f:
    for line in tqdm.tqdm(f,total = 279):
        name = line["name"]
        vid_name = name.split('/')[-1].replace('.mp4','')
        score = line["score"][0]
        name_score_dict[vid_name] = score
        assert name2anno_dict.get(vid_name)
        print(vid_name, name2anno_dict.get(vid_name))
print('all test available!')

results = []
for name in name2anno_dict.keys():
    prob = name_score_dict[name]
    sep = lambda x: 1 if x >= 0.5 else 0
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

target_path = 'test_pred/DRIVE_label'
file = "SM_predictions.jsonl"
def write_jsonl(f, data):
    with jsonlines.open(f, "w") as f:
        f.write_all(data)

write_jsonl(os.path.join(target_path,file),results)



