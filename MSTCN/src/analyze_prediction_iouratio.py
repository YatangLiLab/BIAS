import json
import pickle
import numpy as np


def _count_iou(pred_label:np.ndarray, _cls:int, cls_gt:np.ndarray)->np.ndarray:
    """ We adopt a slightly different Implementation of IOU from Causal-Effect Traffic Accident Dataset here
, due to the difference of the ground truth format.

    params:
    pred_label: np.ndarray, shape: [B, T], B for batch size, T for time steps = 208;
    pred_label is the computed max_label, which is the class index with the highest probability for each time step.

    _cls: int, class index, 1 for cause and 2 for effect;

    cls_gt: np.ndarray, shape: [B, T], B for batch size, T for time steps = 208;
    cls_gt is the ground truth label, which is 1 for cause and 2 for effect.
    """
    pred_eq_cls = pred_label == _cls
    cls_gt_eq_cls = cls_gt == _cls
    inter = np.sum(np.logical_and(pred_eq_cls, cls_gt_eq_cls), axis=0)
    print(np.logical_and(pred_eq_cls, cls_gt_eq_cls))
    union = np.sum(np.logical_or(pred_eq_cls,cls_gt_eq_cls), axis=0)
    iou = inter / (union + 1e-8)
    return iou

def compute_exact_iou(output:np.ndarray, cls_gt:np.ndarray, temporal_mask:np.ndarray, predtype:str='both'):# -> tuple[np.ndarray, np.ndarray]: 
    """
    output: [C, T], prediction logits
    cls_gt: [T], 0 for background, 1 for foreground
    temporal_mask: [T], 1 for valid, 0 for invalid
    """
    #new_output = np.zeros((3,len(output)))
    #for i in range(3):
    #    idxs = output == i
    #    new_output[i,idxs] = 1
    # pred_label = np.argmax(output, axis=1)
    valid_label = output 
    assert valid_label.shape == cls_gt.shape
    if predtype == 'both':
        return _count_iou(valid_label, 1, cls_gt), _count_iou(valid_label, 2, cls_gt)
    elif predtype == 'cause':
        return _count_iou(valid_label, 1, cls_gt)
    elif predtype == 'effect':
        return _count_iou(valid_label, 2, cls_gt)

def compute_temporalIoU(iou_set) -> np.ndarray: # list[numpy.ndarray]
    """
    analyze a series of IOU, compute the amount of each class prediction iou over some threshold
    params:
    iou_set: list of np.ndarray, each tensor is the iou of a class over all samples
    return:
    cnt: np.ndarray, shape: [9], the amount of each class prediction iou over some threshold,
    the threshold is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    cnt[0] is the amount of iou over 0.1, cnt[1] is the amount of iou over 0.2, and so on.
    """
    cnt = np.zeros(9)
    for bi in range(0,len(iou_set)):
        for thr in range(1,10):
            if iou_set[bi] > thr/10:
                cnt[thr-1] += 1
    cnt /= len(iou_set)
    return cnt

# squeeze 2 stage
# model\model\v2-2-128\2025-08-11-12-23-37\tb_logs2025-08-11-12-23-37\events.out.tfevents.1754886217.localhost.localdomain.1555063.0 → Squeeze2Stage
# squeeze 0 stage
# model\v2-1-128\2025-08-09-18-16-48\tb_logs2025-08-09-18-16-48\events.out.tfevents.1754734608.localhost.localdomain.3479651.0 → Squeeze0Stage
# squeeze 1 stage 32
# model\model\v2-2-32\2025-08-09-22-05-52\tb_logs2025-08-09-22-05-52\events.out.tfevents.1754748352.localhost.localdomain.3973973.0 → Squeeze1Stage
# squeeze 1 stage 128:
# our default result.
# squeeze 1 stage 256
# model\model\v2-2-256\2025-08-12-11-13-02\tb_logs2025-08-12-11-13-02\events.out.tfevents.1754968382.localhost.localdomain.2678242.0 → Squeeze1Stage



RGB_annotation_path = 'data/annotation-Mar9th-25fps.pkl'
Sal_annotation_path = 'data/saliency_annotation.pkl'

RGB_split_num = (1355, 1355 + 290, 1355 + 290 + 290)  # 290 条
Sal_spit_num  = (1355, 1355 + 264, 1355 + 264 + 279)  # 279 条


RGB_anno = pickle.load(open(RGB_annotation_path, 'rb'))[RGB_split_num[1]:RGB_split_num[2]]
video_names = [f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in RGB_anno]
Sal_anno = pickle.load(open(Sal_annotation_path, 'rb'))[Sal_spit_num[1]:Sal_spit_num[2]]
sal_names = set([f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in Sal_anno])

RGB_anno = [anno for anno in RGB_anno if f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' in sal_names]
for anno_RGB, anno_Sal in zip(RGB_anno, Sal_anno):
    assert anno_RGB == anno_Sal

self_saliency_dict_path = 'labels.json'
self_saliency_dict = json.load(open(self_saliency_dict_path))
dict_version = {json_dict['video']:json_dict for json_dict in self_saliency_dict}
self_saliency_dict_names = [element['video'] for element in self_saliency_dict]

acquire_labels = []
for idx, video_name in enumerate(video_names):
    name = video_name[2:]+'.mp4'
    if name in self_saliency_dict_names:
        acquire_labels.append(idx)

pred_jsonline_RGB_path = 'test_pred/RGB_predictions.jsonl'
pred_jsonline_SM_path = 'test_pred/seueeze2stage128/SM_predictions.jsonl'

with open(pred_jsonline_RGB_path, 'r', encoding='utf8') as f:
    pred_jsonline_RGB = f.readlines()
pred_jsonline_RGB = [line for idx, line in enumerate(pred_jsonline_RGB) if idx in acquire_labels]
with open(pred_jsonline_SM_path, 'r', encoding='utf8') as f:
    pred_jsonline_SM = f.readlines()

pred_and_tgt = []
assert len(pred_jsonline_RGB) == len(pred_jsonline_SM)
for idx in range(len(pred_jsonline_RGB)):
    RGB_dict = json.loads(pred_jsonline_RGB[idx])
    len_RGB = int(np.sum(RGB_dict['mask']))
    SM_dict = json.loads(pred_jsonline_SM[idx])
    len_SM = int(np.sum(SM_dict['mask']))
    pred_and_tgt.append([np.array(RGB_dict['predicted'][:len_RGB]), np.array(SM_dict['predicted'][:len_SM]),\
                         np.array(RGB_dict['target'][:len_RGB])   , np.array(SM_dict['target'][:len_SM])\
                        , np.arange(len_RGB+1)/len_RGB, np.arange(len_SM+1)/len_SM])


RGB_before_t0 = [0] * len(pred_and_tgt)
SM_before_t0 = [0] * len(pred_and_tgt)
time_diff = []

RGB_cause_ious = []
RGB_effect_ious = []
SM_cause_ious = []
SM_effect_ious = []

for idx, pred_and_tgt_element in enumerate(pred_and_tgt):
    time_length = RGB_anno[idx][0][2] - RGB_anno[idx][0][1]
    len_RGB = len(np.array(pred_and_tgt_element[0]))
    len_SM = len(np.array(pred_and_tgt_element[1]))
    RGB_Cause_iou_, RGB_Effect_iou_ = compute_exact_iou(pred_and_tgt_element[0], pred_and_tgt_element[2], np.ones(len_RGB))
    SM_Cause_iou_, SM_Effect_iou_ = compute_exact_iou(pred_and_tgt_element[1], pred_and_tgt_element[3], np.ones(len_SM))
    RGB_cause_ious.append(RGB_Cause_iou_)
    RGB_effect_ious.append(RGB_Effect_iou_)
    SM_cause_ious.append(SM_Cause_iou_)
    SM_effect_ious.append(SM_Effect_iou_)


# print('RGB iou: ')
# print(RGB_ious)
# print('SM iou: ')
# print(SM_ious)
# print('temporial RGB cause')
# print(compute_temporalIoU(RGB_cause_ious))
print('temporial SM cause')
print(compute_temporalIoU(SM_cause_ious))
# print('temporial RGB cause')
# print(compute_temporalIoU(RGB_effect_ious))
print('temporial SM effect')
print(compute_temporalIoU(SM_effect_ious))

"""
temporial RGB cause: 
[0.1,       0.2,       0.3,       0.4,       0.5,       0.6,        0.7,       0.8,       0.9,      ]
[0.53763441 0.46953405 0.41577061 0.34767025 0.20071685 0.13620072  0.0609319  0.01792115 0.        ]
temporial SM cause: 
[0.1,       0.2,       0.3,       0.4,       0.5,       0.6,        0.7,       0.8,       0.9,      ]
[0.5125448  0.41935484 0.32258065 0.22222222 0.14336918 0.08243728  0.05376344 0.00716846 0.        ]
temporial RGB cause: 
[0.1,       0.2,       0.3,       0.4,       0.5,       0.6,        0.7,       0.8,       0.9,      ]
[0.63799283 0.55555556 0.46236559 0.36917563 0.27598566 0.18637993  0.12544803 0.06451613 0.01075269]
temporial SM effect: 
[0.1,       0.2,       0.3,       0.4,       0.5,       0.6,        0.7,       0.8,       0.9,      ]
[0.79569892 0.70250896 0.60573477 0.47670251 0.34767025 0.23655914  0.13620072 0.05734767 0.01433692]
"""

'''
# Squeeze0
temporial SM cause
[0.20430108 0.11469534 0.08602151 0.05734767 0.02867384 0.01075269
 0.00716846 0.         0.        ]
temporial SM effect
[0.2437276  0.17204301 0.12903226 0.08602151 0.04659498 0.03942652
 0.02150538 0.00358423 0.        ]

Squeeze 1 stage 32:
  temporial SM cause
[0.01433692 0.00716846 0.         0.         0.         0.
 0.         0.         0.        ]
temporial SM effect
[0.03584229 0.01792115 0.01075269 0.00358423 0.         0.
 0.         0.         0.        ]
Squeeze 1 stage 64
temporial SM cause
[0.45519713 0.3655914  0.3046595  0.21863799 0.12544803 0.07526882
 0.02508961 0.00358423 0.        ]
temporial SM effect
[0.71326165 0.64157706 0.55913978 0.46594982 0.35125448 0.2078853
 0.09677419 0.02508961 0.00358423]

  Squeeze 1 stage 256:

temporial SM cause
[0.45519713 0.35842294 0.28673835 0.21505376 0.14695341 0.07526882
 0.03942652 0.01433692 0.00358423]
temporial SM effect
[0.74193548 0.65949821 0.56272401 0.48028674 0.34050179 0.21863799
 0.09677419 0.04659498 0.02150538]


  Squeeze 2 stage 128:
  temporial SM cause
[0.03942652 0.01075269 0.         0.         0.         0.
 0.         0.         0.        ]
temporial SM effect
[0.00716846 0.00358423 0.00358423 0.         0.         0.
 0.         0.         0.        ]
'''


