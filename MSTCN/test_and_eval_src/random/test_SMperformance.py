'''
Processing test_pred/DSTA_label/random:
  Binary Threshold Analysis: [0.92508961 0.61111111 0.30716846 0.14516129 0.05842294 0.02831541  0.01362007 0.00143369 0.00107527]

Processing test_pred/ustring_label/random:
  Binary Threshold Analysis: [0.49569892 0.38781362 0.27956989 0.17921147 0.10430108 0.05376344  0.03046595 0.00752688 0.00071685]

Processing test_pred/DRIVE_label/random:
  Binary Threshold Analysis: [0.93154122 0.70609319 0.37419355 0.2311828  0.109319   0.06487455 0.02508961 0.00358423 0.00071685]
'''

import os
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

def process_cause_effect_format(jsonl_file_path):
    """
    处理第一种格式的JSONL文件：同时包含cause和effect，predicted为0、1、2
    计算双类别IoU（cause和effect）以及总IoU
    
    Args:
        jsonl_file_path: JSONL文件路径
    
    Returns:
        dict: 包含各种IoU结果的字典
    """
    predicted_labels = []
    target_labels = []
    masks = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predicted_labels.append(data['predicted'])
            target_labels.append(data['target'])
            masks.append(data['mask'])
    
    # 处理可变长度序列，分别计算每个样本的IoU
    cause_ious = []
    effect_ious = []
    total_ious = []
    
    for pred, target, mask in zip(predicted_labels, target_labels, masks):
        # 确保所有序列长度一致
        pred_array = np.array(pred)
        target_array = np.array(target)
        mask_array = np.array(mask)
        
        # 应用mask
        masked_pred = pred_array * mask_array
        masked_target = target_array * mask_array
        
        # 分别计算每个样本的IoU
        # 计算cause IoU (class 1)
        pred_eq_cause = masked_pred == 1
        target_eq_cause = masked_target == 1
        cause_inter = np.sum(np.logical_and(pred_eq_cause, target_eq_cause))
        cause_union = np.sum(np.logical_or(pred_eq_cause, target_eq_cause))
        cause_iou = cause_inter / (cause_union + 1e-8)
        cause_ious.append(cause_iou)
        
        # 计算effect IoU (class 2)
        pred_eq_effect = masked_pred == 2
        target_eq_effect = masked_target == 2
        effect_inter = np.sum(np.logical_and(pred_eq_effect, target_eq_effect))
        effect_union = np.sum(np.logical_or(pred_eq_effect, target_eq_effect))
        effect_iou = effect_inter / (effect_union + 1e-8)
        effect_ious.append(effect_iou)
        
        # 计算总体IoU（合并cause和effect为一个类别）
        # 将预测值和目标值都转换为二分类（0 vs 非0）
        binary_pred = (masked_pred > 0).astype(int)
        binary_target = (masked_target > 0).astype(int)
        
        # 总体IoU计算
        total_inter = np.sum(np.logical_and(binary_pred, binary_target))
        total_union = np.sum(np.logical_or(binary_pred, binary_target))
        total_iou = total_inter / (total_union + 1e-8)
        total_ious.append(total_iou)
    
    # 转换为numpy数组
    cause_iou_array = np.array(cause_ious)
    effect_iou_array = np.array(effect_ious)
    total_iou_array = np.array(total_ious)
    
    return {
        'cause_iou': cause_iou_array,
        'effect_iou': effect_iou_array,
        'total_iou': total_iou_array,
        'cause_threshold_analysis': compute_temporalIoU(cause_iou_array),
        'effect_threshold_analysis': compute_temporalIoU(effect_iou_array),
        'total_threshold_analysis': compute_temporalIoU(total_iou_array)
    }


def process_binary_format(jsonl_file_path):
    """
    处理第二种格式的JSONL文件：仅0或1，表示有无交通事故
    计算二分类IoU以及总IoU
    
    Args:
        jsonl_file_path: JSONL文件路径
    
    Returns:
        dict: 包含各种IoU结果的字典
    """
    predicted_labels = []
    target_labels = []
    masks = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predicted_labels.append(data['predicted'])
            target_labels.append(data['target'])
            masks.append(data['mask'])
    
    # 处理可变长度序列，分别计算每个样本的IoU
    binary_ious = []
    
    for pred, target, mask in zip(predicted_labels, target_labels, masks):
        # 确保所有序列长度一致
        pred_array = np.array(pred)
        target_array = np.array(target)
        mask_array = np.array(mask)
        
        # 应用mask
        masked_pred = pred_array * mask_array
        masked_target = target_array * mask_array
        
        # 计算当前样本的IoU
        inter = np.sum(np.logical_and(masked_pred, masked_target))
        union = np.sum(np.logical_or(masked_pred, masked_target))
        sample_iou = inter / (union + 1e-8)
        
        binary_ious.append(sample_iou)
    
    # 返回所有样本的IoU数组
    binary_iou_array = np.array(binary_ious)
    
    return {
        'binary_iou': binary_iou_array,
        'threshold_analysis': compute_temporalIoU(binary_iou_array)
    }


def compute_overall_iou_for_cause_effect(jsonl_file_path):
    """
    为第一种格式（cause-effect）计算整体IoU（所有非背景类别的综合IoU）
    
    Args:
        jsonl_file_path: JSONL文件路径
    
    Returns:
        float: 整体IoU值
    """
    results = process_cause_effect_format(jsonl_file_path)
    # 取所有时间步的平均IoU作为整体IoU
    overall_cause_iou = np.mean(results['cause_iou'])
    overall_effect_iou = np.mean(results['effect_iou'])
    overall_total_iou = np.mean(results['total_iou'])
    
    return {
        'overall_cause_iou': overall_cause_iou,
        'overall_effect_iou': overall_effect_iou,
        'overall_total_iou': overall_total_iou
    }


def compute_overall_iou_for_binary(jsonl_file_path):
    """
    为第二种格式（binary）计算整体IoU
    
    Args:
        jsonl_file_path: JSONL文件路径
    
    Returns:
        float: 整体IoU值
    """
    results = process_binary_format(jsonl_file_path)
    # 取所有时间步的平均IoU作为整体IoU
    overall_binary_iou = np.mean(results['binary_iou'])
    
    return {
        'overall_binary_iou': overall_binary_iou
    }


general_traffic_iou_lst = [
    'test_pred/DSTA_label/random',
    'test_pred/ustring_label/random',
    'test_pred/DRIVE_label/random',
]

for random_folder in general_traffic_iou_lst:
    print(f"\nProcessing {random_folder}:")
    general_random_path = os.listdir(random_folder)
    assert len(general_random_path) == 10 , 'we expect 10 random examples.'
    average_ioudata = np.zeros(9)
    for file_path in general_random_path:
        file_path = os.path.join(random_folder, file_path)
        results = process_binary_format(file_path)
        overall_results = compute_overall_iou_for_binary(file_path)
        average_ioudata += np.array(results['threshold_analysis'])
    print(f"  Binary Threshold Analysis: {average_ioudata/10}")
