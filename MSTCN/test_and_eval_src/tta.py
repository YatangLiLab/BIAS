import os
import json
import numpy as np
import pickle


'''
mTTA Advance Time Results Summary:
============================================================
test_pred/itti/SM_predictions.jsonl: mean Traffic Time Anticipation = 0.3097 (samples: 279)
test_pred/salfom/SM_predictions.jsonl: mean Traffic Time Anticipation = 0.5355 (samples: 279)
test_pred/DSTA_label/SM_predictions.jsonl: mean Traffic Time Anticipation = 3.8867 (samples: 279)
test_pred/ustring_label/SM_predictions.jsonl: mean Traffic Time Anticipation = 5.4738 (samples: 279)
test_pred/DRIVE_label/SM_predictions.jsonl: mean Traffic Time Anticipation = 6.6810 (samples: 279)
test_pred/RGB_predictions.jsonl: mean Traffic Time Anticipation = 0.9803 (samples: 279)
test_pred/SM_predictions.jsonl: mean Traffic Time Anticipation = 1.2025 (samples: 279)
'''

fps = 5

cause_effect_iou_lst = ['test_pred/itti/SM_predictions.jsonl',
                        'test_pred/salfom/SM_predictions.jsonl',
                        'test_pred/DSTA_label/SM_predictions.jsonl',
                        'test_pred/ustring_label/SM_predictions.jsonl',
                        'test_pred/DRIVE_label/SM_predictions.jsonl',
                        'test_pred/RGB_predictions.jsonl',
                        'test_pred/SM_predictions.jsonl']

# annotation
RGB_annotation_path = 'data/annotation-Mar9th-25fps.pkl'
Sal_annotation_path = 'data/saliency_annotation.pkl'

RGB_split_num = (1355, 1355 + 290, 1355 + 290 + 290)  # 290 条
Sal_spit_num  = (1355, 1355 + 264, 1355 + 264 + 279)  # 279 条

def get_first_nonzero_index(sequence, mask=None, target_value=1):
    """
    获取序列中第一个等于target_value的位置
    """
    if mask is not None:
        len_valid = int(np.sum(mask))
        valid_seq = np.array(sequence)[:len_valid]
    else:
        valid_seq = np.array(sequence)
    
    nonzero_indices = np.where(valid_seq == target_value)[0]
    if len(nonzero_indices) > 0:
        return nonzero_indices[0]
    else:
        return -1

def calculate_advance_time_for_jsonl(jsonl_file_path):
    """
    计算指定JSONL文件的提前时间
    """
    # 根据文件类型选择对应的注释
    if 'RGB' in jsonl_file_path:
        RGB_anno = pickle.load(open(RGB_annotation_path, 'rb'))[RGB_split_num[1]:RGB_split_num[2]]
        anno = pickle.load(open(Sal_annotation_path, 'rb'))[Sal_spit_num[1]:Sal_spit_num[2]]
        RGB_line_idxs = [idx for idx, _anno in enumerate(RGB_anno) if _anno in anno]
        anno = RGB_anno
    else:
        anno = pickle.load(open(Sal_annotation_path, 'rb'))[Sal_spit_num[1]:Sal_spit_num[2]]
    
    advance_times = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            if 'RGB' in jsonl_file_path:
                if line_idx not in RGB_line_idxs:
                    continue
            data = json.loads(line.strip())
            
            predicted_seq = data['predicted']
            target_seq = data['target']
            mask = data['mask']
            
            # 获取预测和目标的第一个等于1的位置
            pred_first_idx = get_first_nonzero_index(predicted_seq, mask, target_value=1)
            target_first_idx = get_first_nonzero_index(target_seq, mask, target_value=1)
            # 计算提前帧数
            advance_frames = target_first_idx - pred_first_idx if pred_first_idx != -1 and target_first_idx != -1 else 0
            
            # 计算提前时间（秒）
            if len(jsonl_file_path.split('/')) == 2:
                curr_anno = anno[line_idx][0]
                sum_time = curr_anno[2] - curr_anno[1]
                fps = np.sum(mask)/sum_time
            else:
                fps = 5
            advance_time = advance_frames / fps
            
            advance_times.append(max(advance_time,0))
    
    print(f"  Total lines processed: {len(lines)}, Valid samples: {len(advance_times)}")
    return advance_times

def calculate_avg_advance_time_all_methods():
    """
    计算所有方法的平均提前时间
    """
    results = {}
    
    for file_path in cause_effect_iou_lst:
        print(f"Processing {file_path}...")
        
        try:
            advance_times = calculate_advance_time_for_jsonl(file_path)
            
            if advance_times:
                avg_advance_time = np.mean(advance_times)
                std_advance_time = np.std(advance_times)
                results[file_path] = {
                    'avg_advance_time': avg_advance_time,
                    'std_advance_time': std_advance_time,
                    'advance_time_list': advance_times,
                    'count': len(advance_times)
                }
                print(f"  Average Advance Time: {avg_advance_time:.4f} (std: {std_advance_time:.4f}, based on {len(advance_times)} samples)")
            else:
                results[file_path] = {
                    'avg_advance_time': 0.0,
                    'std_advance_time': 0.0,
                    'advance_time_list': [],
                    'count': 0
                }
                print(f"  No valid samples found")
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")
            results[file_path] = {
                'avg_advance_time': 0.0,
                'std_advance_time': 0.0,
                'advance_time_list': [],
                'count': 0
            }
    
    return results

# 计算所有方法的平均提前时间
print("Calculating average advance time for all methods...")
avg_advance_time_results = calculate_avg_advance_time_all_methods()

print("\nAverage Advance Time Results Summary:")
print("=" * 60)
for method, result in avg_advance_time_results.items():
    filename = method.split('/')[-1]
    print(f"{method}: mean Traffic Time Anticipation = {result['avg_advance_time']:.4f} (samples: {result['count']})")
