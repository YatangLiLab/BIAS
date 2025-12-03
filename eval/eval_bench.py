from os import path, listdir, mkdir
import random
from tqdm import tqdm, trange
import numpy as np
from glob import glob
import json
from multiprocessing import Pool, Manager
from itertools import chain
import pandas as pd
from utils import read_sm, padding, padding_fixation
import argparse
import warnings
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numba import jit
warnings.filterwarnings("error")



###metrics###

#@jit(nopython=True)
def nss(s_map, gt):
    x,y = np.where(gt > 0)
    s_map_norm = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    temp:list[np.float64] = []
    for i in zip(x,y):
        temp.append(s_map_norm[i[0], i[1]])
    nss_value = sum(temp) / len(temp)
    return nss_value

#@jit(nopython=True)
def similarity(s_map, gt):
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    similarity_value = np.sum(np.minimum(s_map, gt))
    return similarity_value

#@jit(nopython=True)
def cc(s_map, gt):
    a = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt))/(np.std(gt) + 1e-7)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum() + 1e-7)
    return r

#@jit(nopython=True)
def auc_judd(S, F):
    S_flattened = S.flatten()
    F_flattened = F.flatten()
    Sth = S_flattened[F_flattened > 0]
    Nfixations = len(Sth)
    Npixels = S_flattened.shape[0]
    #if Npixels == -1:
    #    raise RuntimeError(f"The number of pixels in the saliency map and fixation map are not equal?\n s_shape = {S.shape}, fixation_map_shape = {F.shape}")

    allthreshes = sorted(Sth, reverse=True)
    tp = np.zeros(Nfixations + 2)
    fp = np.zeros(Nfixations + 2)
    tp[0] = fp[0] = 0
    tp[-1] = fp[-1] = 1

    for i in np.arange(1, Nfixations + 1):
        aboveth = np.sum(S >= allthreshes[i - 1])
        tp[i] = i / Nfixations
        fp[i] = (aboveth - i) / (Npixels - Nfixations)

    auc_judd_value = np.trapz(tp, fp)
    
    return auc_judd_value


def auc_shuffled(saliency_map,fixation_map,fixation_folder:str):
    """
    Saliency map is the saliency map we created, 
    Fixation map is the binary fixation map, 
    Fix folder is the folder for fixation maps, we have to sample some images from it.
    """
    saliency_map = np.array(saliency_map, dtype=np.float32)
    fixation_map = np.array(fixation_map, dtype=np.float32)
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map)+1e-6)
    fixation_folder_path = fixation_folder[0:int(len(fixation_folder)-9)] + '\\maps'
    # print(fixation_folder_path)
    Sth = saliency_map[fixation_map > 0]
    # fixation_points = np.argwhere(fixation_map > 0)
    Nsplits = 10
    other_fixation_points = []
    for _ in range(Nsplits):
        # 随机选择一个fixation文件
        fixation_files = [f for f in listdir(fixation_folder_path) if f.endswith('.mat')]
        random_fixation_file = random.choice(fixation_files)
        fixation_data = load_fixation_data(path.join(fixation_folder_path, random_fixation_file))
        other_fixation_points.extend(np.argwhere(fixation_data > 0))
    
    # 计算AUC
    # 计算AUC
    auc = 0
    stepSize = 0.1
    thresholds = np.arange(1, 0 - stepSize, -stepSize)
    num_thresholds = len(thresholds)
    
    for _ in range(Nsplits):
        # 对于每个split，使用所有other_fixation_points
        rand_fixation_points = np.array(other_fixation_points)
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        
        tp[0], fp[0] = 0, 0
        tp[-1], fp[-1] = 1, 1
        
        for i, thresh in enumerate(thresholds):
            tp[i] = np.sum(Sth >= thresh) / len(Sth)
            # 计算FP时，需要确保rand_fixation_points是二维坐标点
            if len(rand_fixation_points.shape) == 1:
                rand_fixation_points = rand_fixation_points[:, np.newaxis]
            fp[i] = np.sum(saliency_map[rand_fixation_points[:, 0].astype(int), rand_fixation_points[:, 1].astype(int)] >= thresh) / len(rand_fixation_points)
        auc += np.trapz(tp, fp)
    
    auc /= Nsplits
    # auc = compute_sauc(Sth, saliency_map, other_fixation_points, Nsplits)
    return auc

def load_fixation_data(fixation_path):
    # load fixation data from matlab mat file
    data = loadmat(fixation_path)
    return data['I']




# @jit(nopython=True)
def kldiv(s_map, gt):
    s_map = s_map / (np.sum(s_map) * 1.0 + 1e-7)
    gt = gt / (np.sum(gt) * 1.0 + 1e-7)
    eps = 2.2204e-16
    res = np.sum(gt * np.log(eps + gt / (s_map + eps)))
    return res


######

def calculate_frame_metrics(frame):
    gt_fix = read_sm(frame['gt_fixations_path'])
    gt_120_sm = read_sm(frame['gt_saliency_path'])
    pred_sm = read_sm(frame['predictions_path'])
    # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    # ax1.imshow(gt_fix)
    # ax2.imshow(gt_120_sm)
    # ax3.imshow(pred_sm)
    # plt.show()
    return {
        'sim_score': similarity(pred_sm, gt_120_sm),
        'nss_score':  nss(pred_sm, gt_fix),
        'cc_score': cc(pred_sm, gt_120_sm),
        'auc_judd_score': auc_judd(pred_sm, gt_fix),
        'kldiv_score': kldiv(pred_sm, gt_120_sm),
        'auc_shuffled_score': auc_shuffled(pred_sm, gt_fix,frame['gt_fixations_path'])
    }


def calculate_metrics(video_name, temp_predictions_path, temp_gt_saliency_path, temp_gt_fixations_path,num_workers):
    predictions_path = glob(temp_predictions_path)[0]
    gt_saliency_path = glob(temp_gt_saliency_path)[0]
    gt_fixations_path = glob(temp_gt_fixations_path)[0]

    scores = []
    assert_func = lambda path: set([int(x.split('.')[0]) for x in listdir(path)])
    fix_path = listdir(gt_fixations_path)
    if "maps" in fix_path:  
        new_set = set(map(lambda x:int(x.split(".")[0]),fix_path[0:len(fix_path)-1]))
    assert new_set == assert_func(gt_saliency_path) == assert_func(predictions_path)
    # 要生成001这样的img，不能image001，吐了
    frames = list([
        {
            'gt_fixations_path': gt_fix, 
            'gt_saliency_path': gt_sal, 
            'predictions_path': pred
        } for gt_fix, gt_sal, pred in zip(
            [path.join(gt_fixations_path, x) for x in sorted(listdir(gt_fixations_path))], 
            [path.join(gt_saliency_path, x) for x in sorted(listdir(gt_saliency_path))], 
            [path.join(predictions_path, x) for x in sorted(listdir(predictions_path))],
    )])
    scores = []
    #for i in trange(len(frames)):
    #    scores.append(calculate_frame_metrics(frames[i]))
    with Pool(num_workers) as pool:
        scores = pool.map(calculate_frame_metrics, frames)
    
    conv_scores = {metric: [x[metric] for x in scores] for metric in scores[0].keys()}
    length = len(scores)

    return {'video_name' : video_name,
            'cc' : sum(conv_scores['cc_score'])/length,
            'sim' : sum(conv_scores['sim_score'])/length,
            'nss' : sum(conv_scores['nss_score'])/length,
            'auc_judd' : sum(conv_scores['auc_judd_score'])/length,
            'kldiv' : sum(conv_scores['kldiv_score'])/length,
            'auc_shuffled_score': sum(conv_scores['auc_shuffled_score'])/length}


def calculate_all_videos(video_names, num_workers):
    detail_result = []
    for video_name in tqdm(video_names):
        # full_video_name = f'{video_name}*'
        print(video_name)
        print(video_name.split("_")[0])
        full_video_name = video_name.split("_")[0]
        gt_video_name = f"{int(full_video_name):04d}"
        model_output = path.join(FROM_MODEL, video_name)
        gt_gaussians = path.join(GT, gt_video_name, 'maps')
        gt_fixations = path.join(GT, gt_video_name, 'fixation')
        #gt_gaussians = path.join(GT, full_video_name)
        #gt_fixations = path.join(GT, full_video_name)
        tmp_result = [calculate_metrics(video_name, model_output, gt_gaussians, gt_fixations, num_workers)]
        name = FROM_MODEL.split("\\")[-1]
        with open(f'{name}.json', mode='a') as output:
            output.write(json.dumps(tmp_result))
            output.write("\n")

        detail_result += tmp_result
    
    return detail_result


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compares multiple models by a variety of metrics using multithreaded data processing')
    parser.add_argument('--models_root', help='path to directory with models predictions', default='root\\VAL_results')
    parser.add_argument('--gt_root', dest='GT', help='path to directory with Ground Truth saliency maps and fixations', default='root\\video\\annotation')
    parser.add_argument('--dont_use_domain_adaptation', action='store_true', help='specifies not to use domain adaptation',default=True)
    parser.add_argument('--num_workers', type=int, help='number of used threads', default=8)
    parser.add_argument('--skip_incomplete_results_test', action='store_true', help='allowed the results of the tested model do not contain all the videos of the dataset')
    args = parser.parse_args()
    models_root = args.models_root
    GT = args.GT
    # use_robust_metric = not args.dont_use_domain_adaptation
    use_robust_metric = True
    num_workers = args.num_workers
    skip_inc_res = args.skip_incomplete_results_test
    skip_inc_res = True
    
    print(num_workers, 'worker(s)')
    models = listdir(models_root)
    models = ['3-4']
    models_num = len(models)
    models = sorted(models)
    print(models)
    continue_flag = False
    while not continue_flag:
        label = input("do you want to continue?[y/n]\n")
        if label == 'y':
            continue_flag = True
        elif label == 'n':
            exit()
        else:
            print('wrong input\n')
    
    for model_num, model_root in enumerate(models):
        print(f'testing {model_root} ({model_num + 1}/{models_num})')
        FROM_MODEL = path.join(models_root, model_root)
        def read_folder(folder_dir):
            return sorted([path.join(folder_dir, p) for p in listdir(folder_dir)])

        sm_listdir = listdir(FROM_MODEL)
        GT_listdir = listdir(GT)
        if len(sm_listdir) < len(GT_listdir):
            msg = f'There are results for only a few videos ({len(sm_listdir)}/{len(GT_listdir)})!'
            if skip_inc_res:
                print(f'Warning: {msg}')
            else:
                raise ValueError(msg)
        video_names = sorted(sm_listdir)

        #GT_listdir_filtered = [x for x in GT_listdir if str(int(x)) in sm_listdir]
        #sm_maps = list(map(lambda x: read_folder(path.join(FROM_MODEL, str(int(x)))), sorted(sm_listdir)))
        #gt_maps = list(map(lambda x: read_folder(path.join(GT, x, 'maps')), sorted(GT_listdir_filtered)))
            #gt_maps = list(map(lambda x: read_folder(path.join(GT, x)), sorted(GT_listdir_filtered)))

        detail_result = calculate_all_videos(video_names, num_workers)

        result_name = 'Result'
        if not path.exists(result_name):
            mkdir(result_name)
        json_root = path.join(result_name, model_root)
        detail_result = sorted(detail_result, key=lambda res: res['video_name'])
        with open(f'{json_root}.json', mode='w') as output:
            output.write(json.dumps(detail_result)+",")

        result = {'cc' : [], 'sim' : [], 'nss' : [], 'auc_judd' : [], 'kldiv' : [],'auc_shuffled_score':[]}
        for i in result.keys():
            for j in detail_result:
                result[i].append(j[i])

        model_res = {'model': model_root}
        [model_res.update({key: [np.mean(result[key])]}) for key in result.keys()]
        header = not path.exists(f'{result_name}.csv')
        pd.DataFrame.from_dict(model_res, orient='columns').to_csv(f'{result_name}.csv', mode='a', header=header, index=False)
        print(model_res)
