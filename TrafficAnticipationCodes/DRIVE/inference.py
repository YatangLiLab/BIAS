# modified from main_sac.py
# we want to use it to do inference for the model

import argparse, os
import torch
import numpy as np
import itertools
import datetime
import random
import yaml
from easydict import EasyDict
import time

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
from torchvision import transforms

from src.DADA2KS import DADA2KS, evalDADA2KS
from src.data_transform import ProcessImages, ProcessFixations
from tqdm import tqdm
import datetime

from src.enviroment import DashCamEnv
from RLlib.SAC.sac import SAC

import jsonlines
# from RLlib.SAC.replay_buffer import ReplayMemory, ReplayMemoryGPU
# from metrics.eval_tools import evaluation_accident, evaluation_fixation, evaluation_auc_scores, evaluation_accident_new, evaluate_earliness


def parse_configs():
    parser = argparse.ArgumentParser(description='PyTorch SAC implementation')
    # For training and testing
    parser.add_argument('--config', default="cfgs/sac_ae_test_cause.yml",
                        help='Configuration file for SAC algorithm.')
    parser.add_argument('--phase', default='test', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--gpu_id', type=int, default=0, metavar='N',
                        help='The ID number of GPU. Default: 0')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--baseline', default='random', choices=['random', 'all_pos', 'all_neg', 'none'],
                        help='setup baseline results for testing comparison')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                        help='number of epoches (default: 50)')
    parser.add_argument('--snapshot_interval', type=int, default=5, metavar='N',
                        help='The epoch interval of model snapshot (default: 5)')
    parser.add_argument('--test_epoch', type=int, default=-1, 
                        help='The snapshot id of trained model for testing.')
    parser.add_argument('--output', default='output/DADA2KS_Full_SACAE_Final',
                        help='Directory of the output. ')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)

    cfg.SAC.image_shape = cfg.ENV.image_shape
    cfg.SAC.input_shape = cfg.ENV.input_shape

    return cfg


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def setup_dataloader(cfg, num_workers=0, isTraining=True):
    transform_dict = {'image': transforms.Compose([ProcessImages(cfg.input_shape, mean=[0.218, 0.220, 0.209], std=[0.277, 0.280, 0.277])]),
                      'salmap': transforms.Compose([ProcessImages(cfg.output_shape)]), 
                      'fixpt': transforms.Compose([ProcessFixations(cfg.input_shape, cfg.image_shape)])}
    # testing dataset
    if not isTraining:
        test_data = evalDADA2KS(cfg.data_path, 'testing', interval=cfg.frame_interval, transforms=transform_dict, use_salmap=cfg.use_salmap)
        testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
        print("# test set: %d"%(len(test_data)))
        return testdata_loader
    

def infer_all(testdata_loader:DataLoader, env, agent):
    all_pred_scores, all_pred_fixations, = [], []
    for i, (video_data, salmap_data, coord_data, data_info) in enumerate(tqdm(testdata_loader)):  # (B, T, H, W, C)
        print(f"Testing video {i+1}/{len(testdata_loader)}, file: {testdata_loader.dataset.data_list[i].split('/')[-1]}, frame #: {video_data.size(1)} (fps={30/cfg.ENV.frame_interval:.2f}).")
        # set environment data
        state = env.set_data(video_data, coord_data, data_info)

        # init vars before each episode
        rnn_state = (torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device),
                        torch.zeros((cfg.ENV.batch_size, cfg.SAC.hidden_size), dtype=torch.float32).to(cfg.device))
        score_pred = np.zeros((cfg.ENV.batch_size, env.max_steps), dtype=np.float32)
        fixation_pred = np.zeros((cfg.ENV.batch_size, env.max_steps, 2), dtype=np.float32)
        # fixation_gt = np.zeros((cfg.ENV.batch_size, env.max_steps, 2), dtype=np.float32)
        i_steps = 0
        while i_steps < env.max_steps:
            # select action
            actions, rnn_state = agent.select_action(state, rnn_state, evaluate=True)
            # step
            state, reward, info = env.step(actions, isTraining=False)
            # gather actions
            score_pred[:, i_steps] = info['pred_score'].cpu().numpy().tolist()  # shape=(B,)
            fixation_pred[:, i_steps] = info['pred_fixation'].cpu().numpy().tolist()  # shape=(B, 2)
            # next_step = env.cur_step if i_steps != env.max_steps - 1 else env.cur_step - 1
            # fixation_gt[:, i_steps] = env.coord_data[:, next_step*env.step_size, :].cpu().numpy()
            # next step
            i_steps += 1

        # save results
        all_pred_scores.append(score_pred)  # (B, T)
        # all_gt_labels.append(env.clsID.cpu().numpy())  # (B,)
        all_pred_fixations.append(fixation_pred)  # (B, T, 2)
        # all_gt_fixations.append(fixation_gt)      # (B, T, 2)
        #all_toas.append(env.begin_accident.cpu().numpy())  # (B,)
        #all_vids.append(data_info[:,:4].numpy())
        #print(score_pred)
        #print(fixation_pred)
        #print(score_pred.shape, fixation_pred.shape)
    
    # all_pred_scores = np.concatenate(all_pred_scores)
    # all_gt_labels = np.concatenate(all_gt_labels)
    # all_pred_fixations = np.concatenate(all_pred_fixations)
    # ll_gt_fixations = np.concatenate(all_gt_fixations)
    # all_toas = np.concatenate(all_toas)
    # all_vids = np.concatenate(all_vids)
    return all_pred_scores, all_pred_fixations

def infer_batch(cfg):
    # initilize environment
    env = DashCamEnv(cfg.ENV, device=cfg.device)
    env.set_model(pretrained=True, weight_file=cfg.ENV.env_model)
    cfg.ENV.output_shape = env.output_shape
    # initialize dataset
    testdata_loader = setup_dataloader(cfg.ENV, 0, isTraining=False)
    # AgentENV
    agent = SAC(cfg.SAC, device=cfg.device)
    # load agent models (by default: the last epoch)
    ckpt_dir = os.path.join(cfg.output, 'checkpoints')
    agent.load_models(ckpt_dir, cfg)

    # prepare output directory
    output_dir = os.path.join(cfg.output, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_file = os.path.join(output_dir, 'results.npz')
    # if os.path.exists(result_file):
    #     save_dict = np.load(result_file, allow_pickle=True)
    #     all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = \
    #         save_dict['pred_scores'], save_dict['gt_labels'], save_dict['pred_fixations'], save_dict['gt_fixations'], save_dict['toas'], save_dict['vids']
    # else:
    # start to test 
    agent.set_status('eval')
    with torch.no_grad():
        all_pred_scores, all_pred_fixations = infer_all(testdata_loader, env, agent)
    # np.savez(result_file, pred_scores=all_pred_scores, pred_fixations=all_pred_fixations)
    result_jsonlines_score = os.path.join(output_dir,'results.jsonl')
    result = []
    for vid_name, score, fix in zip(testdata_loader.dataset.data_list,all_pred_scores,all_pred_fixations):
      result.append({'name':vid_name,'score':score.tolist(),'fix':fix.tolist()})
      print({'name':vid_name,'score':score,'fix':fix})
    with jsonlines.open(result_jsonlines_score,'w') as f:
      f.write_all(result)

if __name__ == "__main__":
    
    # parse input arguments
    cfg = parse_configs()
    set_deterministic(cfg.seed)
    infer_batch(cfg)