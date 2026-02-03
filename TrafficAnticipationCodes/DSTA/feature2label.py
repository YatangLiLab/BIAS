'''
    # run inference
    echo "Run accident inference..."
    source activate py37
    CUDA_VISIBLE_DEVICES=$GPUS python demo.py \
        --task inference \
        --feature_file $FEAT_FILE \
        --ckpt_file demo/final_model_ccd.pth \
        --gpu_id $GPUS
    conda deactivate
    echo "Saved in: $RESULT_FILE"
'''

# changed demo.py as feature2label.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os, sys
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.dim_feat = 4096

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

def init_feature_extractor(backbone='vgg16', device=torch.device('cuda')):
    feat_extractor = None
    if backbone == 'vgg16':
        feat_extractor = VGG16()
        feat_extractor = feat_extractor.to(device=device)
        feat_extractor.eval()
    else:
        raise NotImplementedError
    return feat_extractor


def bbox_sampling(bbox_result, nbox=19, imsize=None, topN=5):
    """
    imsize[0]: height
    imsize[1]: width
    """
    assert not isinstance(bbox_result, tuple)

    bboxes = np.vstack(bbox_result)  # n x 5

    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]

    labels = np.concatenate(labels)  # n

    ndet = bboxes.shape[0]

    # fix bbox
    new_boxes = []
    for box, label in zip(bboxes, labels):
        x1 = min(max(0, int(box[0])), imsize[1])
        y1 = min(max(0, int(box[1])), imsize[0])
        x2 = min(max(x1 + 1, int(box[2])), imsize[1])
        y2 = min(max(y1 + 1, int(box[3])), imsize[0])
        if (y2 - y1 + 1 > 2) and (x2 - x1 + 1 > 2):
            new_boxes.append([x1, y1, x2, y2, box[4], label])

    if len(new_boxes) == 0:  # no bboxes
        new_boxes.append([0, 0, imsize[1]-1, imsize[0]-1, 1.0, 0])
    new_boxes = np.array(new_boxes, dtype=int)

    # sampling
    n_candidate = min(topN, len(new_boxes))
    if len(new_boxes) <= nbox - n_candidate:
        indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=True)
        sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
    elif len(new_boxes) > nbox - n_candidate and len(new_boxes) <= nbox:
        indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=False)
        sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
    else:
        sampled_boxes = new_boxes[:nbox]

    return sampled_boxes


def bbox_to_imroi(transform, bboxes, image):
    imroi_data = []
    for bbox in bboxes:
        imroi = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        imroi = transform(Image.fromarray(imroi))  # (3, 224, 224), torch.Tensor
        imroi_data.append(imroi)
    imroi_data = torch.stack(imroi_data)
    return imroi_data

def extract_features(detector, feat_extractor, video_file, n_frames=50, n_boxes=19):
    assert os.path.join(video_file), video_file
    # prepare video reader and data transformer
    videoReader = mmcv.VideoReader(video_file)
    transform = transforms.Compose([
        # transforms.Resize(256),
        transforms.Resize(512),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    features = np.zeros((n_frames, n_boxes + 1, feat_extractor.dim_feat), dtype=np.float32)
    detections = np.zeros((n_frames, n_boxes, 6))  # (50 x 19 x 6)
    frame_prev = None
    for idx in range(n_frames):
        if idx >= len(videoReader):
            print("Copy frame from previous time step.")
            frame = frame_prev.copy()
        else:
            frame = videoReader.get_frame(idx)
        # run object detection inference
        bbox_result = inference_detector(detector, frame)



        # sampling a fixed number of bboxes
        bboxes = bbox_sampling(bbox_result, nbox=n_boxes, imsize=frame.shape[:2])



        detections[idx, :, :] = bboxes
        # prepare frame data
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            # bboxes to roi feature
            ims_roi = bbox_to_imroi(transform, bboxes, frame)
            ims_roi = ims_roi.float().to(device=device)
            feature_roi = feat_extractor(ims_roi)
            # extract image feature
            ims_frame = transform(Image.fromarray(frame))
            ims_frame = torch.unsqueeze(ims_frame, dim=0).float().to(device=device)
            feature_frame = feat_extractor(ims_frame)
        # obtain feature matrix
        features[idx, 0, :] = np.squeeze(feature_frame.cpu().numpy()) if feature_frame.is_cuda else np.squeeze(feature_frame.detach().numpy())
        features[idx, 1:, :] = np.squeeze(feature_roi.cpu().numpy()) if feature_roi.is_cuda else np.squeeze(feature_roi.detach().numpy())
        frame_prev = frame
    return detections, features


def init_accident_model(model_file, dim_feature=4096, hidden_dim=512, latent_dim=256, n_obj=19, n_frames=50, fps=10.0):
    # building model
    model = DSTA(dim_feature, hidden_dim, latent_dim,
        n_layers=1, n_obj=n_obj, n_frames=n_frames, fps=fps, with_saa=True)

    model = model.to(device=device)
    model.eval()
    # load check point
    model, _, _ = load_checkpoint(model, filename=model_file, isTraining=False)
    return model


def load_input_data(feature_file, device=torch.device('cuda')):
    # load feature file and return the transformed data
    data = np.load(feature_file)
    features = data['data']  # 50 x 20 x 4096
    labels = [0, 1]
    detections = data['det']  # 50 x 19 x 6
    toa = [45]  # [useless]



    # transform to torch.Tensor
    features = torch.Tensor(np.expand_dims(features, axis=0)).to(device)         #  50 x 20 x 4096
    labels = torch.Tensor(np.expand_dims(labels, axis=0)).to(device)

    toa = torch.Tensor(np.expand_dims(toa, axis=0)).to(device)
    detections = np.expand_dims(detections, axis=0)
    vid = feature_file.split('/')[-1].split('.')[0]

    return features, labels, toa, detections, vid


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def parse_results(all_outputs, batch_size=1, n_frames=50):
    # parse inference results
    pred_score = np.zeros((batch_size, n_frames), dtype=np.float32)

    # run inference
    for t in range(n_frames):
        pred = all_outputs[t]  # B x 2
        pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
        pred_score[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
    return pred_score


def get_video_frames(video_file, n_frames=50):
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert len(video_data) >= n_frames, video_file
    video_data = video_data[:n_frames]
    return video_data


def preprocess_results(pred_score, cumsum=False):
    from scipy.interpolate import make_interp_spline

    # sampling
    xvals = np.linspace(0,len(pred_score)-1,10)
    pred_mean_reduce = pred_score[xvals.astype(np.int)]

    xvals_new = np.linspace(1,len(pred_score)+1, p.n_frames)
    pred_score = make_interp_spline(xvals, pred_mean_reduce)(xvals_new)

    pred_score[pred_score >= 1.0] = 1.0-1e-3
    xvals = np.copy(xvals_new)
    # copy the first value into x=0
    xvals = np.insert(xvals_new, 0, 0)
    pred_score = np.insert(pred_score, 0, pred_score[0])
    # take cummulative sum of results

    if cumsum:
        pred_score = np.cumsum(pred_score)
        pred_score = pred_score / np.max(pred_score)
    return xvals, pred_score


def draw_curve(xvals, pred_score):
    # pred_score = pred_score *100
    plt.plot(xvals, pred_score, linewidth=3.0)
    plt.axhline(y=0.5, xmin=0, xmax=max(xvals)/(p.n_frames + 2), linewidth=3.0, color='g', linestyle='--')
    # plt.grid(True)
    plt.tight_layout()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='visualize', choices=['extract_feature', 'inference', 'visualize'])
    parser.add_argument('--gpu_id', help='GPU ID', type=int, default=0)
    parser.add_argument('--n_frames', type=int, help='The number of input video frames.', default=50)
    parser.add_argument('--fps', type=float, help='The fps of input video.', default=5) # the fps of causality detection 
    # feature extraction
    parser.add_argument('--video_root_folder', type=str, default="/root/cvpr/data/test_videos/")
    parser.add_argument('--feature_extract_folder', type=str, default='../mmdet_extract_feature')
    parser.add_argument('--mmdetection', type=str, help="the path to the mmdetection.", default="../mmdetection")
    # inference
    parser.add_argument('--label_folder', type=str, help="the path to the feature file.", default="./DSTA_label")
    parser.add_argument('--ckpt_file', type=str, help="the path to the model file.", default="demo/final_model_ccd.pth")
    # visualize
    parser.add_argument('--result_file', type=str, help="the path to the result file.", default="demo/000821_result.npz")
    parser.add_argument('--vis_file', type=str, help="the path to the visualization file.", default="demo/000821_vis.avi")
    p = parser.parse_args()

    import tqdm
    device = torch.device('cuda:'+str(p.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    from src.Models import DSTA
    feature_folder = p.feature_extract_folder
    feature_paths = os.listdir(feature_folder)
    
    # prepare model
    model = init_accident_model(p.ckpt_file, dim_feature=4096, n_frames=50, fps=p.fps)
    # the n_frames = 50 is set by default

    # load feature file
    for feature_path in tqdm.tqdm(feature_paths):
        features, labels, toa, detections, vid = load_input_data(os.path.join(feature_folder, feature_path), device=device)
        # print(features.shape) # 1,87,20,4096 [batch, n_frames, boxes, feature]
        all_outputs_lst = []
        all_alphas_lst = []

        true_frames = features.shape[1]
        left = 0
        clipped_feature = torch.zeros((1,50,features.shape[2], features.shape[3]), device=device)
        while left<true_frames:
            clipped_feature *= 0.0
            clipped_feature[:,0:min(left+50, true_frames)-left,:,:] = features[:,left:min(left+50, true_frames),:,:]
            left += 50
            with torch.no_grad():
                # run inference
                losses, all_outputs, all_hidden, all_alphas = model(clipped_feature, labels, toa, hidden_in=None)
                # print( f'typeof losses = {type(losses)}',
                #        f'typeof all_outputs = {type(all_outputs)}',
                #        f'typeof all_hidden = {type(all_hidden)}',
                #        f'typeof all_alphas = {type(all_alphas)}',
                #        f'typeof detections = {type(detections)}')
                '''
                typeof losses = <class 'dict'> 
                typeof all_outputs = <class 'list'> 
                typeof all_hidden = <class 'list'> 
                typeof all_alphas = <class 'list'> 
                typeof detections = <class 'numpy.ndarray'>
                '''
                all_outputs_lst.extend(all_outputs)
                all_alphas_lst.extend(all_alphas)


        alphas = all_alphas_lst
        # parse and save results
        pred_score= parse_results(all_outputs_lst,true_frames)
        result_file = osp.join(p.label_folder, feature_path.split('/')[-1].split('.')[0] + '_result.npz')
        np.savez_compressed(result_file, score=pred_score[0], det=detections[0],alphas = alphas)

