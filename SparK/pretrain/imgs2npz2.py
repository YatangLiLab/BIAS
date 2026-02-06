import torch
import pickle
import os
import tqdm
import cv2
import numpy as np

USING_GPU_IF_AVAILABLE = True
_ = torch.empty(1)
if torch.cuda.is_available() and USING_GPU_IF_AVAILABLE:
    _ = _.cuda()
DEVICE = _.device
print(f'[DEVICE={DEVICE}]')

from PIL import Image
import torchvision.transforms.functional as TF
import pathlib
IMAGENET_RGB_MEAN = torch.tensor((0.27), device=DEVICE).reshape(1, 1, 1, 1) # torch.tensor((0.485, 0.456, 0.406), device=DEVICE).reshape(1, 3, 1, 1)
IMAGENET_RGB_STD  = torch.tensor((0.27), device=DEVICE).reshape(1, 1, 1, 1) # torch.tensor((0.229, 0.224, 0.225), device=DEVICE).reshape(1, 3, 1, 1)

def transform_image(size: int, img:Image.Image):
    img = img.convert('L')
    img = TF.resize(img, [size,size])
    img = TF.to_tensor(img).unsqueeze(0).to(DEVICE).sub(IMAGENET_RGB_MEAN).div_(IMAGENET_RGB_STD)
    return img
def load_image(size: int, img_file: str):
    img = Image.open(img_file).convert('L')
    img = TF.resize(img, [size,size])
    img = TF.to_tensor(img).unsqueeze(0).to(DEVICE).sub(IMAGENET_RGB_MEAN).div_(IMAGENET_RGB_STD)
    return img
def denormalize(img_bchw):
    return img_bchw.mul(IMAGENET_RGB_STD).add_(IMAGENET_RGB_MEAN).clamp_(0., 1.)


anno_path = '/mnt/e/Li Lab/MS-TCA/MyMSTCN/data/saliency_annotation.pkl'
annodata = pickle.load(open(anno_path, 'rb'))
name_list = [f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}'.replace('v_','') for anno in annodata]
sm_img_root = '/mnt/e/Li Lab/MS-TCA/MyMSTCN/data/Ittidataset/itti_dataset/val'
image_folders = [os.path.join(sm_img_root, name) for name in name_list]
cnt = 0
for img_folder in tqdm.tqdm(image_folders):
    if os.path.exists(img_folder): cnt += 1
print(f'{cnt} videos exist in {len(image_folders)}, start processing.')

from models.resnet18_inference import SaliencyMap_INF
from timm.models import create_model

dense_encoder = create_model('SaliencyMap_INF')# SaliencyMap()
dense_encoder.load_state_dict(torch.load(os.path.join('ResNet18-Itti-exp','SaliencyMap_1kpretrained_timm_style.pth'))) # load corresponding
# dense_encoder.to_dense()
# convert_sparse_bn_to_regular_bn(dense_encoder)
dense_encoder.eval()
dense_encoder = dense_encoder.cuda()

tgt_npz_folder = '/mnt/e/Li Lab/MS-TCA/MyMSTCN/data/Ittidataset/Ittifeature_npy'
tgt_path = pathlib.Path(tgt_npz_folder)
tgt_path.mkdir(parents=True, exist_ok=True)

for idx, img_folder in enumerate(tqdm.tqdm(image_folders)):
    # if os.path.exists(os.path.join(tgt_path, f'{idx}.npy')): continue
    tmp_process_list = []

    # vid_cap = cv2.VideoCapture(vid_name)
    # vid_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # while vid_cap.isOpened():              
    #     ret, frame = vid_cap.read()         
    #     if ret:                         
    #         image = Image.fromarray(frame)
    #         image = transform_image(224, image)
    #         tmp_process_list.append(dense_encoder(image,False).flatten().unsqueeze(dim=0).cpu().detach())
    #     else:
    #         vid_cap.release()
    if os.path.exists(img_folder):
        imgs = os.listdir(img_folder)
        imgs.sort(key=lambda x: int(x.split('.')[0]))
        imgs = [img for img in imgs if int(img.split('.')[0]) % 6 == 0]
        for img_name in tqdm.tqdm(imgs):
            image = Image.open(os.path.join(img_folder, img_name))
            image = transform_image(224, image)
            tmp_process_list.append(dense_encoder(image,False).flatten().unsqueeze(dim=0).cpu().detach())
        tmp_process_list = torch.cat(tmp_process_list, dim=0).numpy()
        np.save(os.path.join(tgt_path, f'{idx}.npy'), tmp_process_list)
        print(tmp_process_list.shape)
        torch.cuda.empty_cache()

