import numpy as np
import cupy as cp
import argparse
import cv2
from cupyx.scipy.ndimage import convolve
from cudaconv import convolve as convolve2d
from cudaconv.presets import get_kernel
from utils_cupy import cpnormalize_img,cpnormalize_img3d_dict, CupyImageProcessing, CupyImagePyramid, cpnormalize_img3d
from isaliency_cupy import ICpyrimid, resize2normal



def simple_minus_separation(mat_lst1,mat_lst2):
    """
    simply use minus to find the true value of each point optic flow, not after calculating center-surrounding
    """
    assert len(mat_lst1) == len(mat_lst2)
    pos_lst = []
    neg_lst = []
    for m1, m2 in zip(mat_lst1, mat_lst2):
        diff = m1 - m2
        pos_lst.append(cp.maximum(diff, 0))
        neg_lst.append(cp.maximum(-diff, 0))
    return pos_lst, neg_lst


def zero_edge(image):
    """
    Zero out the edge pixels of the image
    """
    image[:1, :] = image[-1:, :] = image[:, :1] = image[:, -1:] = 0
    return image

class DynamicSaliency:
    def __init__(self, args:argparse.Namespace):
        self.args = args
        self.directions = [(-1,0), (0,1), (1,0), (0,-1)] # [down, right, up, left]
        self.subtract_idxs = [2,3,0,1]
        self.c_set = sorted(self.args.center)
        self.delta_set = sorted(self.args.surrounding)
        self.cs_lst = [(c,c+d) for c in self.c_set for d in self.delta_set]
        self.cal_lst = list(set([c for (c,s) in self.cs_lst] + [s for (c,s) in self.cs_lst]))
        self.half = len(self.directions)//2
        self.shapes = [(240, 320), (120, 160), (60, 80), (30, 40), (15, 20), (8, 10), (4, 5), (2, 3)]
        self.ImageProcessing = CupyImageProcessing()
        # 初始化3x3高斯核
        self.gaussian_kernel_3x3 = cp.array([[1,2,1],[2,4,2],[1,2,1]], dtype = cp.float32) / 16.0
    
    def reset(self):
        self.motion_pyramid = [cp.zeros((*self.shapes[level],4)) for level in range(self.args.total_height)]
        self.motion_dict = {(c,s):cp.zeros((*self.shapes[c],4)) for (c,s) in self.cs_lst}

    def non_linear(self, x):
        M = cp.max(x)
        M2 = M * M + 1e-2
        return 2 * x / cp.sqrt(M2 + 4 * x * x)

    def sum_of_image_lst(self, mat_lst):
        """
        calculate the sum of a matrix list, after some resize.
        """
        rst_image = mat_lst[0]
        for i in range(1,len(mat_lst)):
            rst_image = self.ImageProcessing.addition_torch(rst_image, mat_lst[i])
        return rst_image
    
    def scaling(self, pyramid,c,s):
        diff = pyramid[c][:,:,:2] - pyramid[c][:,:,2:]
        diff = self.ImageProcessing.subtraction_torch(diff, pyramid[s][:,:,:2] - pyramid[s][:,:,2:])
        diff -= cp.mean(diff, axis=(0,1), keepdims=True)
        self.motion_dict[(c,s)] = cp.concatenate((cp.maximum(diff, 0), cp.maximum(-diff, 0)), axis=2)
        
    def diff_process(self, pyramid):
        for (c,s) in self.cs_lst:
            self.scaling(pyramid, c,s)
        cpnormalize_img3d_dict(self.motion_dict)
    
    def generate_motion_saliency_map(self):
        M_bar = None
        for mat in self.motion_dict.values():
            if M_bar is None:
                M_bar = mat
            else:
                M_bar = self.ImageProcessing.addition_torch(M_bar, mat)
        return M_bar

    def four_dir_sim(self,Is0,Is1)->cp.array:
        """
        use Hassenstein-Reichardt-like model to detect motion information.
        """
        self.reset()
        for spatial_index in range(self.args.total_height):
            if spatial_index not in self.cal_lst:
                continue
            current_frame = cp.asarray(Is0[spatial_index][:, :, 0], dtype=cp.float16)
            former_frame = cp.asarray(Is1[spatial_index][:, :, 0], dtype=cp.float16)

            max_form = cp.maximum(current_frame,former_frame)+1e-2
            # tmp = cp.abs(current_frame - former_frame)
            shape = current_frame.shape
            edge = 2
            padded_former_frame = cp.pad(former_frame,edge,mode='constant')
            for dir_pair_index in range(self.half):
                l_pair = cp.abs(current_frame - padded_former_frame[self.directions[dir_pair_index][0]+edge: self.directions[dir_pair_index][0]+edge+shape[0],\
                                                                    self.directions[dir_pair_index][1]+edge: self.directions[dir_pair_index][1]+edge+shape[1]]) /max_form
                r_pair = cp.abs(current_frame - padded_former_frame[self.directions[self.half + dir_pair_index][0]+edge: self.directions[self.half + dir_pair_index][0]+edge+shape[0],\
                                                                    self.directions[self.half + dir_pair_index][1]+edge: self.directions[self.half + dir_pair_index][1]+edge+shape[1]]) /max_form
                padd_r_pair = cp.pad(r_pair,1,mode='reflect')
                l = cp.exp(-1/50 * l_pair) # * tmp
                r = cp.exp(-1/50 * padd_r_pair[self.directions[dir_pair_index][0]+1 : self.directions[dir_pair_index][0]+1+shape[0],\
                                                                                          self.directions[dir_pair_index][1]+1 : self.directions[dir_pair_index][1]+1+shape[1]])# *tmp
                diff = l-r
                self.motion_pyramid[spatial_index][:,:,dir_pair_index] = cp.maximum(diff, 0)
                self.motion_pyramid[spatial_index][:,:,self.half + dir_pair_index] = cp.maximum(-diff, 0)

        processing = lambda x,y: self.ImageProcessing.subtraction_torch(x,y)
        split_cat = lambda x: cp.concatenate((cp.maximum(x,0), cp.maximum(-x,0)),axis=2)
        # print([pyramid.shape for pyramid in self.motion_pyramid])
        motion_pyramid = [(split_cat(processing(pyramid[:,:,:self.half],pyramid[:,:,self.half:]))) for pyramid in self.motion_pyramid]
        # each idx stands for a tensor [H,W,C], C=4, C=0,1,2,3 stands for [down, right, up, left]
        
        # 计算中心-周围差异
        self.diff_process(motion_pyramid)
        
        # 生成运动显著性图
        saliency_map = self.generate_motion_saliency_map()
        
        # 确保数据类型一致后再进行卷积
        saliency_map = saliency_map.astype(cp.float32)
        gaussian_kernel = self.gaussian_kernel_3x3.astype(cp.float32)[:,:,None]
        saliency_map = convolve(saliency_map, gaussian_kernel, mode='constant') # [H,W,4]
        
        return saliency_map # return [H,W,4] as motion saliency map.

if __name__ == '__main__':
    # 解析命令行参数
    import matplotlib.pyplot as plt
    from isaliency_cupy import CupyPreProcessor
    def parse_args():
        parse = argparse.ArgumentParser(description='Essential parameters for gabor processing') 
        parse.add_argument('--image_path', default="E:\\your_path\\RealtimeSaliency\\test_image\\circle.jpg", type=str, help='path of sample image') 
        parse.add_argument('--total_height',default=8,type=int,help='total height of orientation pyramid, equals to height of gaussian pyramid + kernel size Pyramid')
        parse.add_argument('--pyramid_height', default=5, type=int, help='height of Gaussian Pyramid')
        parse.add_argument('--gabor_kernel_size',default=33,type=int,help='the minimal value of gabor kernel size. when meet some constrains, we would double some params.')
        parse.add_argument('--num_of_thetas',default=4,type=int,help='different gabor filter orientations.')
        parse.add_argument('--mini_sigma',default=0.5,type=float,help='sigma of gabor kernel, if the image is too small hori2then double it.')
        parse.add_argument('--gabor_lambda',default=np.pi/np.sqrt(2*np.log(1/0.5)),type=float,help = 'lambda for gabor kernels')
        parse.add_argument('--gabor_gamma',default=1,type=float,help = 'gamma value for gabor filter.')
        parse.add_argument('--default_size',default=(320,240),type=tuple,help = 'default size of image')
        parse.add_argument('--center',default = (1,),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
        parse.add_argument('--surrounding',default = (4,),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
        parse.add_argument('--gamma_correction', type=float, default=2.2, help='Gamma correction value')
        args = parse.parse_args() 
        return args
    
    args = parse_args()
    
    # 初始化DynamicSaliency对象
    preprossor = CupyPreProcessor(args,(240, 320))
    motion_saliency = DynamicSaliency(args)
    
    # 视频路径
    video_path = "/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/example_data/demo.AVI"
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit()
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        exit()
    
    ic_pyramid = ICpyrimid(args)
    _prev_frame = cv2.resize(prev_frame, (320, 240))
    _prev_frame = preprossor.process(_prev_frame)
    ic_pyramid.build(cp.asarray(_prev_frame))
    Is1 = ic_pyramid.ICs

    frame_count = 0
    while True:
        # 读取当前帧
        ret, curr_frame = cap.read()
        if not ret:
            break
                
        # 生成图像金字塔
        _curr_frame = cv2.resize(curr_frame, (320, 240))
        curr_frame_processed = preprossor.process(_curr_frame)
        ic_pyramid.build(cp.asarray(curr_frame_processed))
        Is0 = ic_pyramid.ICs
        
        
        # 计算运动显著性
        saliency_map = motion_saliency.four_dir_sim(Is0, Is1)
        
        saliency_map = cp.sum(cpnormalize_img3d(saliency_map),axis=2,keepdims=False)
        
        # 转换回CPU并处理
        saliency_map_cpu = cp.asnumpy(saliency_map)
        saliency_map_cpu = cv2.normalize(saliency_map_cpu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 调整大小以匹配原始帧
        saliency_map_resized = cv2.resize(saliency_map_cpu, (curr_frame.shape[1], curr_frame.shape[0]))
        # 转换为彩色图以便叠加
        saliency_colored = cv2.applyColorMap(saliency_map_resized, cv2.COLORMAP_JET)
        
        # 叠加显著性图到原始帧
        overlay = cv2.addWeighted(curr_frame, 0.7, saliency_colored, 0.3, 0)
        
        # 显示结果
        cv2.imshow('Original Frame', curr_frame)
        cv2.imshow('Motion Saliency Map', saliency_map_resized)
        cv2.imshow('Overlay', overlay)
        
        # 保存每10帧的结果
        # if frame_count % 10 == 0:
        #     cv2.imwrite(f"motion_saliency_frame_{frame_count}.jpg", saliency_map_resized)
        #     cv2.imwrite(f"overlay_frame_{frame_count}.jpg", overlay)
        
        Is1 = Is0
        frame_count += 1
        
        # 按ESC键退出
        if cv2.waitKey(10) & 0xFF == 27:
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Test completed. Processed {frame_count} frames.")