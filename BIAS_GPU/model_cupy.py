import numpy as np
import cupy as cp
import cv2
import argparse
from isaliency_cupy import ICpyrimid, resize2normal, CupyPreProcessor
from cat_osal_cupy import Orientation_Saliency
from motion_saliency_cupy import DynamicSaliency
from utils_cupy import cpnormalize_img, cpnormalize_img3d

class BIAS():
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 初始化各个显著性模型
        self.ic_pyramid = ICpyrimid(args)
        self.orientation_saliency = Orientation_Saliency(args)
        self.motion_saliency = DynamicSaliency(args)
        self.preprocessor = CupyPreProcessor(args, (240, 320))

    def image_saliency(self, processed_img):
        # 构建强度/颜色金字塔
        self.ic_pyramid.build(processed_img)
        # 计算强度/颜色显著性
        self.ic_pyramid.diff_process()
        I_bar, C_bar = self.ic_pyramid.get_conspicuous_map()
        
        # # 构建方向金字塔
        self.orientation_saliency.build_pyramid(self.ic_pyramid.ICs)
        self.orientation_saliency.Orientation_maps()
        O_bar = self.orientation_saliency.synthesis_O_map()
        
        # 整合静态显著性
        static_saliency =  cpnormalize_img(O_bar) + cpnormalize_img(I_bar) + cpnormalize_img(C_bar) 
        static_saliency = cpnormalize_img(static_saliency)
        
        return static_saliency, self.ic_pyramid.ICs

    def dynamic_saliency(self, Is0, Is1):
        # 计算运动显著性
        motion_map = self.motion_saliency.four_dir_sim(Is0, Is1)
        motion_saliency = cp.sum(cpnormalize_img3d(motion_map), axis=2, keepdims=False)
        motion_saliency = cpnormalize_img(motion_saliency)
        return motion_saliency

    def normalization(self, saliency_map):
        # 对显著性图进行归一化
        return cpnormalize_img(saliency_map)

    def forward(self, current_frame, previous_ICs=None):
        _current_frame = cv2.resize(current_frame, (320, 240))
        # 预处理图像
        processed_img = self.preprocessor.process(_current_frame)
        
        # 计算静态显著性
        static_saliency, current_ICs = self.image_saliency(processed_img)
        
        # 计算运动显著性（如果有前一帧的信息）
        if previous_ICs is not None:
            motion_saliency = self.dynamic_saliency(current_ICs, previous_ICs)
            # 整合静态和运动显著性
            combined_saliency = static_saliency + motion_saliency
            combined_saliency = self.normalization(combined_saliency)
            return combined_saliency, current_ICs
        
        return static_saliency, current_ICs

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
    parse.add_argument('--default_size',default=(640,480),type=tuple,help = 'default size of image')
    parse.add_argument('--center',default = (1,2),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
    parse.add_argument('--surrounding',default = (3,4),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
    parse.add_argument('--gamma_correction', type=float, default=2.2, help='Gamma correction value')
    args = parse.parse_args() 
    return args

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    args = parse_args()
    # 初始化BIAS模型
    bias_model = BIAS(args)
    
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
    
    # 处理第一帧，获取初始ICs
    _, prev_ICs = bias_model.forward(prev_frame)

    frame_count = 0
    while True:
        # 读取当前帧
        ret, curr_frame = cap.read()
        if not ret:
            break
                
        # 计算综合显著性
        # saliency_map, curr_ICs = bias_model.forward(curr_frame, prev_ICs)
        saliency_map, curr_ICs = bias_model.forward(curr_frame, prev_ICs)

        
        # 转换回CPU并处理
        saliency_map_cpu = cp.asnumpy(saliency_map).astype(np.float32)
        saliency_map_cpu = cv2.normalize(saliency_map_cpu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 调整大小以匹配原始帧
        saliency_map_resized = cv2.resize(saliency_map_cpu, (curr_frame.shape[1], curr_frame.shape[0]))
        # 转换为彩色图以便叠加
        saliency_colored = cv2.applyColorMap(saliency_map_resized, cv2.COLORMAP_JET)
        
        # 叠加显著性图到原始帧
        overlay = cv2.addWeighted(curr_frame, 0.7, saliency_colored, 0.3, 0)
        
        # 显示结果
        cv2.imshow('Original Frame', curr_frame)
        cv2.imshow('Combined Saliency Map', saliency_map_resized)
        cv2.imshow('Overlay', overlay)

        prev_ICs = curr_ICs
        frame_count += 1
        
        # 按ESC键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Test completed. Processed {frame_count} frames.")
