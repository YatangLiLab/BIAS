import numpy as np
import cupy as cp
import argparse
import cv2
from cupyx.scipy.ndimage import convolve
from cudaconv import convolve as convolve2d
from cudaconv.presets import get_kernel
from utils_cupy import cpnormalize_img,cpnormalize_img3d_dict, CupyImageProcessing, CupyImagePyramid
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
        self.gaussian_kernel_3x3 = cp.array([[1,2,1],[2,4,2],[1,2,1]], dtype=cp.float32) / 16.0
    
    def reset(self):
        self.motion_pyramid = [cp.zeros((*self.shapes[level],4)) for level in range(1,5)]
        self.motion_dict = {(c,s):cp.zeros((*self.shapes[c],4)) for (c,s) in self.cs_lst}

    
    def non_linear(self, x):
        M = cp.max(x)
        M2 = M * M + 1e-7
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
        M_bar = cp.zeros(1,1)
        for mat in self.motion_dict.values():
            M_bar = self.ImageProcessing.addition_torch(M_bar, mat)
        return M_bar

    def four_dir_sim(self,Is0,Is1)->cp.array:
        """
        use Hassenstein-Reichardt-like model to detect motion information.
        """
        self.reset()
        for spatial_index in (1,2,3,4):
            if spatial_index not in self.cal_lst:
                continue
            current_frame = cp.asarray(Is0[spatial_index][:, :, 0], dtype=cp.float32)
            former_frame = cp.asarray(Is1[spatial_index][:, :, 0], dtype=cp.float32)

            max_form = cp.maximum(current_frame,former_frame)+1e-5
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
                l = cp.maximum(diff, 0)
                r = cp.maximum(-diff, 0)
                self.motion_pyramid[spatial_index-1][:,:,dir_pair_index] = l
                self.motion_pyramid[spatial_index-1][:,:,self.half + dir_pair_index] = r
            # shape_value = cp.shape(motion_dict['ls'][-1])[0] * cp.shape(motion_dict['ls'][-1])[1] 
        # tus, tds = simple_minus_separation(self.motion_pyramid[spatial_index-1][:,:,self.half],self.motion_pyramid[spatial_index-1][:,:,self.half+1])
        # tls, trs = simple_minus_separation(self.motion_pyramid[spatial_index-1][:,:,0],self.motion_pyramid[spatial_index-1][:,:,1])
        motion_pyramid = [cp.concatenate((simple_minus_separation(pyramid[:,:,:self.half],pyramid[:,:,self.half:])),axis=2) for pyramid in self.motion_pyramid]
        # each idx stands for a tensor [H,W,C], C=4, C=0,1,2,3 stands for [down, right, up, left]
        sum_pyramid = self.sum_of_image_lst(motion_pyramid)
        result_4ch = convolve(sum_pyramid, self.gaussian_kernel_3x3[:,:,None], mode='constant') # [H,W,C]

        self.scaling(result_4ch)
        return self.generate_motion_saliency_map() # return [H,W,1] as motion saliency map.

if __name__ == '__main__':
    ...