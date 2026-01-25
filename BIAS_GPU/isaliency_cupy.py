import numpy as np
import cupy as cp
import cv2
from cupyx.scipy.ndimage import zoom, convolve1d
from utils_cupy import cpnormalize_img, CupyImageProcessing, CupyImagePyramid

def resize2normal(image):
    tgtw = 320
    tgth = 240
    ratio_x = tgtw/image.shape[0]
    ratio_y = tgth/image.shape[1]
    return zoom(image,[ratio_x, ratio_y])

def seperate_RGB_chanells(image):
   """
    Seperate RGB chanells to going through following algorithms
    """
   return image[:,:,0], image[:,:,1], image[:,:,2]

class CupyPreProcessor:
    def __init__(self, args, shape_hw):  # shape_hw = (H, W)

        self.build_pyramid = CupyImagePyramid()
        gamma = args.gamma_correction
        inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        
        H, W = shape_hw
        # Gaussian kernels: (H, 1) and (W, 1)
        k_h = cv2.getGaussianKernel(H, H / 2.0)
        k_w = cv2.getGaussianKernel(W, W / 2.0)
        k_h /= k_h.max()
        k_w /= k_w.max()
        
        # mask: (H, W)
        self.mask = (k_h @ k_w.T).astype(np.float32)  # outer product
        self.neg_mask = 1.0 - self.mask

        # Transfer to GPU once
        self.mask_gpu = cp.asarray(self.mask)
        self.neg_mask_gpu = cp.asarray(self.neg_mask)

        self.transformI = cp.asarray([0.299, 0.587, 0.114])

    def process(self, image):
        """
        Args:
            image: np.ndarray of shape (H, W, C), uint8, on CPU
        
        Returns:
            cp.ndarray of shape (H, W, C), float32, on GPU
        """
        # Gamma correction on CPU (OpenCV)
        gamma_corrected = cv2.LUT(image, self.table)  # (H, W, C), uint8
        
        # Move to GPU and convert to float32
        img_gpu = cp.asarray(gamma_corrected, dtype=cp.float32)
        
        # Compute mean on GPU
        img_mean = cp.mean(img_gpu)
        
        # Apply mask: (H, W, 1) * (H, W, C) → (H, W, C)
        mask_expanded = self.mask_gpu[..., cp.newaxis]      # (H, W, 1)
        neg_mask_expanded = self.neg_mask_gpu[..., cp.newaxis]  # (H, W, 1)
        
        gaussian_img = mask_expanded * img_gpu + neg_mask_expanded * img_mean

        ICs = []
        # 还是没想好怎么写……应该先算一下这么搞最后是多少，然后再手动分配一下。

        Is = cp.tensordot(gaussian_img,self.transformI,axes=([-1],[0]))
        
        
        
        
        

class ICpyrimid:
    def __init__(self, args):
        self.args = args
        self.total_height = args.total_height
        self.img_process = CupyImageProcessing()
        self.reset()
    
    def reset(self):
        self.Is = [None] * self.total_height
        self.Ds = [None] * self.total_height
        self.Rs = [None] * self.total_height
        self.Gs = [None] * self.total_height
        self.Bs = [None] * self.total_height
        self.Ys = [None] * self.total_height
    
    def scaling(self, channel1, channel2, c,s):
        tmp = self.img_process(channel1[c] - channel2[c], channel2[s] - channel2[s])
        tmp -= cp.mean(tmp)
        return (cp.float32(cp.where(tmp>0, tmp, 0)), cp.float32(cp.where(tmp<0, -tmp, 0)))

    def ID_scale(self,c,s):
        return self.scaling(self.Is, self.Ds, c, s)
    
    def RG_scale(self,c,s):
        return self.scaling(self.Rs, self.Gs, c, s)
    
    def BY_scale(self,c,s):
        return self.scaling(self.Bs, self.Ys, c, s)