import cv2
import argparse
import cupy as cp
from cupyx.scipy.ndimage import convolve1d, zoom, maximum_filter
import numpy as np
import time
from functools import wraps
from typing import Optional, Tuple

from cudaconv import convolve as convolve2d
from cudaconv.presets import get_kernel
'''
@software{cuda_conv_accelerator,
  title={CUDA Convolution Accelerator},
  author={Ayomide Caleb Adekoya},
  year={2025},
  url={https://github.com/elcruzo/cuda-conv}
}
'''

class TimingDecorator:
    @staticmethod
    def timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} running time: {end_time - start_time} seconds")
            return result
        return wrapper


class CupyGaborFilter:
    
    def __init__(self, device=None):
        '''
        Gabor Filter for Cupy programming.
        '''
        self.device = device
        pass

    def gabor_1d_kernel(self, frequency: float, ksize: int, theta: float, device=None) -> cp.ndarray:           
        sigma = 2 * cp.pi**2 / frequency
       
        t = cp.linspace(-ksize, ksize, 2 * ksize + 1, device=device) # shape (2 * ksize + 1,)
        S_sigma = cp.exp(-(t - ksize)**2 / (2 * sigma**2))
        cos_c = cp.cos((t - ksize) * theta)
        sin_c = cp.sin((t - ksize) * theta)
        
        gabor_real = S_sigma * cos_c
        gabor_imag = S_sigma * sin_c
        
        return cp.complex64(gabor_real + 1j * gabor_imag)  # shape (1, 2 * ksize + 1), type = Complex64

    def gabor_conv2d_cupy(self, image: cp.ndarray, orientation: float, frequency: float, ksize: int) -> cp.ndarray:
        """2D Gabor Convolution using Cupy."""  
        # assert the shape of image should be 2, width * height
        assert len(image.shape) == 2, "Input image should be 2D (width * height)"
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=3, theta=orientation, lambd=4, gamma=1, psi=0, ktype=cv2.CV_32F)
        kernel_tensor = cp.asarray(kernel)
        result = convolve2d(image,kernel_tensor,use_shared_mem=True, return_numpy=False) # return with same shape
        return result  


class CupyImagePyramid:
    
    def __init__(self):
        self.gaussian_kernel_1d = cp.array([1./ 16.0, 4./ 16.0, 6./ 16.0, 4./ 16.0, 1./ 16.0], dtype=cp.float16)  # 5
        self.kernel_2d = get_kernel('gaussian_5x5').astype(cp.float16)

    def gaussian_pyramid_cupy(self, image: cp.ndarray) -> cp.ndarray:
        """
        Using Cupy implementation of Gaussian pyramid (fully PyTorch-based with separable convolution optimization)
        """
        # Ensure input is 2D tensor (height, width)
        original_shape = image.shape
        assert 2<= len(original_shape) <=3
        # assert image.dtype ==self.gaussian_kernel_1d.dtype== cp.float32
        
        # Use separable convolution: apply 1D Gaussian kernels along height and width separately
        # First apply along width direction (dim=0)
        # print('img dtype=',image.dtype,'kernel dtype=', self.gaussian_kernel_1d.dtype)
        # quit()
        temp_result = convolve1d(image,self.gaussian_kernel_1d, axis = 0,mode='reflect')
        
        # Then apply along height direction (dim=1)
        temp_result = convolve1d(temp_result,self.gaussian_kernel_1d, axis = 1,mode='reflect')
        
        # Downsample: take every other pixel
        if len(original_shape) == 3:
            return temp_result[::2,::2,:]
        return temp_result[::2,::2]

    def gaussian_pyramid_cupy_2d(self, image: cp.ndarray) -> cp.ndarray:
        """
        Using Cupy implementation with 2D convolution for Gaussian pyramid
        """
        assert 2<=len(image.shape) <=3
        
        # Apply 2D convolution using precomputed 2D kernel
        blurred = convolve2d(image=image,kernel=self.kernel_2d,use_shared_mem=True,return_numpy=False)
        
        # Downsample: take every other pixel
        if len(original_shape) == 3:
            downsampled = blurred[::2, ::2, :]
        else:
            downsampled = blurred[::2, ::2]
        
        # Remove dimensions if original input was 2D or 3D
        return downsampled

    def gaussian_pyramid_numpy_opencv(self, image: cp.ndarray) -> cp.ndarray:
       
        # Convert tensor to numpy
        cp_flag = False
        if isinstance(image, cp.ndarray):
            cp_flag = True
            image_np = cp.asnumpy(image)
        else:
            image_np = image
            
        # Apply OpenCV's pyrDown function
        downsampled_np = cv2.pyrDown(image_np)
        
        # Convert back to tensor on original device
        if cp_flag:
            return cp.asarray(downsampled_np)
        else:
            return downsampled_np

    def eight_pyramid_built_torch(self, image: cp.ndarray, height=8) -> list:
        assert 0 < height < 9
        image_list = []
        current = cp.asarray(image)  # ensure cupy
        for _ in range(height):
            image_list.append(current.copy())  # copy = clone
            current = self.gaussian_pyramid_cupy(current)
        return image_list
    
    def eight_pyramid_built_3d(self, image_lst:list[cp.ndarray], height=8) -> list:
        assert 0 < height < 9
        for idx in range(1,len(image_lst)):
            image_lst[idx] += self.gaussian_pyramid_cupy(image_lst[idx-1])


class CupyImageProcessing:
    def __init__(self):
        pass

    def align(self, img1, img2, debug=True):
        """优化后的 2D 对齐"""
        if img1.shape == img2.shape:
            return img1, img2
        
        target_h = max(img1.shape[0], img2.shape[0])
        target_w = max(img1.shape[1], img2.shape[1])
        
        # 使用更快的 zoom 参数，并确保在 float16 下执行
        def _res(img, th, tw):
            if img.shape[0] == th and img.shape[1] == tw:
                return img
            # order=1 是双线性插值，prefilter=False 提速明显
            return zoom(img.astype(cp.float16), (th/img.shape[0], tw/img.shape[1]), 
                        order=1, mode='nearest', prefilter=False)

        return _res(img1, target_h, target_w), _res(img2, target_h, target_w)

    def align3d(self, img1, img2, debug=True):
        """优化后的 3D 对齐"""
        if img1.shape == img2.shape:
            return img1, img2
            
        target_h = max(img1.shape[0], img2.shape[0])
        target_w = max(img1.shape[1], img2.shape[1])
        # 假设通道数 C (如 theta) 是一致的，不需要缩放通道维度
        
        def _res(img, th, tw):
            if img.shape[0] == th and img.shape[1] == tw:
                return img
            # 只在 H, W 维度缩放，保持 C 维度不变 (zoom_factor=1)
            return zoom(img.astype(cp.float16), (th/img.shape[0], tw/img.shape[1], 1), 
                        order=1, mode='nearest', prefilter=False)

        return _res(img1, target_h, target_w), _res(img2, target_h, target_w)

    def subtraction_torch(self, img1, img2, debug=True):
        # 确保减法在 float16 下执行以节省显存
        i1, i2 = self.align3d(img1, img2) if img1.ndim==3 else self.align(img1, img2)
        return i1.astype(cp.float16) - i2.astype(cp.float16)

    def addition_torch(self, img1, img2, debug=True):
        i1, i2 = self.align3d(img1, img2) if img1.ndim==3 else self.align(img1, img2)
        return i1.astype(cp.float16) + i2.astype(cp.float16)

    def conv_function_torch(self, image: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        if len(image.shape) == 2 and len(kernel.shape) == 2:
            result = convolve2d(image=image, kernel=kernel, use_shared_mem=True, return_numpy= False)
            return result
        elif len(image.shape) == 3 and len(kernel.shape) == 3:
            raise NotImplementedError('3d conv are not supported now')
            # result = convolve3d(image=image, kernel=kernel, use_shared_mem=True, return_numpy= False)
            return result
        else:
            raise ValueError(f'Inputs must be 2D or 3D, got {image.shape} and {kernel.shape}')


def cpnormalize_img(img: cp.ndarray, M: float = 1.0, tol: float = 1e-2) -> cp.ndarray:
    """
    Normalize image and scale by (M - mean_of_local_maxima)^2.
    
    Args:
        img: 2D CuPy array (H, W)
        M: target max value after normalization
    
    Returns:
        Scaled normalized image (H, W)
    """
    img_min = img.min()
    img_max = img.max()
    
    # Handle constant or zero image
    if img_max <= img_min:
        return cp.zeros_like(img, dtype=cp.float16)
    
    normalized = (img - img_min) / (img_max - img_min+1e-2) * M
    normalized = normalized.astype(cp.float16)
    
    w, h = img.shape
    size = (max(w // 10, 3), max(h // 10, 3))  # Integer window size
    
    max_filtered = maximum_filter(normalized, size=size)
    maxima_mask = normalized >= max_filtered - tol
    
    mnum = maxima_mask.sum()
    if mnum == 0:
        mbar = 0.0
    else:
        mbar = float(cp.sum(normalized * maxima_mask)) / float(mnum)

    scale_factor = (M - mbar) ** 2
    result = normalized * scale_factor
    
    return result

def cpnormalize_img3d_dict(img_dict: dict, M: float = 1.0, tol: float = 1e-2):
    """
    修正后的字典归一化，支持 float16
    """
    for key in img_dict:
        if img_dict[key] is not None:
            img_dict[key] = cpnormalize_img3d(img_dict[key], M, tol)

def cpnormalize_img3d(img: cp.ndarray, M: float = 1.0, tol: float = 1e-2, axis=(0,1)) -> cp.ndarray:
    """
    针对 float16 优化的 Itti 归一化算法
    """
    # 1. 转为 float16
    img = img.astype(cp.float16)
    
    # 2. 基础归一化到 [0, M]
    img_min = img.min(axis=axis, keepdims=True)
    img_max = img.max(axis=axis, keepdims=True)
    
    
    diff = img_max - img_min
    if diff[0,0,0] <= tol:
        return cp.zeros_like(img, dtype=cp.float16)
    normalized = cp.where(diff > tol, (img - img_min) / (diff + 1e-2) * M, cp.zeros_like(img))
    
    w, h = img.shape[0], img.shape[1]
    # 局部窗口通常取图像宽高的 1/10
    sw, sh = max(w // 10, 3), max(h // 10, 3)
    
    # 这里的 maximum_filter 是计算瓶颈，size=(sw, sh, 1) 保证通道独立
    max_filtered = maximum_filter(normalized, size=(sw, sh, 1))
    
    # 找到局部最大值点
    maxima_mask = (normalized >= max_filtered - tol) & (normalized > 0)
    
    # 计算所有局部最大值的平均值 m_bar
    # 对每个通道独立计算
    sum_maxima = cp.sum(normalized * maxima_mask, axis=axis)
    count_maxima = cp.sum(maxima_mask, axis=axis)
    
    # 避免 count 为 0
    m_bar = cp.where(count_maxima > 0, sum_maxima / (count_maxima + 1e-2), 0)
    
    scale_factor = (M - m_bar) ** 2
    return (normalized * scale_factor).astype(cp.float16)


        
