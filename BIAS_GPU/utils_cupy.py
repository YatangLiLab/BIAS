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
        self.gaussian_kernel_1d = cp.array([1., 4., 6., 4., 1.], dtype=cp.float32) / 16.0 # 5
        self.kernel_2d = get_kernel('gaussian_5x5')

    def gaussian_pyramid_cupy(self, image: cp.ndarray) -> cp.ndarray:
        """
        Using Cupy implementation of Gaussian pyramid (fully PyTorch-based with separable convolution optimization)
        """
        # Ensure input is 2D tensor (height, width)
        original_shape = image.shape
        assert 2<= len(original_shape) <=3
        
        # Use separable convolution: apply 1D Gaussian kernels along height and width separately
        # First apply along width direction (dim=0)
        temp_result = convolve1d(image,self.gaussian_kernel_1d,axis = 0,mode='reflect')
        
        # Then apply along height direction (dim=1)
        temp_result = convolve1d(temp_result,self.gaussian_kernel_1d,axis = 1,mode='reflect')
        
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


class CupyImageProcessing:
    """image processing using module"""
    
    def __init__(self):
        pass

    def align(self, img1, img2, debug=True):
        """
        Align two 2D CuPy arrays (H, W) to the same spatial size via bilinear interpolation.

        Args:
            img1, img2: cp.ndarray of shape (H, W)
            debug: bool, enable shape assertions

        Returns:
            (img1_resized, img2_resized): both of shape (H_max, W_max)
        """
        if debug:
            assert img1.ndim == 2 and img2.ndim == 2, \
                f'Inputs must be 2D, got {img1.shape} and {img2.shape}'

        # Early return if both are zero
        if cp.max(img1) == 0 and cp.max(img2) == 0:
            # Return in consistent order; choose larger or first
            shape = img1.shape if img1.shape[0] >= img2.shape[0] else img2.shape
            return cp.zeros(shape=shape), cp.zeros(shape=shape)

        target_h = max(img1.shape[0], img2.shape[0])
        target_w = max(img1.shape[1], img2.shape[1])
        def resize_to_target(x, target_h, target_w):
            if x.shape[0] == target_h and x.shape[1] == target_w:
                return x
            zoom_h = target_h / x.shape[0]
            zoom_w = target_w / x.shape[1]
            # Apply zoom only on 2D array
            resized = zoom(x, (zoom_h, zoom_w), order=1, mode='nearest', prefilter=False)
            return resized

        img1_resized = resize_to_target(img1, target_h, target_w)
        img2_resized = resize_to_target(img2, target_h, target_w)

        return img1_resized, img2_resized

    def subtraction_torch(self, img1: cp.ndarray, img2: cp.ndarray, ifshow: bool = False, debug = True) -> cp.ndarray:
        """
        subtraction two images
        """
        img1_resized, img2_resized = self.align(img1, img2, debug=debug)
        return img1_resized - img2_resized

    def addition_torch(self, img1: cp.ndarray, img2: cp.ndarray) -> cp.ndarray:
        return self.subtraction_torch(img1, -img2)

    def multiplication_torch(self, img1: cp.ndarray, img2: cp.ndarray, debug=True) -> cp.ndarray:
        img1_resized, img2_resized = self.align(img1, img2, debug=debug)
        return img1_resized * img2_resized

    def conv_function_torch(self, image: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        result = convolve2d(image=image, kernel=kernel, use_shared_mem=True, return_numpy= False)
        return result


def cpnormalize_img(img: cp.ndarray, M: float = 1.0, tol: float = 1e-5) -> cp.ndarray:
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
        return cp.zeros_like(img, dtype=cp.float32)
    
    normalized = (img - img_min) / (img_max - img_min) * M
    normalized = normalized.astype(cp.float32)
    
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

class TorchVideoProcessor:
    """Torch based Video data processor"""
    
    def __init__(self):
        pass

    def read_video(self, path: str) -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
        cap = cv2.VideoCapture(path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        final_shape = (width, height)
        return cap, final_shape

    def pre_processing_cupy(self, image: np.ndarray, args: argparse.Namespace) -> cp.ndarray:
        gray_image = cv2.cvtColor(cv2.resize(image, args.default_size), cv2.COLOR_BGR2GRAY)
        tensor_image = cp.asarray(gray_image)
        return tensor_image
    

        
