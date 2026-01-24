import cv2
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
from functools import wraps
from typing import Optional, Tuple

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


class TorchGaborFilter:
    """Gabor Filter for Torch programming."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def gabor_kernel(self, frequency: float, ksize: int, theta: float, device=None) -> torch.Tensor:
        if device is None:
            device = self.device
            
        sigma = 2 * torch.pi**2 / frequency
       
        # 生成核
        t = torch.linspace(-ksize, ksize, 2 * ksize + 1, device=device) # shape (2 * ksize + 1,)
        S_sigma = torch.exp(-(t - ksize)**2 / (2 * sigma**2))
        cos_c = torch.cos((t - ksize) * theta)
        sin_c = torch.sin((t - ksize) * theta)
        
        gabor_real = S_sigma * cos_c
        gabor_imag = S_sigma * sin_c
        
        return torch.complex(gabor_real, gabor_imag).unsqueeze(0)  # shape (1, 2 * ksize + 1), type = Complex

    def gabor_conv2d_torch(self, image: torch.Tensor, orientation: float, frequency: float, ksize: int) -> torch.Tensor:
        """2D Gabor Convolution using PyTorch."""
        # ensure the input size = (batch, channel, height, width)
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # add batch and channel dimension
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)  # add batch dimension
        
        kernel = cv2.getGaborKernel((ksize, ksize), sigma=3, theta=orientation, 
                                    lambd=4, gamma=1, psi=0, ktype=cv2.CV_32F)
        kernel_tensor = torch.from_numpy(kernel).to(image.device).float()
        kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)  # (out_channels, in_channels, H, W)
        
        padding = ksize // 2
        result = F.conv2d(image, kernel_tensor, padding=padding, groups=1)
        
        return result  


class TorchImagePyramid:
    """基于PyTorch的图像金字塔类"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.gaussian_kernel_1d = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32, device=self.device)
        self.gaussian_kernel_1d = self.gaussian_kernel_1d / self.gaussian_kernel_1d.sum()
        kernel_2d = self.gaussian_kernel_1d[:, None] * self.gaussian_kernel_1d[None, :]
        self.gaussian_kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (out_channels, in_channels, H, W)

    def gaussian_pyramid_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Using PyTorch implementation of Gaussian pyramid (fully PyTorch-based with separable convolution optimization)
        """
        # Ensure input is 4D tensor (batch, channel, height, width)
        original_shape = image.shape
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Use separable convolution: apply 1D Gaussian kernels along height and width separately
        # First apply along width direction (dim=-1)
        gaussian_kernel_1d_W = self.gaussian_kernel_1d.view(1, 1, 1, -1)  # Shape: (out_channels, in_channels, 1, kernel_width)
        temp_result = F.conv2d(image, gaussian_kernel_1d_W, padding=(0, 2), groups=1)
        
        # Then apply along height direction (dim=-2)
        gaussian_kernel_1d_H = self.gaussian_kernel_1d.view(1, 1, -1, 1)  # Shape: (out_channels, in_channels, kernel_height, 1)
        blurred = F.conv2d(temp_result, gaussian_kernel_1d_H, padding=(2, 0), groups=1)
        
        # Downsample: take every other pixel
        downsampled = blurred[:, :, ::2, ::2]
        
        return downsampled

    def gaussian_pyramid_torch_2d(self, image: torch.Tensor) -> torch.Tensor:
        """
        Using PyTorch implementation with 2D convolution for Gaussian pyramid
        """
        import torch.nn.functional as F
        
        # Ensure input is 4D tensor (batch, channel, height, width)
        original_shape = image.shape
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Apply 2D convolution using precomputed 2D kernel
        padding = self.gaussian_kernel_2d.shape[-1] // 2  # Same padding as kernel size
        blurred = F.conv2d(image, self.gaussian_kernel_2d, padding=padding, groups=1)
        
        # Downsample: take every other pixel
        downsampled = blurred[:, :, ::2, ::2]
        
        # Remove dimensions if original input was 2D or 3D
        return downsampled

    def gaussian_pyramid_numpy_opencv(self, image: torch.Tensor) -> torch.Tensor:
       
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image
            
        # Apply OpenCV's pyrDown function
        downsampled_np = cv2.pyrDown(image_np)
        
        # Convert back to tensor on original device
        if isinstance(image, torch.Tensor):
            return torch.from_numpy(downsampled_np).to(image.device).float()
        else:
            return downsampled_np

    def eight_pyramid_built_torch(self, image: torch.Tensor) -> list:
        """
        Using PyTorch implementation of eight-level pyramid construction
        """
        image_list = []
        image_current = image.clone() if isinstance(image, torch.Tensor) else torch.from_numpy(image).float()
        
        for i in range(8):
            image_list.append(image_current.clone())
            image_current = self.gaussian_pyramid_torch(image_current)
        return image_list


class TorchImageProcessing:
    """image processing using module"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def align(self, img1:torch.Tensor, img2:torch.Tensor, debug=True):
        if debug:
            assert img1.shape[0:2] == img2.shape[0:2], f'for we wish they have the same batch and channel, and we get img1.shape = {img1.shape}, img2.shape = {img2.shape}'
            assert img1.device == img2.device == self.device, f'for we wish they are on the same device, which is cuda, and we get img1.device = {img1.device}, img2.device = {img2.device}, self.device = {self.device}'
        
        if torch.max(img1) == 0 and torch.max(img2) == 0:
            return img1 if img1.shape[2] > img2.shape[2] else img2
        
        _shape = img1.shape if img1.shape[2] >= img2.shape[2] else img2.shape
        # different shape:
        if img1.shape != img2.shape:
            target_height, target_width = max(img1.shape[2], img2.shape[2]), max(img1.shape[3], img2.shape[3])  
            if img1.shape != (target_height, target_width):
                img1_resized = F.interpolate(
                    img1,
                    size=(img1.shape[0], img1.shape[1], target_height, target_width), 
                    mode='bilinear', 
                    align_corners=True
                )
            else:
                img1_resized = img1
                
            if img2.shape != (target_height, target_width):
                img2_resized = F.interpolate(
                    img2, 
                    size=(img2.shape[0], img2.shape[1],target_height, target_width), 
                    mode='bilinear', 
                    align_corners=True
                )
            else:
                img2_resized = img2
        else:
            img1_resized = img1
            img2_resized = img2
        return img1_resized, img2_resized

    def subtraction_torch(self, img1: torch.Tensor, img2: torch.Tensor, ifshow: bool = False, debug = True) -> torch.Tensor:
        """
        subtraction two images
        """
        # img1, img2 size = [batch, channel, height, width]
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        img1_resized, img2_resized = self.align(img1, img2, debug=debug)
        return img1_resized - img2_resized

    def addition_torch(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self.subtraction_torch(img1, -img2)

    def multiplication_torch(self, img1: torch.Tensor, img2: torch.Tensor, debug=True) -> torch.Tensor:
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        img1_resized, img2_resized = self.align(img1, img2, debug=debug)
        return img1_resized * img2_resized

    def conv_function_torch(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (batch, channel, H, W)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)  # (batch, H, W) -> (batch, channel, H, W)
        if len(kernel.shape) == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (out_ch, in_ch, H, W)
        pad_size = kernel.shape[-1] // 2
        padded_image = F.pad(image, (pad_size, pad_size, pad_size, pad_size), mode='reflect')      
        result = F.conv2d(padded_image, kernel)
        return result.squeeze()


class TorchNormalization:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def normalize_img_torch(self, img: torch.Tensor, M: float = 1.0) -> torch.Tensor:
        """
        使用PyTorch实现图像归一化
        """
        img_min = torch.min(img)
        img_max = torch.max(img)
        
        if img_max == 0:
            if img_min == 0:
                return img
            else:
                raise RuntimeError("奇怪的事情发生了")

        # 归一化到[0, M]
        if img_max - img_min != 0:
            normalized = (img - img_min) / (img_max - img_min) * M
        else:
            normalized = img * 0  # 防止除零
        
        # 计算局部最大值（简化版本，使用最大池化）
        # 这里我们使用一个简化的版本，因为原始的find_maximum_mat依赖于DLL
        # 我们使用最大池化来近似局部最大值检测
        if len(normalized.shape) == 2:
            normalized = normalized.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        elif len(normalized.shape) == 3:
            normalized = normalized.unsqueeze(0)  # 添加批次维度
        
        # 使用最大池化近似局部最大值
        local_max = F.max_pool2d(normalized, kernel_size=3, stride=1, padding=1)
        local_maximum_mask = (normalized == local_max).float()
        
        # 计算掩码中的元素数量和平均值
        mnum = torch.sum(local_maximum_mask)
        if mnum == 0:
            return torch.zeros_like(normalized).squeeze()
        
        maxima = normalized * local_maximum_mask
        mbar = torch.sum(maxima) / (mnum + 1e-5)
        
        result = normalized * (M - mbar)**2
        return result.squeeze()

    def not_normalize_img_torch(self, img: torch.Tensor, M: float = 1.0) -> torch.Tensor:
        """
        使用PyTorch实现非归一化图像处理
        """
        img_min = torch.min(img)
        img_max = torch.max(img)
        
        if (img_max - img_min) > 0:
            image = (img - img_min) / (img_max - img_min) * M
        else:
            image = img * 0
        
        # 获取形状
        w, h = image.shape
        
        # 使用scipy的maximum_filter的逻辑，但用PyTorch实现
        # 这是一个简化的实现
        pool_size = (max(int(w/10), 3), max(int(h/10), 3))
        
        if len(image.shape) == 2:
            img_unsq = image.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        else:
            img_unsq = image.unsqueeze(0)
        
        # 使用最大池化
        max_filtered = F.max_pool2d(img_unsq, kernel_size=pool_size, stride=1, padding=(pool_size[0]//2, pool_size[1]//2))
        maxima_mask = (img_unsq == max_filtered).float()
        
        # 应用掩码
        maxima_values = img_unsq * maxima_mask
        mnum = torch.sum(maxima_mask)
        mbar = torch.sum(maxima_values) / (mnum + 1e-3) if mnum > 0 else torch.tensor(0.0)
        
        result = img_unsq * ((M - mbar)**2)
        return result.squeeze()


class TorchVideoProcessor:
    """基于PyTorch的视频处理器类"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def read_video(self, path: str) -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
        """
        读取视频
        """
        cap = cv2.VideoCapture(path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        final_shape = (width, height)
        return cap, final_shape

    def pre_processing_torch(self, image: np.ndarray, args: argparse.Namespace) -> torch.Tensor:
        """
        执行必要的预处理工作，返回PyTorch张量
        """
        # OpenCV预处理
        gray_image = cv2.cvtColor(cv2.resize(image, args.default_size), cv2.COLOR_BGR2GRAY)
        # 转换为PyTorch张量并移动到指定设备
        tensor_image = torch.from_numpy(gray_image).float().to(self.device)
        return tensor_image


class TorchInterferenceManager:
    """基于PyTorch的干预管理器类"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def get_gaussian_kernel_torch(self, ksize: int, sigma: float, dim: int = 2) -> torch.Tensor:
        """
        使用PyTorch创建高斯核
        """
        # 创建1D高斯核
        coords = torch.arange(ksize, dtype=torch.float32, device=self.device) - (ksize - 1) / 2.0
        gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        if dim == 1:
            return gaussian_1d
        elif dim == 2:
            gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
            return gaussian_2d
        else:
            raise NotImplementedError

    def interference_function_torch(self, current_Saliency_map: torch.Tensor, 
                                   former_fixation_point: Tuple[int, int], 
                                   mode: str = "Gaussian", 
                                   boxwidth_param: float = 0.2) -> torch.Tensor:
        """
        使用PyTorch实现干预函数
        """
        mat_shape = current_Saliency_map.shape
        device = current_Saliency_map.device
        
        if mode == "Gaussian":
            # 创建高斯核
            gaussian_gen = TorchInterferenceManager(device)
            long_gaussian = gaussian_gen.get_gaussian_kernel_torch(max(mat_shape)*2, max(mat_shape)/2, dim=1)
            short_gaussian = gaussian_gen.get_gaussian_kernel_torch(min(mat_shape)*2, min(mat_shape)/2, dim=1)
            Gaussian_kernal = torch.outer(short_gaussian.squeeze(), long_gaussian.squeeze())
            
            # 计算调整矩阵
            row_start = min(mat_shape[0]) - former_fixation_point[0]
            row_end = min(mat_shape[0]) + mat_shape[0] - former_fixation_point[0]
            col_start = max(mat_shape[1]) - former_fixation_point[1]
            col_end = max(mat_shape[1]) + mat_shape[1] - former_fixation_point[1]
            
            # 处理边界情况
            adjust_mat = torch.zeros_like(current_Saliency_map)
            k_rows, k_cols = Gaussian_kernal.shape
            
            # 确定有效的索引范围
            r_start = max(0, -row_start)
            r_end = min(mat_shape[0], k_rows - row_start)
            c_start = max(0, -col_start)
            c_end = min(mat_shape[1], k_cols - col_start)
            
            kr_start = max(0, row_start)
            kr_end = kr_start + (r_end - r_start)
            kc_start = max(0, col_start)
            kc_end = kc_start + (c_end - c_start)
            
            if r_start < r_end and c_start < c_end and kr_start < kr_end and kc_start < kc_end:
                adjust_mat[r_start:r_end, c_start:c_end] = Gaussian_kernal[kr_start:kr_end, kc_start:kc_end]
            
            adjust_mat = adjust_mat / torch.max(adjust_mat)
            weight_mat = adjust_mat * current_Saliency_map
            
        elif mode == "box":
            adjust_mat = torch.ones_like(current_Saliency_map) * 0.1
            upper_bound = int(max(0, former_fixation_point[0] - boxwidth_param * mat_shape[0]))
            lower_bound = int(min(mat_shape[0], former_fixation_point[0] + boxwidth_param * mat_shape[0]))
            left_bound = int(max(0, former_fixation_point[1] - boxwidth_param * mat_shape[1]))
            right_bound = int(min(mat_shape[1], former_fixation_point[1] + boxwidth_param * mat_shape[1]))
            adjust_mat[upper_bound:lower_bound, left_bound:right_bound] += 0.9
            weight_mat = adjust_mat * current_Saliency_map
            
        elif mode == "None":
            weight_mat = current_Saliency_map
        else:
            raise NotImplementedError
            
        return weight_mat

# class GPUDLLManager:
#     """GPU DLL管理器类 - 管理与GPU相关的DLL功能"""
#     
#     def __init__(self, maximum_dll_path: str = './find_local_maximas.dll',
#                  gabor_dll_path: str = "./gabor_filter.dll",
#                  fillin_dll_path: str = "./fillin.dll"):
#         self.maximum_dll = self.load_maximum_dll(maximum_dll_path)
#         self.gabor_lib = self.initialize_gabor_lib(gabor_dll_path)
#         self.fillIn_lib = self.initialize_fillIn_lib(fillin_dll_path)
#     
#     def load_maximum_dll(self, path: str = './find_local_maximas.dll'):
#         """加载查找局部最大值的DLL"""
#         maximum_dll = ctypes.CDLL(path)
#         return maximum_dll
# 
#     def find_maximum_mat(self, image: np.ndarray) -> np.ndarray:
#         """使用DLL查找局部最大值"""
#         assert len(image.shape) == 2
#         image = np.float32(image)/(np.max(image)+1e-6)
#         M, N = image.shape
#         result = np.zeros((M,N),dtype=np.uint8)
#         image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
#         self.maximum_dll.find_local_maximas_wrapper.argtypes = [
#             ctypes.POINTER(ctypes.c_float), 
#             ctypes.POINTER(ctypes.c_uint8), 
#             ctypes.c_int, 
#             ctypes.c_int
#         ]
#         self.maximum_dll.find_local_maximas_wrapper(image_ptr, result_ptr, M, N)
#         return result
# 
#     def initialize_gabor_lib(self, path: str = "./gabor_filter.dll"):
#         """初始化Gabor滤波库"""
#         gabor_lib = ctypes.cdll.LoadLibrary(path)
#         gabor_lib.gabor_filter.argtypes = [
#             ctypes.POINTER(ctypes.c_float), 
#             ctypes.c_int, 
#             ctypes.c_int, 
#             ctypes.c_float, 
#             ctypes.c_float, 
#             ctypes.c_int
#         ]
#         return gabor_lib
# 
#     def gabor_filter(self, image: np.ndarray, orientation: float, frequency: float, kernal_size: int):
#         """Gabor滤波核心函数"""
#         image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         M, N = image.shape
#         try:
#             self._specific_gabor_filter(image_ptr, M, N, orientation, frequency, kernal_size)
#         except Exception as e:
#             print(f"Error: {e}")
#             raise RuntimeError(e)
# 
#     def _specific_gabor_filter(self, image_ptr, M: int, N: int, orientation: float, frequency: float, kernal_size: int):
#         """使用DLL执行Gabor滤波"""
#         self.gabor_lib.gabor_filter(image_ptr, M, N, orientation, frequency, kernal_size)
# 
#     def initialize_fillIn_lib(self, path: str = "./fillin.dll"):
#         """初始化填充库"""
#         if path[-4:] != ".dll":
#             raise AssertionError("路径可能不正确，文件扩展名不是.dll")
#         fillIn_lib = ctypes.cdll.LoadLibrary(path)
#         fillIn_lib.update_saliency_map.argtypes = [
#             ctypes.POINTER(ctypes.c_float), 
#             ctypes.POINTER(ctypes.c_float), 
#             ctypes.c_int, 
#             ctypes.c_int, 
#             ctypes.c_float,
#             ctypes.c_float,
#             ctypes.c_float, 
#             ctypes.c_int
#         ]
#         return fillIn_lib
# 
#     def fill_blank(self, saliency_img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
#         """填充空白的核心函数"""
#         assert saliency_img.shape == original_img.shape
#         saliency_img = np.float32(saliency_img)/255
#         original_img = np.float32(original_img)/255
#         saliency_ptr = saliency_img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         image_ptr = original_img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#         M, N = saliency_img.shape
#         
#         try:
#             self._fill_in_it(saliency_ptr, image_ptr, M, N, 
#                             np.mean(original_img), np.std(original_img, ddof=0))
#         except Exception as e:
#             print(f"Error: {e}")
#             raise RuntimeError(e)
#         return saliency_img
# 
#     def _fill_in_it(self, saliency_ptr, image_ptr, M: int, N: int, 
#                     image_mean: float, image_sd: float, 
#                     threshold: float = 0.03, max_iteration: int = 50):
#         """使用DLL填充空白"""
#         self.fillIn_lib.update_saliency_map(
#             saliency_ptr, image_ptr, M, N, 
#             image_mean, image_sd, threshold, max_iteration
#         )
