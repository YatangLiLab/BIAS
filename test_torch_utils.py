#!/usr/bin/env python
"""
测试GPU版utils模块的功能（PyTorch版本）
"""

import torch
import numpy as np
import cv2
from BIAS_GPU.utils import (
    TorchGaborFilter, 
    TorchImagePyramid, 
    TorchImageProcessing, 
    TorchNormalization, 
    TorchVideoProcessor, 
    TorchInterferenceManager,
    TimingDecorator
)

def test_torch_functionality():
    """测试PyTorch版本的基本功能"""
    print("开始测试PyTorch版GPU utils模块...")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建一个示例图像用于测试
    test_image_np = np.random.rand(100, 100).astype(np.float32)
    test_image_torch = torch.from_numpy(test_image_np).to(device)
    
    print("\n1. 测试PyTorch Gabor滤波器...")
    gabor_filter = TorchGaborFilter(device=device)
    result = gabor_filter.gabor_conv2d_torch(test_image_torch, orientation=0.0, frequency=0.1, ksize=5)
    print(f"   Gabor滤波结果形状: {result.shape}")
    print(f"   结果在设备: {result.device}")
    
    print("\n2. 测试PyTorch图像处理...")
    processor = TorchImageProcessing(device=device)
    processed = processor.addition_torch(test_image_torch, test_image_torch * 0.5)
    print(f"   图像加法结果形状: {processed.shape}")
    print(f"   结果在设备: {processed.device}")
    
    print("\n3. 测试PyTorch图像金字塔...")
    pyramid = TorchImagePyramid(device=device)
    reduced = pyramid.gaussian_pyramid_torch(test_image_torch)
    print(f"   高斯金字塔结果形状: {reduced.shape}")
    print(f"   结果在设备: {reduced.device}")
    
    print("\n4. 测试PyTorch归一化...")
    norm = TorchNormalization(device=device)
    normalized = norm.normalize_img_torch(test_image_torch)
    print(f"   归一化结果形状: {normalized.shape}")
    print(f"   结果在设备: {normalized.device}")
    
    print("\n5. 测试PyTorch干预管理器...")
    interference = TorchInterferenceManager(device=device)
    saliency_map = torch.rand(50, 50, device=device)
    result = interference.interference_function_torch(saliency_map, (25, 25), mode="Gaussian")
    print(f"   干预函数结果形状: {result.shape}")
    print(f"   结果在设备: {result.device}")
    
    print("\n6. 测试计时装饰器...")
    @TimingDecorator.timer
    def sample_func():
        return sum(range(1000))
    
    result = sample_func()
    print(f"   计时装饰器测试结果: {result}")
    
    print("\n所有PyTorch功能测试完成！")
    
    # 性能对比测试
    print("\n7. 性能对比测试:")
    import time
    
    # CPU vs GPU 时间比较（如果GPU可用）
    cpu_tensor = test_image_torch.cpu()
    
    # CPU处理时间
    start_time = time.time()
    cpu_result = gabor_filter.gabor_conv2d_torch(cpu_tensor, orientation=0.5, frequency=0.2, ksize=7)
    cpu_time = time.time() - start_time
    
    if torch.cuda.is_available():
        gpu_tensor = test_image_torch.cuda()
        start_time = time.time()
        gpu_result = gabor_filter.gabor_conv2d_torch(gpu_tensor, orientation=0.5, frequency=0.2, ksize=7)
        gpu_time = time.time() - start_time
        
        print(f"   CPU处理时间: {cpu_time:.4f}s")
        print(f"   GPU处理时间: {gpu_time:.4f}s")
        print(f"   GPU加速比: {cpu_time/gpu_time:.2f}x" if gpu_time > 0 else "N/A")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_torch_functionality()