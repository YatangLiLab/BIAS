#!/usr/bin/env python
"""
测试GPU版utils模块的功能（PyTorch版本）- 改进后
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

def test_improved_torch_functionality():
    """测试改进后的PyTorch版本功能"""
    print("开始测试改进版PyTorch GPU utils模块...")
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
    
    print("\n3. 测试PyTorch图像金字塔（纯PyTorch实现）...")
    pyramid = TorchImagePyramid(device=device)
    reduced = pyramid.gaussian_pyramid_torch(test_image_torch)
    print(f"   高斯金字塔结果形状: {reduced.shape}")
    print(f"   结果在设备: {reduced.device}")
    
    print("\n4. 测试PyTorch归一化...")
    norm = TorchNormalization(device=device)
    normalized = norm.normalize_img_torch(test_image_torch)
    print(f"   归一化结果形状: {normalized.shape}")
    print(f"   结果在设备: {normalized.device}")
    
    print("\n5. 测试PyTorch干预管理器（纯PyTorch高斯核）...")
    interference = TorchInterferenceManager(device=device)
    saliency_map = torch.rand(50, 50, device=device)
    result = interference.interference_function_torch(saliency_map, (25, 25), mode="Gaussian")
    print(f"   干预函数结果形状: {result.shape}")
    print(f"   结果在设备: {result.device}")
    
    print("\n6. 测试批量Gabor滤波...")
    batch_images = torch.stack([test_image_torch, test_image_torch*0.8, test_image_torch*0.6])
    batch_result = gabor_filter.gabor_conv2d_batch(batch_images, orientation=0.5, frequency=0.2, ksize=5)
    print(f"   批量Gabor滤波结果形状: {batch_result.shape}")
    print(f"   结果在设备: {batch_result.device}")
    
    print("\n7. 性能对比测试 (如果GPU可用):")
    import time
    
    # 创建更大的测试图像以更好地观察性能差异
    large_image = torch.rand(200, 200, device=device)
    
    # CPU处理时间
    if device == 'cuda':
        cpu_image = large_image.cpu()
        start_time = time.time()
        cpu_result = gabor_filter.gabor_conv2d_torch(cpu_image, orientation=0.5, frequency=0.2, ksize=7)
        cpu_time = time.time() - start_time
        
        # GPU处理时间
        start_time = time.time()
        gpu_result = gabor_filter.gabor_conv2d_torch(large_image, orientation=0.5, frequency=0.2, ksize=7)
        gpu_time = time.time() - start_time
        
        print(f"   CPU处理时间: {cpu_time:.4f}s")
        print(f"   GPU处理时间: {gpu_time:.4f}s")
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"   GPU加速比: {speedup:.2f}x")
        else:
            print("   GPU加速比: N/A")
    
    print("\n所有改进版PyTorch功能测试完成！")
    print("主要改进点：")
    print("- Gabor滤波器现在使用完全基于PyTorch的核生成")
    print("- 图像金字塔使用纯PyTorch实现，不再依赖OpenCV")
    print("- 高斯核生成使用纯PyTorch，不再依赖OpenCV")
    print("- 所有操作都支持GPU加速")


if __name__ == "__main__":
    test_improved_torch_functionality()