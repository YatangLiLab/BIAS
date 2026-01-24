#!/usr/bin/env python
"""
性能测试：比较不同高斯金字塔实现方法的速度
"""

import torch
import numpy as np
import time
from BIAS_GPU.utils import TorchImagePyramid

def performance_test():
    print("开始性能测试：比较不同高斯金字塔实现方法")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建测试图像
    test_sizes = [(100, 100), (200, 200), (400, 400)]
    pyramid_processor = TorchImagePyramid(device=device)
    
    for height, width in test_sizes:
        print(f"\n测试图像尺寸: {height}x{width}")
        
        # 创建测试图像
        test_image_np = np.random.rand(height, width).astype(np.float32)
        test_image_torch = torch.from_numpy(test_image_np).to(device)
        
        # 测试PyTorch可分离卷积方法
        torch_sep_times = []
        for _ in range(5):  # 多次测试取平均值
            start_time = time.time()
            result_torch_sep = pyramid_processor.gaussian_pyramid_torch(test_image_torch)
            torch_sep_time = time.time() - start_time
            torch_sep_times.append(torch_sep_time)
        
        avg_torch_sep_time = sum(torch_sep_times) / len(torch_sep_times)
        
        # 测试PyTorch 2D卷积方法
        torch_2d_times = []
        for _ in range(5):  # 多次测试取平均值
            start_time = time.time()
            result_torch_2d = pyramid_processor.gaussian_pyramid_torch_2d(test_image_torch)
            torch_2d_time = time.time() - start_time
            torch_2d_times.append(torch_2d_time)
        
        avg_torch_2d_time = sum(torch_2d_times) / len(torch_2d_times)
        
        # 测试NumPy + OpenCV pyrDown方法
        numpy_cv_times = []
        for _ in range(5):  # 多次测试取平均值
            start_time = time.time()
            result_numpy_cv = pyramid_processor.gaussian_pyramid_numpy_opencv(test_image_torch)
            numpy_cv_time = time.time() - start_time
            numpy_cv_times.append(numpy_cv_time)
        
        avg_numpy_cv_time = sum(numpy_cv_times) / len(numpy_cv_times)
        
        # 输出结果
        print(f"  PyTorch可分离卷积方法: {avg_torch_sep_time:.6f}s")
        print(f"  PyTorch 2D卷积方法: {avg_torch_2d_time:.6f}s")
        print(f"  NumPy+OpenCV pyrDown方法: {avg_numpy_cv_time:.6f}s")
        
        # 比较各种方法的性能
        methods = [
            ("PyTorch可分", avg_torch_sep_time),
            ("PyTorch 2D", avg_torch_2d_time),
            ("NumPy+OpenCV", avg_numpy_cv_time)
        ]
        
        # 找出最快的方法作为基准
        fastest_method = min(methods, key=lambda x: x[1])
        base_time = fastest_method[1]
        
        print(f"  最快方法: {fastest_method[0]} ({base_time:.6f}s)")
        
        for name, time_taken in methods:
            if name != fastest_method[0]:
                speedup = time_taken / base_time
                print(f"  {name}相对慢 {speedup:.2f}x")
        
        # 验证结果相似性（仅在小尺寸时验证，避免内存问题）
        if height <= 200:
            # 将结果移到CPU进行比较
            result_torch_sep_cpu = result_torch_sep.cpu().numpy()
            result_torch_2d_cpu = result_torch_2d.cpu().numpy()
            result_numpy_cv_cpu = result_numpy_cv.cpu().numpy()
            
            # 计算相对误差
            mse_sep_2d = np.mean((result_torch_sep_cpu - result_torch_2d_cpu) ** 2)
            mse_sep_cv = np.mean((result_torch_sep_cpu - result_numpy_cv_cpu) ** 2)
            
            max_val = max(np.max(result_torch_sep_cpu), np.max(result_torch_2d_cpu), np.max(result_numpy_cv_cpu))
            
            relative_error_sep_2d = np.sqrt(mse_sep_2d) / (max_val + 1e-8)
            relative_error_sep_cv = np.sqrt(mse_sep_cv) / (max_val + 1e-8)
            
            print(f"  可分vs2D卷积相对误差 (RMSE/MaxVal): {relative_error_sep_2d:.6f}")
            print(f"  可分vsNumPy+CV相对误差 (RMSE/MaxVal): {relative_error_sep_cv:.6f}")

    print("\n性能测试完成！")

if __name__ == "__main__":
    performance_test()