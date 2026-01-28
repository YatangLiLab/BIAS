import numpy as np
import cupy as cp
import cv2
import time
import argparse
from cupyx.profiler import benchmark
from model_cupy import BIAS, parse_args

def benchmark_model(args, num_frames=10):
    """
    测试BIAS模型处理单帧所需的平均时间
    """
    # 初始化BIAS模型
    bias_model = BIAS(args)
    
    # 创建模拟输入帧 (320x240 RGB)
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # 预热GPU
    print("预热GPU...")
    for _ in range(3):
        saliency_map, curr_ICs = bias_model.forward(dummy_frame)
        cp.cuda.Stream.null.synchronize()  # 确保GPU操作完成
    
    # 创建测试帧列表
    test_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    print(f"开始测试，处理 {num_frames} 帧...")
    
    # 存储每帧处理时间
    processing_times = []
    prev_ICs = None
    
    # 创建CUDA流用于异步数据传输
    stream = cp.cuda.Stream(non_blocking=True)
    
    for i in range(num_frames):
        frame = test_frames[i]
        
        # 使用cupyx.profiler.benchmark进行精确计时
        # 对于第一帧，由于初始化开销，我们将跳过计时
        if i == 0:
            # 第一帧仅用于初始化，不计入性能测试
            # 预取数据到GPU
            with stream:
                frame_gpu = cp.asarray(frame)
            cp.cuda.Stream.null.synchronize()
            saliency_map, curr_ICs = bias_model.forward(frame_gpu, prev_ICs)
            print(f"帧 {i+1}/{num_frames} (预热帧，不计入统计)...")
        else:
            # 预取当前帧到GPU
            with stream:
                frame_gpu = cp.asarray(frame)
            
            # 为了获取处理时间，我们需要实际调用forward函数
            # 使用一个辅助函数来包装forward调用，以便获取时间
            def timed_forward(frm_gpu, prev_ic):
                return bias_model.forward(frm_gpu, prev_ic)
            
            # 使用cupyx.profiler.benchmark测量实际处理时间
            result = benchmark(timed_forward, (frame_gpu, prev_ICs), n_warmup=0, n_repeat=1)
            processing_time = float(result.gpu_times.mean()) if hasattr(result.gpu_times, 'mean') else result.gpu_times[0]  # 使用GPU时间
            processing_times.append(processing_time)
            
            # 获取新的ICs用于下一帧
            saliency_map, curr_ICs = bias_model.forward(frame_gpu, prev_ICs)
            
            print(f"帧 {i+1}/{num_frames} 处理时间: {processing_time:.4f} 秒")
        
        # 更新prev_ICs用于下一帧的动态显著性计算
        prev_ICs = curr_ICs
    
    if len(processing_times) == 0:
        print("没有有效的性能数据 (至少需要2帧)")
        return None, []
    
    # 计算统计信息
    avg_time = np.mean(processing_times)
    min_time = np.min(processing_times)
    max_time = np.max(processing_times)
    std_time = np.std(processing_times)
    
    print("\n=== 性能测试结果 ===")
    print(f"有效测试帧数: {len(processing_times)} (排除首帧)")
    print(f"平均每帧处理时间: {avg_time:.4f} 秒")
    print(f"最快处理时间: {min_time:.4f} 秒")
    print(f"最慢处理时间: {max_time:.4f} 秒")
    print(f"标准差: {std_time:.4f} 秒")
    print(f"处理速度: {1/avg_time:.2f} FPS")
    
    return avg_time, processing_times

def benchmark_with_real_video(args, video_path, num_frames=10):
    """
    使用真实视频进行基准测试
    """
    # 初始化BIAS模型
    bias_model = BIAS(args)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return None
    
    print(f"使用视频 {video_path} 进行基准测试，处理 {num_frames} 帧...")
    
    # 存储每帧处理时间
    processing_times = []
    prev_ICs = None
    frame_count = 0
    processed_count = 0  # 实际计时的帧数
    
    
    # 创建CUDA流用于异步数据传输
    stream = cp.cuda.Stream(non_blocking=True)
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("视频结束，未能读取足够的帧")
            break
        
        # 确保帧尺寸正确
        frame = cv2.resize(frame, (320, 240))
        
        # 对于第一帧，仅用于初始化，不计入性能测试
        if frame_count == 0:
            # 预取数据到GPU
            with stream:
                frame_gpu = bias_model.preprocessor.process(frame)
            cp.cuda.Stream.null.synchronize()
            saliency_map, curr_ICs = bias_model.forward(frame_gpu, prev_ICs)
            print(f"帧 {frame_count+1}/{num_frames} (预热帧，不计入统计)...")
        else:
            # 预取当前帧到GPU
            with stream:
                frame_gpu = bias_model.preprocessor.process(frame)
            
            # 为了获取处理时间，我们需要实际调用forward函数
            # 使用一个辅助函数来包装forward调用，以便获取时间
            def timed_forward(frm_gpu, prev_ic):
                return bias_model.forward(frm_gpu, prev_ic)
            
            # 使用cupyx.profiler.benchmark测量实际处理时间
            result = benchmark(timed_forward, (frame_gpu, prev_ICs), n_warmup=3, n_repeat=1)
            processing_time = float(result.gpu_times.mean()) if hasattr(result.gpu_times, 'mean') else result.gpu_times[0]  # 使用GPU时间
            processing_times.append(processing_time)
            
            # 获取新的ICs用于下一帧
            saliency_map, curr_ICs = bias_model.forward(frame_gpu, prev_ICs)
            
            print(f"帧 {frame_count+1}/{num_frames} 处理时间: {processing_time:.4f} 秒")
            processed_count += 1
        
        # 更新prev_ICs用于下一帧的动态显著性计算
        prev_ICs = curr_ICs
        
        frame_count += 1
    
    cap.release()
    
    if len(processing_times) == 0:
        print("没有有效的性能数据 (至少需要2帧)")
        return None
    
    # 计算统计信息
    avg_time = np.mean(processing_times)
    min_time = np.min(processing_times)
    max_time = np.max(processing_times)
    std_time = np.std(processing_times)
    
    print("\n=== 真实视频性能测试结果 ===")
    print(f"成功处理帧数: {len(processing_times)} (排除首帧)")
    print(f"平均每帧处理时间: {avg_time:.4f} 秒")
    print(f"最快处理时间: {min_time:.4f} 秒")
    print(f"最慢处理时间: {max_time:.4f} 秒")
    print(f"标准差: {std_time:.4f} 秒")
    print(f"处理速度: {1/avg_time:.2f} FPS")
    
    return avg_time, processing_times

if __name__ == '__main__':
    args = parse_args()
    
    print("选择测试模式:")
    print("1. 随机数据测试")
    print("2. 真实视频测试")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        # 随机数据测试
        num_test_frames = int(input("请输入测试帧数 (默认10): ") or "10")
        benchmark_model(args, num_test_frames)
    elif choice == '2':
        # 真实视频测试
        video_path = input("请输入视频路径 (按回车使用默认路径): ").strip()
        if not video_path:
            video_path = "/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/example_data/demo.AVI"
        
        num_test_frames = int(input("请输入测试帧数 (默认10): ") or "10")
        benchmark_with_real_video(args, video_path, num_test_frames)
    else:
        print("无效选择，使用随机数据测试")
        benchmark_model(args, 10)