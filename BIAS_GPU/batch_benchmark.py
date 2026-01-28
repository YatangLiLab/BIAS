import numpy as np
import cupy as cp
import cv2
import time
import argparse
import multiprocessing as mp
from cupyx.profiler import benchmark
from model_cupy import BIAS, parse_args

def process_single_frame_serial(args_and_frame):
    """
    处理单个帧的函数，用于多进程
    """
    model_args, frame_data, prev_ICs = args_and_frame
    
    # 每个进程需要重新初始化模型
    bias_model = BIAS(model_args)
    
    frame, frame_idx = frame_data
    start_time = time.time()
    
    # 处理当前帧
    saliency_map, curr_ICs = bias_model.forward(frame, prev_ICs)
    
    # 确保GPU操作完成
    cp.cuda.Stream.null.synchronize()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 返回结果和当前帧的ICs（用于下一次迭代）
    return processing_time, curr_ICs, frame_idx

def simulate_batch_processing(args, batch_size=4, num_frames=20):
    """
    使用多进程模拟批量处理，测试不同batch size的性能
    """
    print(f"开始批量处理性能测试，batch_size={batch_size}, num_frames={num_frames}")
    
    # 创建测试帧
    test_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    # 预热GPU
    print("预热GPU...")
    bias_model = BIAS(args)
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    for _ in range(3):
        saliency_map, curr_ICs = bias_model.forward(dummy_frame)
        cp.cuda.Stream.null.synchronize()
    del bias_model  # 删除模型实例，每个进程会重新创建
    
    # 存储处理时间
    processing_times = []
    
    # 模拟连续帧处理，保存上一帧的ICs
    prev_ICs = None
    
    # 分批处理帧
    for i in range(0, num_frames, batch_size):
        batch_frames = []
        current_batch_size = min(batch_size, num_frames - i)
        
        # 准备当前批次的帧数据
        for j in range(current_batch_size):
            if i + j < num_frames:
                batch_frames.append((test_frames[i + j], i + j))
        
        print(f"处理批次 {i//batch_size + 1}, 包含 {len(batch_frames)} 帧...")
        
        # 为当前批次准备参数，第一个帧使用之前的ICs，后续帧基于批次内的顺序
        batch_args_list = []
        for idx, frame_data in enumerate(batch_frames):
            # 第一个帧使用全局的prev_ICs，后续帧使用批次内前一帧的ICs
            frame_prev_ICs = prev_ICs if idx == 0 else None
            batch_args_list.append((args, frame_data, frame_prev_ICs))
        
        # 为当前批次启动多个进程
        with mp.Pool(processes=len(batch_frames)) as pool:
            # 并行处理批次中的帧
            results = pool.map(process_single_frame_serial, batch_args_list)
        
        # 按帧索引排序结果，以维护正确的顺序
        results.sort(key=lambda x: x[2])  # 按frame_idx排序
        
        # 处理结果并更新全局prev_ICs（使用最后一个处理完的帧的ICs）
        for proc_time, curr_ICs, frame_idx in results:
            processing_times.append(proc_time)
            print(f"帧 {frame_idx} 处理时间: {proc_time:.4f} 秒")
            
            # 更新prev_ICs用于下一批次的第一个帧
            if frame_idx == max(idx for _, _, idx in results):  # 最大索引的帧
                prev_ICs = curr_ICs
    
    if len(processing_times) == 0:
        print("没有有效的性能数据")
        return None, []
    
    # 计算统计信息
    avg_time = np.mean(processing_times)
    total_time = sum(processing_times)
    fps = len(processing_times) / total_time
    
    print(f"\n=== Batch Size {batch_size} 性能测试结果 ===")
    print(f"总帧数: {len(processing_times)}")
    print(f"总处理时间: {total_time:.4f} 秒")
    print(f"平均每帧处理时间: {avg_time:.4f} 秒")
    print(f"处理速度: {fps:.2f} FPS")
    
    return avg_time, processing_times

def sequential_processing(args, num_frames=20):
    """
    顺序处理作为对比基准
    """
    print(f"开始顺序处理性能测试，num_frames={num_frames}")
    
    # 初始化BIAS模型
    bias_model = BIAS(args)
    
    # 创建测试帧
    test_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    # 预热GPU
    print("预热GPU...")
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    for _ in range(3):
        saliency_map, curr_ICs = bias_model.forward(dummy_frame)
        cp.cuda.Stream.null.synchronize()
    
    # 存储处理时间
    processing_times = []
    prev_ICs = None
    
    for i, frame in enumerate(test_frames):
        start_time = time.time()
        
        # 处理当前帧
        saliency_map, curr_ICs = bias_model.forward(frame, prev_ICs)
        
        # 确保GPU操作完成
        cp.cuda.Stream.null.synchronize()
        
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        print(f"帧 {i} 处理时间: {processing_time:.4f} 秒")
        
        # 更新prev_ICs用于下一帧的动态显著性计算
        prev_ICs = curr_ICs
    
    if len(processing_times) == 0:
        print("没有有效的性能数据")
        return None, []
    
    # 计算统计信息
    avg_time = np.mean(processing_times)
    total_time = sum(processing_times)
    fps = len(processing_times) / total_time
    
    print(f"\n=== 顺序处理性能测试结果 ===")
    print(f"总帧数: {len(processing_times)}")
    print(f"总处理时间: {total_time:.4f} 秒")
    print(f"平均每帧处理时间: {avg_time:.4f} 秒")
    print(f"处理速度: {fps:.2f} FPS")
    
    return avg_time, processing_times

def advanced_batch_simulation(args, batch_sizes=[1, 2, 4, 8], num_frames_per_batch=16):
    """
    测试多种batch size的性能对比
    """
    print("开始多batch size性能对比测试...")
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size == 1:
            # 对于batch_size=1，使用顺序处理作为基准
            avg_time, times = sequential_processing(args, num_frames_per_batch)
        else:
            # 对于更大的batch_size，使用多进程模拟
            avg_time, times = simulate_batch_processing(args, batch_size, num_frames_per_batch)
        
        if avg_time is not None:
            results[batch_size] = {
                'avg_time': avg_time,
                'times': times,
                'fps': len(times) / sum(times),
                'total_time': sum(times)
            }
    
    # 打印综合对比结果
    print("\n=== Batch Size 性能对比 ===")
    print(f"{'Batch Size':<10} {'Avg Time (s)':<15} {'FPS':<10} {'Total Time (s)':<15}")
    print("-" * 55)
    
    for batch_size, result in sorted(results.items()):
        print(f"{batch_size:<10} {result['avg_time']:<15.4f} {result['fps']:<10.2f} {result['total_time']:<15.4f}")
    
    return results

if __name__ == '__main__':
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    print("批量处理性能测试")
    print("选择测试模式:")
    print("1. 单一batch size测试")
    print("2. 多batch size对比测试")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        batch_size = int(input("请输入batch size (默认4): ") or "4")
        num_frames = int(input("请输入测试帧数 (默认20): ") or "20")
        simulate_batch_processing(args, batch_size, num_frames)
    elif choice == '2':
        batch_sizes_input = input("请输入要测试的batch sizes (用逗号分隔，默认1,2,4): ")
        if batch_sizes_input:
            batch_sizes = [int(x.strip()) for x in batch_sizes_input.split(',')]
        else:
            batch_sizes = [1, 2, 4]
        
        num_frames = int(input("请输入每个batch size的测试帧数 (默认16): ") or "16")
        advanced_batch_simulation(args, batch_sizes, num_frames)
    else:
        print("无效选择，运行多batch size对比测试")
        advanced_batch_simulation(args, [1, 2, 4], 16)