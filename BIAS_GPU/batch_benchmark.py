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
    
    # 确保frame是cupy数组
    if isinstance(frame, np.ndarray):
        frame = cp.asarray(frame).astype(cp.float16)
    
    # 处理当前帧
    saliency_map, curr_ICs = bias_model.forward(frame, prev_ICs)
    
    # 确保GPU操作完成
    cp.cuda.Stream.null.synchronize()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 返回结果和当前帧的ICs（用于下一次迭代）
    return processing_time, curr_ICs, frame_idx

def process_frame_sequence(args_and_group):
    """
    处理一个帧序列的函数，用于多进程
    序列内部按顺序处理（因为需要前一帧的ICs），但不同序列之间可以并行
    """
    model_args, frame_group = args_and_group
    
    # 每个进程需要重新初始化模型
    bias_model = BIAS(model_args)
    
    processing_times = []
    prev_ICs = None  # 每个组内部从None开始
    
    for frame, frame_idx in frame_group:
        start_time = time.time()
        
        # 确保frame是cupy数组
        if isinstance(frame, np.ndarray):
            frame = cp.asarray(frame).astype(cp.float16)
        
        # 处理当前帧
        saliency_map, curr_ICs = bias_model.forward(frame, prev_ICs)
        
        # 确保GPU操作完成
        cp.cuda.Stream.null.synchronize()
        
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # 更新prev_ICs用于下一帧
        prev_ICs = curr_ICs
    
    # 返回该组的处理时间和最终ICs
    return processing_times, prev_ICs

def simulate_batch_processing(args, batch_size=4, num_frames=20):
    """
    使用多进程模拟批量处理，测试不同batch size的性能
    注意：由于视频显著性算法需要前一帧信息，我们不能完全并行处理连续帧
    因此，我们将整个序列分成多个子序列，每个子序列内部串行处理
    """
    print(f"开始批量处理性能测试，batch_size={batch_size}, num_frames={num_frames}")
    
    # 创建测试帧
    test_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    # 预热GPU
    print("预热GPU...")
    bias_model = BIAS(args)
    dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    dummy_frame = cp.asarray(dummy_frame).astype(cp.float16)  # 转换为cupy数组
    for _ in range(3):
        saliency_map, curr_ICs = bias_model.forward(dummy_frame)
        cp.cuda.Stream.null.synchronize()
    del bias_model  # 删除模型实例，每个进程会重新创建
    
    # 存储处理时间
    processing_times = []
    
    # 将帧分成batch_size个组，每组串行处理
    num_groups = min(batch_size, num_frames)
    frames_per_group = num_frames // num_groups
    remainder = num_frames % num_groups
    
    # 创建帧组
    frame_groups = []
    start_idx = 0
    for i in range(num_groups):
        group_size = frames_per_group + (1 if i < remainder else 0)
        group = test_frames[start_idx:start_idx + group_size]
        frame_groups.append([(frame, start_idx + j) for j, frame in enumerate(group)])
        start_idx += group_size
    
    print(f"分成 {len(frame_groups)} 个组进行并行处理，每个组大小: {[len(g) for g in frame_groups]}")
    
    # 为每个组准备参数
    group_args = [(args, group) for group in frame_groups]
    
    # 使用多进程处理不同的组
    start_total_time = time.time()  # 记录总的处理时间
    with mp.Pool(processes=len(group_args)) as pool:
        # 每个进程处理一个帧序列
        results = pool.map(process_frame_sequence, group_args)
    end_total_time = time.time()
    total_wall_time = end_total_time - start_total_time  # 实际经过的墙钟时间
    
    # 收集结果并打印
    frame_counter = 0
    for group_result in results:
        group_proc_times, group_final_ICs = group_result
        for proc_time in group_proc_times:
            processing_times.append(proc_time)
            print(f"帧 {frame_counter + 1}/{num_frames} 处理时间: {proc_time:.4f} 秒")
            frame_counter += 1
    
    if len(processing_times) == 0:
        print("没有有效的性能数据")
        return None, []
    
    # 计算统计信息
    avg_time = np.mean(processing_times)  # 每帧平均处理时间
    total_time_sum = sum(processing_times)  # 所有帧处理时间之和
    fps_based_on_individual = len(processing_times) / total_time_sum  # 基于单个处理时间的FPS
    fps_based_on_wall_time = len(processing_times) / total_wall_time  # 基于实际经过时间的FPS
    
    print(f"\n=== Batch Size {batch_size} 性能测试结果 ===")
    print(f"总帧数: {len(processing_times)}")
    print(f"各帧处理时间总和: {total_time_sum:.4f} 秒")
    print(f"*实际经过时间 (Wall Time): {total_wall_time:.4f} 秒")
    print(f"平均每帧处理时间: {avg_time:.4f} 秒")
    print(f"处理速度 (基于各帧时间总和): {fps_based_on_individual:.2f} FPS")
    print(f"*处理速度 (基于实际经过时间): {fps_based_on_wall_time:.2f} FPS")
    print(f"*关键指标: 用时 {total_wall_time:.2f} 秒处理了 {len(processing_times)} 帧")
    
    return avg_time, processing_times, total_wall_time

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
    dummy_frame = cp.asarray(dummy_frame).astype(cp.float16)  # 转换为cupy数组
    for _ in range(3):
        saliency_map, curr_ICs = bias_model.forward(dummy_frame)
        cp.cuda.Stream.null.synchronize()
    
    # 存储处理时间
    processing_times = []
    prev_ICs = None
    
    for i, frame in enumerate(test_frames):
        start_time = time.time()
        
        # 确保frame是cupy数组
        if isinstance(frame, np.ndarray):
            frame = cp.asarray(frame).astype(cp.float16)
        
        # 处理当前帧
        saliency_map, curr_ICs = bias_model.forward(frame, prev_ICs)
        
        # 确保GPU操作完成
        cp.cuda.Stream.null.synchronize()
        
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        print(f"帧 {i+1}/{len(test_frames)} 处理时间: {processing_time:.4f} 秒")
        
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
    print(f"各帧处理时间总和: {total_time:.4f} 秒")
    print(f"平均每帧处理时间: {avg_time:.4f} 秒")
    print(f"处理速度: {fps:.2f} FPS")
    
    wall_time = total_time  # 对于顺序处理，wall time就是总处理时间
    return avg_time, processing_times, wall_time

def advanced_batch_simulation(args, batch_sizes=[1, 2, 4, 8], num_frames_per_batch=16):
    """
    测试多种batch size的性能对比
    """
    print("开始多batch size性能对比测试...")
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size == 1:
            # 对于batch_size=1，使用顺序处理作为基准
            avg_time, times, total_wall_time = sequential_processing(args, num_frames_per_batch)
        else:
            # 对于更大的batch_size，使用多进程模拟
            avg_time, times, total_wall_time = simulate_batch_processing(args, batch_size, num_frames_per_batch)
        
        if avg_time is not None:
            results[batch_size] = {
                'avg_time': avg_time,
                'times': times,
                'fps': len(times) / sum(times) if sum(times) > 0 else 0,
                'total_time': sum(times) if times else 0,
                'total_wall_time': total_wall_time
            }
    
    # 打印综合对比结果
    print("\n=== Batch Size 性能对比 ===")
    print(f"{'Batch Size':<10} {'Avg Time (s)':<12} {'FPS':<10} {'Total Wall Time (s)':<20} {'Actual FPS':<12}")
    print("-" * 70)
    
    for batch_size, result in sorted(results.items()):
        # 计算实际FPS（基于wall time）
        actual_fps = len(result['times']) / result['total_wall_time'] if 'total_wall_time' in result else result['fps']
        print(f"{batch_size:<10} {result['avg_time']:<12.4f} {result['fps']:<10.2f} {result['total_wall_time']:<20.4f} {actual_fps:<12.2f}")
    
    print("\n*注意: 'Actual FPS' 基于实际经过时间，更能反映批量处理的真实性能")
    
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
        advanced_batch_simulation(args, [batch_size], num_frames)
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