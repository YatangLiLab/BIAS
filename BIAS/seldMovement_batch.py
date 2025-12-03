import multiprocessing
from tqdm import tqdm
import subprocess
import os

def get_filenames(directory):
    with os.scandir(directory) as entries:
        return [f"{entry.name}" for entry in entries if entry.is_file()]
    
def run_script(paths):
    path,anotherpath = paths
    tag = path.split("\\")[-1].split("/")[-1].split(".")[0]
    generate_name = f"{anotherpath}\\{tag}.mp4"
    generate_type = 'static'
    try:
        subprocess.run(['python','selfMovement_detect.py', '--video_path',path])
    except Exception as e:
        print(f"\n An Error {e} happened in video {tag}!\n Check it later!\n")




if __name__ == '__main__':
    # 要处理的数据
    path = 'E:\\Li Lab\\itti_and_lif\\video\\video\\video'
    another_path = 'E:\\Li Lab\\itti_and_lif\\Realtime_result\\OnlyStatic'
    data = get_filenames(path)
    another_data = get_filenames(another_path)
    datas = []
    for file in data:
        if int(file.split(".")[0])>700 or int(file.split(".")[0])>600:
            continue
        another_name = file.split(".")[0] + ".mp4"
        if int(file.split(".")[0])!= 7:
            continue
        datas.append((f"{path}\\{file}",another_path))
    print(datas)
    print(len(datas))
    a = input("\n -------------------------------------------------\n please enter y to move \n")
    if a != "y":
        print("cancel.")
        raise RuntimeError

    
    # 创建一个进程池，pool_size 为 CPU 核心数
    pool_size = 8#multiprocessing.cpu_count()
    print(f'Using {pool_size} processes.')

    # 创建进程池
    with multiprocessing.Pool(pool_size) as pool:
        # 使用 imap 方法并行执行 intensive_task 函数，并使用 tqdm 显示进度条
        results = list(tqdm(pool.imap(run_script, datas), total=len(datas)))
