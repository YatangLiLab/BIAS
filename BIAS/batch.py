import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess
import os

CENTER = '1'
DELTA = '4'
Threshold = 0
GENERATE_TYPE = 'LEVY' # static, dynamic or both, or add _h5, or DoG, or MGF(multiple Gaussian Fitting)
OUTPUT = 'write'
CENTRAL_GAUSSIAN = True
CONTINUTY = True
CHECK_POINT = '2,4,8,16'
motion_center = '1,2,3'
motion_surrounding = '3,4'

# SCRIPT = 'main.py'
if GENERATE_TYPE == 'Flicker':
    SCRIPT = 'main_flicker.py'
    GENERATE_TYPE = 'both'
elif GENERATE_TYPE == 'AdaptWeight':
    SCRIPT = 'main_adaptWeight.py'
    GENERATE_TYPE = 'both'
elif GENERATE_TYPE == 'static':
    SCRIPT = 'pure_image.py'
elif GENERATE_TYPE == 'dynamic':
    SCRIPT = 'pure_motion.py'
elif GENERATE_TYPE == 'both':
    SCRIPT = 'main.py'
elif GENERATE_TYPE == 'Imageboth':
    SCRIPT = 'main_image_input.py'
    GENERATE_TYPE = 'both'
elif GENERATE_TYPE == 'static_h5':
    SCRIPT = 'image_h5.py'
elif GENERATE_TYPE == 'dynamic_h5':
    SCRIPT = 'motion_h5.py'
elif GENERATE_TYPE == 'static_DoG':
    SCRIPT = 'main_DoG.py'
    GENERATE_TYPE = 'static'
elif GENERATE_TYPE == 'Dynamic_DoG':
    SCRIPT = 'main_DoG.py'
    GENERATE_TYPE = 'dynamic'
elif GENERATE_TYPE == 'DoG' or GENERATE_TYPE == 'both_DoG':
    SCRIPT = 'main_DoG.py'
    GENERATE_TYPE = 'both'
elif GENERATE_TYPE == 'MGF':
    SCRIPT = 'multiple_gaussian_fit_main.py'
    GENERATE_TYPE = 'both'
elif GENERATE_TYPE == 'LEVY':
    SCRIPT = 'Levy_main.py'
    GENERATE_TYPE = 'both'
else:
    raise RuntimeError(f'{GENERATE_TYPE} is not available.')
def get_filenames(directory):
    with os.scandir(directory) as entries:
        return [f"{entry.name}" for entry in entries if entry.is_file()]
    
def get_folder_names(directory):
    with os.scandir(directory) as entries:
        return [f"{entry.name}" for entry in entries if entry.is_dir()]
    
def run_script(paths):
    path,anotherpath = paths
    tag = path.split("\\")[-1].split("/")[-1].split(".")[0]
    generate_name = f"{anotherpath}\\{tag}.mp4"
    try:
        subprocess.run(['python',SCRIPT, '--video_path',path,'--generate_name',generate_name,'--generate_type',GENERATE_TYPE,\
                        "--center",f'{CENTER},',"--surrounding",f'{DELTA},',"--output",f'{OUTPUT}', "--selective_threshold",f'{Threshold}'])
                        # , "--add_central_gaussian",f'{CENTRAL_GAUSSIAN}', "--continuity",f'{CONTINUTY}','--default_checkpoint',f'{CHECK_POINT}'\
                        # , "--motion_center",f'{motion_center}', "--motion_surrounding",f'{motion_surrounding}'])
    except Exception as e:
        print(f"\n An Error {e} happened in video {tag}!\n Check it later!\n")

def run_script_image(paths):
    path,anotherpath = paths
    generate_name = f"{anotherpath}"
    try:
        subprocess.run(['python',SCRIPT, '--video_path',path,'--generate_name',generate_name,'--generate_type',GENERATE_TYPE,\
                        "--center",f'{CENTER},',"--surrounding",f'{DELTA},',"--output",f'{OUTPUT}', "--selective_threshold",f'{Threshold}'\
                        , "--add_central_gaussian",f'{CENTRAL_GAUSSIAN}', "--continuity",f'{CONTINUTY}','--default_checkpoint',f'{CHECK_POINT}'\
                        , "--motion_center",f'{motion_center}', "--motion_surrounding",f'{motion_surrounding}'])
    except Exception as e:
        print(f"\n An Error {e} happened in video {generate_name}!\n Check it later!\n")

if __name__ == '__main__':
    # 要处理的数据
    path = 'E:\\your_path\\video\\video\\video'
    another_path = r'E:\YourPath(some logs are changed after generated only for anonymity)\itti_and_lif\Realtime_result\Levy'
    if os.path.exists(f"{another_path}") == False:
        os.mkdir(f"{another_path}")
    data = get_filenames(path)# get_folder_names(path)# 
    another_data = get_filenames(another_path) # get_folder_names(another_path)
    datas = []
    for file in data:
        if int(file.split(".")[0])>700 or int(file.split(".")[0])<600:
            continue
        another_name = file.split(".")[0] + ".mp4"
        if another_name not in another_data:
            datas.append((os.path.join(path,file),another_path))
    print(datas)
    pool_size = 4 if GENERATE_TYPE.find('h5') == -1 else 1
    print("-------------------------------------------------")
    while True:
        print(f'Using {pool_size} processes.')
        print(f"len(data) = {len(datas)}, center = {CENTER}, surrounding = {CENTER} + {DELTA}, generate_type = {GENERATE_TYPE}, script = {SCRIPT}, selective_threshold = {Threshold}")
        a = input("move on?[y/n]\n")
        if a == "n":
            print("cancel.")
            exit()
        elif a == "y":
            print("move on.")
            break
        else:
            print("wrong input!")

    with multiprocessing.Pool(pool_size) as pool:
        results = list(tqdm(pool.imap(run_script, datas), total=len(datas)))
        # results = list(tqdm(pool.imap(run_script_image, datas), total=len(datas)))
    
    warning = np.random.rand(100,100)
    plt.imshow(warning)
    plt.show()
